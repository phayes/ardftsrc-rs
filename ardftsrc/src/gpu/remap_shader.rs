use std::fmt::Write as _;
use std::sync::Arc;

use ash::vk;
use realfft::num_complex::Complex;
use vkfft_rs::backend::vulkan::runtime::{VulkanBufferSlice, VulkanComputePipeline};
use vkfft_rs::backend::vulkan::{VulkanDescriptorBinding, VulkanDescriptorType, VulkanShaderSource};
use vkfft_rs::{BufferAccess, BufferRole, DispatchGeometry, ScalarType, WorkgroupSize};

use super::buffer::{GpuBuffer, GpuScalar};
use super::context::GpuContext;
use super::error::GpuError;
use super::fft_program::record_compute_barrier;

const WORKGROUP_SIZE: u32 = 64;

/// The custom spectral-remap compute shader from `gpu_plan.md` section 7: for each output
/// frequency bin, copies (or folds/images, per [`crate::spectral::SpectralPlan`]'s
/// precomputed geometry) the matching input bin(s), applies gain and optional phase, and
/// zeroes everything else -- including forcing the DC/Nyquist bins real, as `realfft`
/// requires.
///
/// This mirrors `SpectralPlan::apply`'s CPU formula exactly (same gain/phase LUTs, computed
/// once on the CPU and uploaded here unchanged), reparameterized as one GPU invocation per
/// `(batch, output bin)` pair instead of a CPU loop over output bins.
pub(crate) struct RemapShader<T> {
    pipeline: VulkanComputePipeline,
    /// Kept alive only because `pipeline`'s bound descriptor set references its buffer;
    /// unused otherwise (never re-uploaded after construction).
    _gain: GpuBuffer<T>,
    _phase: Option<GpuBuffer<T>>,
}

/// Bin-domain geometry the remap shader needs, mirroring
/// [`crate::spectral::FilterGeometry`]/[`crate::spectral::SpectralPlan`].
pub(crate) struct RemapGeometry<'a, T> {
    /// `SpectralPlan`'s `direction == Direction::Up` (input spectrum smaller than output).
    pub(crate) direction_up: bool,
    /// `FilterGeometry::lower_nyquist_bin`.
    pub(crate) n: usize,
    /// `SpectralPlan`'s private `reflect_start_bin`.
    pub(crate) r0: usize,
    /// `2.0` when downsampling (input Nyquist bin folds onto itself), `1.0` otherwise.
    pub(crate) nyquist_fold: f64,
    /// `SpectralPlan::gain` (`upper_nyquist_bin + 1` entries), reused verbatim.
    pub(crate) gain: &'a [T],
    /// `SpectralPlan::phase` (`lower_nyquist_bin + 1` entries) when
    /// `SpectralPlan::phase_enabled`, reused verbatim.
    pub(crate) phase: Option<&'a [Complex<T>]>,
}

fn glsl_scalar_names(scalar: ScalarType) -> Result<(&'static str, &'static str), GpuError> {
    match scalar {
        ScalarType::F32 => Ok(("float", "vec2")),
        ScalarType::F64 => Ok(("double", "dvec2")),
        _ => Err(GpuError::PlanCreationFailed("spectral remap shader only supports f32/f64".to_string())),
    }
}

fn build_glsl<T>(
    scalar: ScalarType,
    src_stride: usize,
    dst_stride: usize,
    dst_len: usize,
    batch_count: usize,
    geometry: &RemapGeometry<'_, T>,
) -> Result<String, GpuError> {
    let (float_ty, cvec_ty) = glsl_scalar_names(scalar)?;
    let phase_enabled = geometry.phase.is_some();
    let n = geometry.n;
    let r0 = geometry.r0;

    let mut glsl = String::new();
    if matches!(scalar, ScalarType::F64) {
        writeln!(glsl, "#version 450\n#extension GL_ARB_gpu_shader_fp64 : require").unwrap();
    } else {
        writeln!(glsl, "#version 450").unwrap();
    }
    writeln!(glsl, "layout(local_size_x = {WORKGROUP_SIZE}) in;").unwrap();
    // `naga`'s validator (used by `VulkanShaderSource::compile_spirv`) rejects a storage
    // buffer with GLSL's `writeonly` qualifier (`StorageAddressSpaceWriteOnlyNotSupported`);
    // `vkfft-rs`'s own generated shaders declare their output buffer as plain read-write
    // `buffer` for the same reason, so this matches that convention.
    writeln!(glsl, "layout(set = 0, binding = 0, std430) readonly buffer SrcBuf {{ {cvec_ty} src[]; }};").unwrap();
    writeln!(glsl, "layout(set = 0, binding = 1, std430) buffer DstBuf {{ {cvec_ty} dst[]; }};").unwrap();
    writeln!(glsl, "layout(set = 0, binding = 2, std430) readonly buffer GainBuf {{ {float_ty} gain[]; }};").unwrap();
    if phase_enabled {
        writeln!(glsl, "layout(set = 0, binding = 3, std430) readonly buffer PhaseBuf {{ {cvec_ty} phase[]; }};").unwrap();
    }
    writeln!(
        glsl,
        r"
{cvec_ty} vkfft_ardftsrc_cmul({cvec_ty} a, {cvec_ty} b) {{
    return {cvec_ty}(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}
{cvec_ty} vkfft_ardftsrc_conj({cvec_ty} a) {{
    return {cvec_ty}(a.x, -a.y);
}}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    uint dst_len = {dst_len}u;
    uint batch_count = {batch_count}u;
    if (idx >= dst_len * batch_count) {{ return; }}
    uint batch = idx / dst_len;
    uint j = idx % dst_len;
    uint src_base = batch * {src_stride}u;
    uint dst_base = batch * {dst_stride}u;
    uint n = {n}u;
    uint r0 = {r0}u;

    {cvec_ty} value = {cvec_ty}(0.0, 0.0);
"
    )
    .unwrap();

    if geometry.direction_up {
        writeln!(
            glsl,
            r"
    if (j <= n) {{
        value = vkfft_ardftsrc_cmul(src[src_base + j], {cvec_ty}(gain[j], 0.0));
        {phase_direct}
    }} else if (j <= 2u * n) {{
        uint k = 2u * n - j;
        if (k >= r0 && k < n) {{
            {cvec_ty} s = src[src_base + k];
            {phase_k}
            value = vkfft_ardftsrc_cmul(vkfft_ardftsrc_conj(s), {cvec_ty}(gain[j], 0.0));
        }}
    }}
",
            phase_direct = if phase_enabled { "value = vkfft_ardftsrc_cmul(value, phase[j]);" } else { "" },
            phase_k = if phase_enabled { "s = vkfft_ardftsrc_cmul(s, phase[k]);" } else { "" },
        )
        .unwrap();
    } else {
        writeln!(
            glsl,
            r"
    if (j < r0) {{
        value = vkfft_ardftsrc_cmul(src[src_base + j], {cvec_ty}(gain[j], 0.0));
        {phase_direct}
    }} else if (j < n) {{
        uint j2 = 2u * n - j;
        {cvec_ty} a = vkfft_ardftsrc_cmul(src[src_base + j], {cvec_ty}(gain[j], 0.0));
        {cvec_ty} b = vkfft_ardftsrc_cmul(vkfft_ardftsrc_conj(src[src_base + j2]), {cvec_ty}(gain[j2], 0.0));
        value = {cvec_ty}(a.x + b.x, a.y + b.y);
        {phase_fold}
    }} else {{
        {cvec_ty} s = src[src_base + n];
        {phase_nyquist}
        value = {cvec_ty}(s.x * gain[n] * {fold}, 0.0);
    }}
",
            phase_direct = if phase_enabled { "value = vkfft_ardftsrc_cmul(value, phase[j]);" } else { "" },
            phase_fold = if phase_enabled { "value = vkfft_ardftsrc_cmul(value, phase[j]);" } else { "" },
            phase_nyquist = if phase_enabled { "s = vkfft_ardftsrc_cmul(s, phase[n]);" } else { "" },
            fold = format!("{:?}", geometry.nyquist_fold),
        )
        .unwrap();
    }

    writeln!(
        glsl,
        r"
    if (j == 0u) {{ value.y = {float_ty}(0.0); }}
    if (j == dst_len - 1u) {{ value.y = {float_ty}(0.0); }}
    dst[dst_base + j] = value;
}}
"
    )
    .unwrap();

    Ok(glsl)
}

impl<T: GpuScalar> RemapShader<T> {
    /// Builds the remap shader for one fixed `(src, dst)` buffer pair and geometry, binding its
    /// descriptor set once; [`RemapShader::record`] only ever records a dispatch afterwards.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        context: &Arc<GpuContext>,
        scalar: ScalarType,
        src: &GpuBuffer<T>,
        src_stride: usize,
        dst: &GpuBuffer<T>,
        dst_stride: usize,
        dst_len: usize,
        batch_count: usize,
        geometry: &RemapGeometry<'_, T>,
    ) -> Result<Self, GpuError> {
        let glsl = build_glsl(scalar, src_stride, dst_stride, dst_len, batch_count, geometry)?;

        let total_invocations = (dst_len * batch_count) as u32;
        let dispatch_x = total_invocations.div_ceil(WORKGROUP_SIZE).max(1);

        let mut descriptors = vec![
            VulkanDescriptorBinding {
                set: 0,
                binding: 0,
                descriptor_type: VulkanDescriptorType::StorageBuffer,
                access: BufferAccess::ReadOnly,
                role: BufferRole::Input,
            },
            VulkanDescriptorBinding {
                set: 0,
                binding: 1,
                descriptor_type: VulkanDescriptorType::StorageBuffer,
                access: BufferAccess::WriteOnly,
                role: BufferRole::Output,
            },
            VulkanDescriptorBinding {
                set: 0,
                binding: 2,
                descriptor_type: VulkanDescriptorType::StorageBuffer,
                access: BufferAccess::ReadOnly,
                role: BufferRole::LookupTable,
            },
        ];
        if geometry.phase.is_some() {
            descriptors.push(VulkanDescriptorBinding {
                set: 0,
                binding: 3,
                descriptor_type: VulkanDescriptorType::StorageBuffer,
                access: BufferAccess::ReadOnly,
                role: BufferRole::LookupTable,
            });
        }

        let shader_source = VulkanShaderSource {
            entry_point: "main",
            glsl,
            scalar,
            sequence_len: dst_len.max(1),
            batch_count: batch_count.max(1),
            workgroup_size: WorkgroupSize { x: WORKGROUP_SIZE, y: 1, z: 1 },
            dispatch: DispatchGeometry { x: dispatch_x, y: 1, z: 1 },
            descriptors,
            required_shared_memory_bytes: 0,
            required_subgroup_size: None,
        };
        let spirv = shader_source
            .compile_spirv()
            .map_err(|err| GpuError::PlanCreationFailed(format!("failed to compile spectral remap shader: {err}")))?;

        // SAFETY: `context.device()` is a live logical device kept alive alongside this
        // `RemapShader` (see the field-ordering discipline documented on `GpuBuffer`); `spirv`
        // was just compiled successfully above.
        let pipeline = unsafe { VulkanComputePipeline::new_with_pipeline_cache(Arc::clone(context.device()), &spirv, context.pipeline_cache()) }
            .map_err(|err| GpuError::PlanCreationFailed(format!("failed to build spectral remap pipeline: {err}")))?;

        let gain_buffer = GpuBuffer::<T>::new(context, geometry.gain.len(), vk::BufferUsageFlags::empty())?;
        gain_buffer.upload(geometry.gain)?;

        let phase_buffer = match geometry.phase {
            Some(phase) => {
                let mut flat = Vec::with_capacity(phase.len() * 2);
                for value in phase {
                    flat.push(value.re);
                    flat.push(value.im);
                }
                let buffer = GpuBuffer::<T>::new(context, flat.len(), vk::BufferUsageFlags::empty())?;
                buffer.upload(&flat)?;
                Some(buffer)
            }
            None => None,
        };

        let mut bindings = vec![
            (0u32, VulkanBufferSlice::whole(src.handle())),
            (1u32, VulkanBufferSlice::whole(dst.handle())),
            (2u32, VulkanBufferSlice::whole(gain_buffer.handle())),
        ];
        if let Some(phase_buffer) = &phase_buffer {
            bindings.push((3u32, VulkanBufferSlice::whole(phase_buffer.handle())));
        }
        // SAFETY: every bound buffer was created above (or passed in, already live) for this
        // exact purpose, and no submission referencing this descriptor set exists yet.
        unsafe { pipeline.update_storage_buffers(&bindings) }
            .map_err(|err| GpuError::PlanCreationFailed(format!("failed to bind spectral remap buffers: {err}")))?;

        Ok(Self {
            pipeline,
            _gain: gain_buffer,
            _phase: phase_buffer,
        })
    }

    /// Total GPU-buffer bytes this shader owns (its gain/phase lookup tables), for sizing a
    /// `GpuCore` ring slot against a memory budget.
    pub(crate) fn total_bytes(&self) -> u64 {
        self._gain.byte_len() + self._phase.as_ref().map_or(0, GpuBuffer::byte_len)
    }

    /// Records this shader's dispatch, followed by a full compute barrier, into
    /// `command_buffer` (which must already be in the recording state).
    pub(crate) fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        // SAFETY: `command_buffer` is in the recording state (caller contract); `self.pipeline`
        // was fully built (pipeline + bound descriptor set) in `build` above.
        unsafe { self.pipeline.record_dispatch(command_buffer) };
        record_compute_barrier(device, command_buffer);
    }
}
