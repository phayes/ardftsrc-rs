use std::sync::Arc;

use vkfft_rs::backend::vulkan::runtime::{VulkanBufferSlice, VulkanComputePipeline};
use vkfft_rs::backend::vulkan::{VulkanDescriptorBinding, VulkanDescriptorType, VulkanShaderSource};
use vkfft_rs::{BufferAccess, BufferRole, DispatchGeometry, ScalarType, WorkgroupSize};

use super::buffer::GpuScalar;
use super::context::GpuContext;
use super::error::GpuError;

const WORKGROUP_SIZE: u32 = 64;

/// Which of `gpu_plan.md` section 9's three overlap/output cases a chunk falls into, mirroring
/// `crate::cpu_core`'s private `TransformMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverlapMode {
    /// Steady-state streaming block: write output from `ifft_first_half + overlap`, then
    /// replace `overlap` with `ifft_second_half`.
    Normal,
    /// Start-edge priming block: output is not emitted; only stage `ifft_second_half` into
    /// `overlap`.
    Start,
    /// Finalize-tail block: accumulate `ifft_first_half` into `overlap` (no output emitted;
    /// the caller reads `overlap` directly afterward, as `CpuCore::finalize` does).
    End,
}

pub(crate) fn glsl_scalar_name(scalar: ScalarType) -> Result<&'static str, GpuError> {
    match scalar {
        ScalarType::F32 => Ok("float"),
        ScalarType::F64 => Ok("double"),
        _ => Err(GpuError::PlanCreationFailed("overlap-add shader only supports f32/f64".to_string())),
    }
}

/// Builds one mode's GLSL source. `ifft_stride` is complex elements per batch row in the
/// (complex-packed) inverse-FFT output buffer; only the real component of each sample is read.
pub(crate) fn build_glsl(
    scalar: ScalarType,
    mode: OverlapMode,
    ifft_stride: usize,
    output_chunk_frames: usize,
    input_chunk_frames: usize,
    batch_count: usize,
) -> Result<String, GpuError> {
    let float_ty = glsl_scalar_name(scalar)?;
    let body = match mode {
        OverlapMode::Normal => {
            r"
    float_t first_half = ifft_in[ifft_base + i * 2u];
    float_t second_half = ifft_in[ifft_base + (output_chunk_frames + i) * 2u];
    float_t scale = float_t(output_chunk_frames) / float_t(input_chunk_frames);
    out_buf[idx] = (first_half + overlap[idx]) * scale;
    overlap[idx] = second_half;
"
        }
        OverlapMode::Start => {
            r"
    float_t second_half = ifft_in[ifft_base + (output_chunk_frames + i) * 2u];
    overlap[idx] = second_half;
    out_buf[idx] = float_t(0.0);
"
        }
        OverlapMode::End => {
            r"
    float_t first_half = ifft_in[ifft_base + i * 2u];
    overlap[idx] = overlap[idx] + first_half;
    out_buf[idx] = float_t(0.0);
"
        }
    };
    let body = body.replace("float_t", float_ty);

    let mut glsl = String::new();
    if matches!(scalar, ScalarType::F64) {
        glsl.push_str("#version 450\n#extension GL_ARB_gpu_shader_fp64 : require\n");
    } else {
        glsl.push_str("#version 450\n");
    }
    glsl.push_str(&format!("layout(local_size_x = {WORKGROUP_SIZE}) in;\n"));
    glsl.push_str(&format!(
        "layout(set = 0, binding = 0, std430) readonly buffer IfftBuf {{ {float_ty} ifft_in[]; }};\n"
    ));
    glsl.push_str(&format!("layout(set = 0, binding = 1, std430) buffer OutBuf {{ {float_ty} out_buf[]; }};\n"));
    glsl.push_str(&format!("layout(set = 0, binding = 2, std430) buffer OverlapBuf {{ {float_ty} overlap[]; }};\n"));
    glsl.push_str(&format!(
        r"
void main() {{
    uint idx = gl_GlobalInvocationID.x;
    uint output_chunk_frames = {output_chunk_frames}u;
    uint input_chunk_frames = {input_chunk_frames}u;
    uint batch_count = {batch_count}u;
    if (idx >= output_chunk_frames * batch_count) {{ return; }}
    uint batch = idx / output_chunk_frames;
    uint i = idx % output_chunk_frames;
    uint ifft_base = batch * {ifft_stride}u * 2u;
{body}
}}
"
    ));
    Ok(glsl)
}

/// Builds one overlap-add pipeline bound to explicit buffer slices -- byte-range windows into a
/// larger shared buffer (`super::batch_overlap_shader::BatchOverlapShader`, which packs every
/// window's forward/inverse FFT batch contiguously and slices this shader's `ifft_in` binding
/// into the right window's region).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pipeline_with_slices<T: GpuScalar>(
    context: &Arc<GpuContext>,
    scalar: ScalarType,
    mode: OverlapMode,
    ifft_slice: VulkanBufferSlice,
    ifft_stride: usize,
    output_chunk_frames: usize,
    input_chunk_frames: usize,
    batch_count: usize,
    output_slice: VulkanBufferSlice,
    overlap_slice: VulkanBufferSlice,
) -> Result<VulkanComputePipeline, GpuError> {
    let glsl = build_glsl(scalar, mode, ifft_stride, output_chunk_frames, input_chunk_frames, batch_count)?;
    let total_invocations = (output_chunk_frames * batch_count) as u32;
    let dispatch_x = total_invocations.div_ceil(WORKGROUP_SIZE).max(1);

    let descriptors = vec![
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
            access: BufferAccess::ReadWrite,
            role: BufferRole::Auxiliary,
        },
    ];

    let shader_source = VulkanShaderSource {
        entry_point: "main",
        glsl,
        scalar,
        sequence_len: output_chunk_frames.max(1),
        batch_count: batch_count.max(1),
        workgroup_size: WorkgroupSize { x: WORKGROUP_SIZE, y: 1, z: 1 },
        dispatch: DispatchGeometry { x: dispatch_x, y: 1, z: 1 },
        descriptors,
        required_shared_memory_bytes: 0,
        required_subgroup_size: None,
    };
    let spirv = shader_source
        .compile_spirv()
        .map_err(|err| GpuError::PlanCreationFailed(format!("failed to compile overlap-add shader ({mode:?}): {err}")))?;

    // SAFETY: `context.device()` is a live logical device kept alive alongside this pipeline
    // (see the field-ordering discipline documented on `GpuBuffer`); `spirv` was just compiled
    // successfully above.
    let pipeline = unsafe { VulkanComputePipeline::new_with_pipeline_cache(Arc::clone(context.device()), &spirv, context.pipeline_cache()) }
        .map_err(|err| GpuError::PlanCreationFailed(format!("failed to build overlap-add pipeline ({mode:?}): {err}")))?;

    let bindings = [(0u32, ifft_slice), (1u32, output_slice), (2u32, overlap_slice)];
    // SAFETY: every bound buffer is live and sized for this exact purpose, and no submission
    // referencing this descriptor set exists yet.
    unsafe { pipeline.update_storage_buffers(&bindings) }
        .map_err(|err| GpuError::PlanCreationFailed(format!("failed to bind overlap-add buffers ({mode:?}): {err}")))?;

    Ok(pipeline)
}
