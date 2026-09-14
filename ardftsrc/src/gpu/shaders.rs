use std::sync::Arc;

use num_traits::Float;
use vkfft_rs::backend::vulkan::{VulkanDescriptorBinding, VulkanDescriptorType, VulkanSpirvShader};
use vkfft_rs::{BufferAccess, BufferRole, DispatchGeometry, RealFftKind, ScalarType, WorkgroupSize};

use crate::config::DerivedConfig;

use super::buffer::GpuScalar;
use super::context::{GpuDevice, GpuDeviceId, GpuDeviceType, GpuInfo, GpuPipelineCacheId};
use super::error::GpuError;
use super::fft_program::{FromF64, compile_fft};
use super::overlap_shader::{OverlapMode, compile_overlap_shader};
use super::remap_shader::{RemapGeometry, compile_remap_shader};

const MAGIC: &[u8; 8] = b"ARDFTGSH";
const ARCHIVE_VERSION: u32 = 2;
const SHADER_ABI_VERSION: u32 = 1;
const PLANNER_COMMIT_BYTES: usize = 40;
const HEADER_LEN: usize = 8 + 4 + 8 + 8;
const MAX_ARCHIVE_BYTES: usize = 256 * 1024 * 1024;

/// Serializable compiled GPU shaders for one device and resampling geometry.
///
/// This contains SPIR-V and Vulkan's opaque pipeline-cache bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuShaders {
    archive_version: u32,
    shader_abi_version: u32,
    planner_commit: [u8; PLANNER_COMMIT_BYTES],
    gpu_info: GpuInfo,
    geometry: GpuShaderGeometry,
    pub(crate) single: TransformShaders,
    pub(crate) grouped: TransformShaders,
    pipeline_cache: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuShaderGeometry {
    scalar: u8,
    channels: usize,
    group_chunks: usize,
    input_fft_size: usize,
    output_fft_size: usize,
    input_chunk_frames: usize,
    output_chunk_frames: usize,
    direction_up: bool,
    lower_nyquist_bin: usize,
    reflect_start_bin: usize,
    nyquist_fold_bits: u64,
    phase_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransformShaders {
    pub(crate) batch_count: usize,
    pub(crate) forward_fft: Vec<CompiledShader>,
    pub(crate) remap: CompiledShader,
    pub(crate) inverse_fft: Vec<CompiledShader>,
    pub(crate) overlap: OverlapShaders,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OverlapShaders {
    start: CompiledShader,
    normal: CompiledShader,
    end: CompiledShader,
}

impl OverlapShaders {
    pub(crate) fn get(&self, mode: OverlapMode) -> &VulkanSpirvShader {
        match mode {
            OverlapMode::Start => self.start.as_spirv(),
            OverlapMode::Normal => self.normal.as_spirv(),
            OverlapMode::End => self.end.as_spirv(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledShader(VulkanSpirvShader);

impl CompiledShader {
    pub(crate) fn as_spirv(&self) -> &VulkanSpirvShader {
        &self.0
    }
}

impl GpuShaders {
    pub(crate) fn compile<T>(
        device: &Arc<GpuDevice>,
        derived: &DerivedConfig<T>,
        channels: usize,
        group_chunks: usize,
    ) -> Result<Self, GpuError>
    where
        T: Float + GpuScalar + FromF64,
    {
        let geometry = GpuShaderGeometry::new::<T>(derived, channels, group_chunks);
        let single = compile_transform::<T>(device, derived, channels, channels)?;
        let grouped_batch = channels
            .checked_mul(group_chunks)
            .ok_or_else(|| GpuError::InvalidConfig("GPU shader batch count overflow".to_string()))?;
        let grouped = if group_chunks == 1 {
            single.clone()
        } else {
            compile_transform::<T>(device, derived, grouped_batch, channels)?
        };
        Ok(Self {
            archive_version: ARCHIVE_VERSION,
            shader_abi_version: SHADER_ABI_VERSION,
            planner_commit: current_planner_commit(),
            gpu_info: device.info().clone(),
            geometry,
            single,
            grouped,
            pipeline_cache: device.pipeline_cache_data()?,
        })
    }

    pub(crate) fn validate<T>(
        &self,
        device: &GpuDevice,
        derived: &DerivedConfig<T>,
        channels: usize,
        group_chunks: usize,
    ) -> Result<(), GpuError>
    where
        T: Float + GpuScalar,
    {
        if self.archive_version != ARCHIVE_VERSION {
            return Err(incompatible("archive version"));
        }
        if self.shader_abi_version != SHADER_ABI_VERSION {
            return Err(incompatible("shader ABI version"));
        }
        if self.planner_commit != current_planner_commit() {
            return Err(incompatible("vkfft-rs planner version"));
        }
        if self.gpu_info.id != device.info().id {
            return Err(incompatible("target GPU"));
        }
        if self.gpu_info.pipeline_cache_id != device.info().pipeline_cache_id {
            return Err(incompatible("pipeline-cache identity"));
        }
        if self.geometry != GpuShaderGeometry::new::<T>(derived, channels, group_chunks) {
            return Err(incompatible("resampling geometry"));
        }
        if self.single.batch_count != channels
            || self.grouped.batch_count
                != channels
                    .checked_mul(group_chunks)
                    .ok_or_else(|| GpuError::InvalidConfig("GPU shader batch count overflow".to_string()))?
        {
            return Err(incompatible("batch geometry"));
        }
        self.single.validate(
            scalar_from_tag(self.geometry.scalar)?,
            channels,
            self.geometry.output_chunk_frames,
        )?;
        self.grouped.validate(
            scalar_from_tag(self.geometry.scalar)?,
            channels,
            self.geometry.output_chunk_frames,
        )?;
        Ok(())
    }

    pub(crate) fn pipeline_cache(&self) -> &[u8] {
        &self.pipeline_cache
    }

    pub(crate) fn with_pipeline_cache(mut self, pipeline_cache: Vec<u8>) -> Self {
        self.pipeline_cache = pipeline_cache;
        self
    }

    /// Information for the device against which these shaders were compiled.
    pub fn gpu_info(&self) -> &GpuInfo {
        &self.gpu_info
    }

    /// Number of chunks compiled into each grouped dispatch.
    pub fn group_chunks(&self) -> usize {
        self.geometry.group_chunks
    }

    /// Encodes this artifact into its canonical versioned binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut payload = Writer::default();
        payload.u32(self.shader_abi_version);
        payload.raw(&self.planner_commit);
        encode_gpu_info(&mut payload, &self.gpu_info);
        self.geometry.encode(&mut payload);
        self.single.encode(&mut payload);
        self.grouped.encode(&mut payload);
        payload.bytes(&self.pipeline_cache);
        let checksum = checksum(&payload.0);
        let mut out = Vec::with_capacity(HEADER_LEN + payload.0.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&ARCHIVE_VERSION.to_le_bytes());
        out.extend_from_slice(&(payload.0.len() as u64).to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&payload.0);
        out
    }

    /// Decodes and structurally validates a compiled shader artifact.
    ///
    /// # Safety
    ///
    /// `bytes` must have been produced by [`GpuShaders::to_bytes`] from a trusted
    /// [`GpuShaders`] value and must not have been modified afterward. The archive contains
    /// executable SPIR-V and opaque Vulkan pipeline-cache data. Its checksum detects accidental
    /// corruption but is not an authenticity check, and structural decoding does not prove that
    /// the SPIR-V obeys its declared descriptor, dispatch, or buffer-bounds contract. Supplying
    /// attacker-controlled or otherwise untrusted bytes can therefore violate Vulkan's safety
    /// requirements when the artifact is loaded and executed.
    pub unsafe fn from_bytes(bytes: &[u8]) -> Result<Self, GpuError> {
        Self::decode(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self, GpuError> {
        if bytes.len() < HEADER_LEN || bytes.len() > MAX_ARCHIVE_BYTES {
            return Err(archive_error("archive size is invalid"));
        }
        if &bytes[..8] != MAGIC {
            return Err(archive_error("magic does not match"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != ARCHIVE_VERSION {
            return Err(archive_error("archive version is unsupported"));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(bytes[12..20].try_into().unwrap()))
            .map_err(|_| archive_error("payload length does not fit this platform"))?;
        if HEADER_LEN.checked_add(payload_len) != Some(bytes.len()) {
            return Err(archive_error("payload length does not match"));
        }
        let expected_checksum = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
        if checksum(&bytes[HEADER_LEN..]) != expected_checksum {
            return Err(archive_error("checksum does not match"));
        }
        let mut reader = Reader::new(&bytes[HEADER_LEN..]);
        let shader_abi_version = reader.u32()?;
        let mut planner_commit = [0; PLANNER_COMMIT_BYTES];
        reader.raw(&mut planner_commit)?;
        let gpu_info = decode_gpu_info(&mut reader)?;
        let geometry = GpuShaderGeometry::decode(&mut reader)?;
        let single = TransformShaders::decode(&mut reader)?;
        let grouped = TransformShaders::decode(&mut reader)?;
        let pipeline_cache = reader.bytes()?;
        reader.finish()?;
        Ok(Self {
            archive_version: version,
            shader_abi_version,
            planner_commit,
            gpu_info,
            geometry,
            single,
            grouped,
            pipeline_cache,
        })
    }
}

fn compile_transform<T>(
    device: &Arc<GpuDevice>,
    derived: &DerivedConfig<T>,
    batch_count: usize,
    channels: usize,
) -> Result<TransformShaders, GpuError>
where
    T: Float + GpuScalar + FromF64,
{
    let forward = compile_fft(
        device,
        RealFftKind::RealToComplex,
        derived.input_fft_size,
        batch_count,
        T::precision(),
        false,
    )?;
    let inverse = compile_fft(
        device,
        RealFftKind::ComplexToReal,
        derived.output_fft_size,
        batch_count,
        T::precision(),
        true,
    )?;
    let remap_geometry = remap_geometry(derived);
    let remap = compile_remap_shader(
        T::scalar_type(),
        forward.output_stride,
        inverse.input_stride,
        derived.output_fft_size / 2 + 1,
        batch_count,
        &remap_geometry,
    )?;
    let overlap = OverlapShaders {
        start: CompiledShader(compile_overlap_shader(
            T::scalar_type(),
            OverlapMode::Start,
            inverse.output_stride,
            derived.output_chunk_frames,
            derived.input_chunk_frames,
            channels,
        )?),
        normal: CompiledShader(compile_overlap_shader(
            T::scalar_type(),
            OverlapMode::Normal,
            inverse.output_stride,
            derived.output_chunk_frames,
            derived.input_chunk_frames,
            channels,
        )?),
        end: CompiledShader(compile_overlap_shader(
            T::scalar_type(),
            OverlapMode::End,
            inverse.output_stride,
            derived.output_chunk_frames,
            derived.input_chunk_frames,
            channels,
        )?),
    };
    Ok(TransformShaders {
        batch_count,
        forward_fft: forward.shaders.into_iter().map(CompiledShader).collect(),
        remap: CompiledShader(remap),
        inverse_fft: inverse.shaders.into_iter().map(CompiledShader).collect(),
        overlap,
    })
}

pub(crate) fn remap_geometry<T>(derived: &DerivedConfig<T>) -> RemapGeometry<'_, T> {
    RemapGeometry {
        direction_up: derived.input_chunk_frames < derived.output_chunk_frames,
        n: derived.spectral.geometry.lower_nyquist_bin,
        r0: derived.spectral.geometry.reflect_start_bin(),
        nyquist_fold: if derived.input_chunk_frames > derived.output_chunk_frames {
            2.0
        } else {
            1.0
        },
        gain: &derived.spectral.gain,
        phase: derived
            .spectral
            .phase_enabled
            .then_some(derived.spectral.phase.as_slice()),
    }
}

impl GpuShaderGeometry {
    fn new<T: Float + GpuScalar>(derived: &DerivedConfig<T>, channels: usize, group_chunks: usize) -> Self {
        Self {
            scalar: scalar_tag(T::scalar_type()),
            channels,
            group_chunks,
            input_fft_size: derived.input_fft_size,
            output_fft_size: derived.output_fft_size,
            input_chunk_frames: derived.input_chunk_frames,
            output_chunk_frames: derived.output_chunk_frames,
            direction_up: derived.input_chunk_frames < derived.output_chunk_frames,
            lower_nyquist_bin: derived.spectral.geometry.lower_nyquist_bin,
            reflect_start_bin: derived.spectral.geometry.reflect_start_bin(),
            nyquist_fold_bits: if derived.input_chunk_frames > derived.output_chunk_frames {
                2.0f64.to_bits()
            } else {
                1.0f64.to_bits()
            },
            phase_enabled: derived.spectral.phase_enabled,
        }
    }

    fn encode(&self, writer: &mut Writer) {
        writer.u8(self.scalar);
        writer.usize(self.channels);
        writer.usize(self.group_chunks);
        writer.usize(self.input_fft_size);
        writer.usize(self.output_fft_size);
        writer.usize(self.input_chunk_frames);
        writer.usize(self.output_chunk_frames);
        writer.bool(self.direction_up);
        writer.usize(self.lower_nyquist_bin);
        writer.usize(self.reflect_start_bin);
        writer.u64(self.nyquist_fold_bits);
        writer.bool(self.phase_enabled);
    }

    fn decode(reader: &mut Reader<'_>) -> Result<Self, GpuError> {
        let scalar = reader.u8()?;
        scalar_from_tag(scalar)?;
        Ok(Self {
            scalar,
            channels: reader.usize()?,
            group_chunks: reader.usize()?,
            input_fft_size: reader.usize()?,
            output_fft_size: reader.usize()?,
            input_chunk_frames: reader.usize()?,
            output_chunk_frames: reader.usize()?,
            direction_up: reader.bool()?,
            lower_nyquist_bin: reader.usize()?,
            reflect_start_bin: reader.usize()?,
            nyquist_fold_bits: reader.u64()?,
            phase_enabled: reader.bool()?,
        })
    }
}

impl TransformShaders {
    fn validate(
        &self,
        scalar: ScalarType,
        overlap_batch_count: usize,
        output_chunk_frames: usize,
    ) -> Result<(), GpuError> {
        if self.forward_fft.is_empty() || self.inverse_fft.is_empty() {
            return Err(incompatible("FFT shader pass list"));
        }
        if self
            .forward_fft
            .iter()
            .chain(&self.inverse_fft)
            .any(|shader| shader.0.scalar != scalar)
            || self.remap.0.scalar != scalar
        {
            return Err(incompatible("shader scalar type"));
        }
        if self.remap.0.batch_count != self.batch_count {
            return Err(incompatible("remap batch count"));
        }
        for shader in [&self.overlap.start, &self.overlap.normal, &self.overlap.end] {
            if shader.0.scalar != scalar
                || shader.0.batch_count != overlap_batch_count
                || shader.0.sequence_len != output_chunk_frames
            {
                return Err(incompatible("overlap shader geometry"));
            }
        }
        Ok(())
    }

    fn encode(&self, writer: &mut Writer) {
        writer.usize(self.batch_count);
        writer.usize(self.forward_fft.len());
        for shader in &self.forward_fft {
            shader.encode(writer);
        }
        self.remap.encode(writer);
        writer.usize(self.inverse_fft.len());
        for shader in &self.inverse_fft {
            shader.encode(writer);
        }
        self.overlap.encode(writer);
    }

    fn decode(reader: &mut Reader<'_>) -> Result<Self, GpuError> {
        let batch_count = reader.usize()?;
        let forward_len = reader.bounded_len(16_384)?;
        let mut forward_fft = Vec::with_capacity(forward_len);
        for _ in 0..forward_len {
            forward_fft.push(CompiledShader::decode(reader)?);
        }
        let remap = CompiledShader::decode(reader)?;
        let inverse_len = reader.bounded_len(16_384)?;
        let mut inverse_fft = Vec::with_capacity(inverse_len);
        for _ in 0..inverse_len {
            inverse_fft.push(CompiledShader::decode(reader)?);
        }
        Ok(Self {
            batch_count,
            forward_fft,
            remap,
            inverse_fft,
            overlap: OverlapShaders::decode(reader)?,
        })
    }
}

impl OverlapShaders {
    fn encode(&self, writer: &mut Writer) {
        self.start.encode(writer);
        self.normal.encode(writer);
        self.end.encode(writer);
    }

    fn decode(reader: &mut Reader<'_>) -> Result<Self, GpuError> {
        Ok(Self {
            start: CompiledShader::decode(reader)?,
            normal: CompiledShader::decode(reader)?,
            end: CompiledShader::decode(reader)?,
        })
    }
}

impl CompiledShader {
    fn encode(&self, writer: &mut Writer) {
        let shader = &self.0;
        writer.string(shader.entry_point);
        writer.u8(scalar_tag(shader.scalar));
        writer.usize(shader.sequence_len);
        writer.usize(shader.batch_count);
        writer.u32(shader.workgroup_size.x);
        writer.u32(shader.workgroup_size.y);
        writer.u32(shader.workgroup_size.z);
        writer.u32(shader.dispatch.x);
        writer.u32(shader.dispatch.y);
        writer.u32(shader.dispatch.z);
        writer.usize(shader.required_shared_memory_bytes);
        writer.option_u32(shader.required_subgroup_size);
        writer.bool(shader.require_full_subgroups);
        writer.usize(shader.descriptors.len());
        for descriptor in &shader.descriptors {
            writer.u32(descriptor.set);
            writer.u32(descriptor.binding);
            writer.u8(access_tag(descriptor.access));
            writer.u8(role_tag(descriptor.role));
        }
        writer.usize(shader.words.len());
        for &word in &shader.words {
            writer.u32(word);
        }
    }

    fn decode(reader: &mut Reader<'_>) -> Result<Self, GpuError> {
        if reader.string()? != "main" {
            return Err(archive_error("unsupported shader entry point"));
        }
        let scalar = scalar_from_tag(reader.u8()?)?;
        let sequence_len = reader.usize()?;
        let batch_count = reader.usize()?;
        let workgroup_size = WorkgroupSize {
            x: reader.u32()?,
            y: reader.u32()?,
            z: reader.u32()?,
        };
        let dispatch = DispatchGeometry {
            x: reader.u32()?,
            y: reader.u32()?,
            z: reader.u32()?,
        };
        let required_shared_memory_bytes = reader.usize()?;
        let required_subgroup_size = reader.option_u32()?;
        let require_full_subgroups = reader.bool()?;
        let descriptor_len = reader.bounded_len(1_024)?;
        let mut descriptors = Vec::with_capacity(descriptor_len);
        for _ in 0..descriptor_len {
            descriptors.push(VulkanDescriptorBinding {
                set: reader.u32()?,
                binding: reader.u32()?,
                descriptor_type: VulkanDescriptorType::StorageBuffer,
                access: access_from_tag(reader.u8()?)?,
                role: role_from_tag(reader.u8()?)?,
            });
        }
        let words_len = reader.bounded_len(MAX_ARCHIVE_BYTES / 4)?;
        let mut words = Vec::with_capacity(words_len);
        for _ in 0..words_len {
            words.push(reader.u32()?);
        }
        if words.first() != Some(&0x0723_0203) {
            return Err(archive_error("shader does not contain SPIR-V"));
        }
        Ok(Self(VulkanSpirvShader {
            entry_point: "main",
            words,
            scalar,
            sequence_len,
            batch_count,
            workgroup_size,
            dispatch,
            descriptors,
            required_shared_memory_bytes,
            required_subgroup_size,
            require_full_subgroups,
        }))
    }
}

fn encode_gpu_info(writer: &mut Writer, info: &GpuInfo) {
    writer.raw(&info.id.device_uuid);
    writer.u32(info.pipeline_cache_id.vendor_id);
    writer.u32(info.pipeline_cache_id.device_id);
    writer.raw(&info.pipeline_cache_id.pipeline_cache_uuid);
    writer.u8(device_type_tag(info.device_type));
    writer.u32(info.api_version);
    writer.u32(info.driver_version);
    writer.bool(info.shader_float64);
    writer.u64(info.max_storage_buffer_range);
    for value in info.max_compute_workgroup_count {
        writer.u32(value);
    }
    for value in info.max_compute_workgroup_size {
        writer.u32(value);
    }
    writer.u32(info.max_compute_work_group_invocations);
    writer.string(&info.device_name);
}

fn decode_gpu_info(reader: &mut Reader<'_>) -> Result<GpuInfo, GpuError> {
    let mut device_uuid = [0; 16];
    reader.raw(&mut device_uuid)?;
    let vendor_id = reader.u32()?;
    let device_id = reader.u32()?;
    let mut pipeline_cache_uuid = [0; 16];
    reader.raw(&mut pipeline_cache_uuid)?;
    Ok(GpuInfo {
        id: GpuDeviceId { device_uuid },
        pipeline_cache_id: GpuPipelineCacheId {
            vendor_id,
            device_id,
            pipeline_cache_uuid,
        },
        device_type: device_type_from_tag(reader.u8()?)?,
        api_version: reader.u32()?,
        driver_version: reader.u32()?,
        shader_float64: reader.bool()?,
        max_storage_buffer_range: reader.u64()?,
        max_compute_workgroup_count: [reader.u32()?, reader.u32()?, reader.u32()?],
        max_compute_workgroup_size: [reader.u32()?, reader.u32()?, reader.u32()?],
        max_compute_work_group_invocations: reader.u32()?,
        device_name: reader.string()?,
    })
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn raw(&mut self, value: &[u8]) {
        self.0.extend_from_slice(value);
    }
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }
    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }
    fn bytes(&mut self, value: &[u8]) {
        self.usize(value.len());
        self.raw(value);
    }
    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }
    fn option_u32(&mut self, value: Option<u32>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.u32(value);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], GpuError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| archive_error("length overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| archive_error("archive is truncated"))?;
        self.offset = end;
        Ok(value)
    }
    fn raw(&mut self, out: &mut [u8]) -> Result<(), GpuError> {
        out.copy_from_slice(self.take(out.len())?);
        Ok(())
    }
    fn u8(&mut self) -> Result<u8, GpuError> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Result<bool, GpuError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(archive_error("boolean is malformed")),
        }
    }
    fn u32(&mut self) -> Result<u32, GpuError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GpuError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, GpuError> {
        usize::try_from(self.u64()?).map_err(|_| archive_error("integer does not fit this platform"))
    }
    fn bounded_len(&mut self, max: usize) -> Result<usize, GpuError> {
        let value = self.usize()?;
        if value > max {
            Err(archive_error("collection length exceeds archive limit"))
        } else {
            Ok(value)
        }
    }
    fn bytes(&mut self) -> Result<Vec<u8>, GpuError> {
        let len = self.bounded_len(MAX_ARCHIVE_BYTES)?;
        Ok(self.take(len)?.to_vec())
    }
    fn string(&mut self) -> Result<String, GpuError> {
        String::from_utf8(self.bytes()?).map_err(|_| archive_error("string is not UTF-8"))
    }
    fn option_u32(&mut self) -> Result<Option<u32>, GpuError> {
        if self.bool()? { Ok(Some(self.u32()?)) } else { Ok(None) }
    }
    fn finish(self) -> Result<(), GpuError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(archive_error("archive contains trailing data"))
        }
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn current_planner_commit() -> [u8; PLANNER_COMMIT_BYTES] {
    vkfft_rs::UPSTREAM_VKFFT_COMMIT
        .as_bytes()
        .try_into()
        .expect("vkfft-rs upstream commit must be a 40-byte hexadecimal SHA-1")
}

fn scalar_tag(value: ScalarType) -> u8 {
    match value {
        ScalarType::F16 => 0,
        ScalarType::F32 => 1,
        ScalarType::F64 => 2,
        ScalarType::DoubleDouble => 3,
    }
}
fn scalar_from_tag(value: u8) -> Result<ScalarType, GpuError> {
    match value {
        0 => Ok(ScalarType::F16),
        1 => Ok(ScalarType::F32),
        2 => Ok(ScalarType::F64),
        3 => Ok(ScalarType::DoubleDouble),
        _ => Err(archive_error("scalar tag is invalid")),
    }
}
fn access_tag(value: BufferAccess) -> u8 {
    match value {
        BufferAccess::ReadOnly => 0,
        BufferAccess::WriteOnly => 1,
        BufferAccess::ReadWrite => 2,
    }
}
fn access_from_tag(value: u8) -> Result<BufferAccess, GpuError> {
    match value {
        0 => Ok(BufferAccess::ReadOnly),
        1 => Ok(BufferAccess::WriteOnly),
        2 => Ok(BufferAccess::ReadWrite),
        _ => Err(archive_error("buffer access tag is invalid")),
    }
}
fn role_tag(value: BufferRole) -> u8 {
    match value {
        BufferRole::Input => 0,
        BufferRole::Output => 1,
        BufferRole::LookupTable => 2,
        BufferRole::TwiddleLookupTable => 3,
        BufferRole::Auxiliary => 4,
    }
}
fn role_from_tag(value: u8) -> Result<BufferRole, GpuError> {
    match value {
        0 => Ok(BufferRole::Input),
        1 => Ok(BufferRole::Output),
        2 => Ok(BufferRole::LookupTable),
        3 => Ok(BufferRole::TwiddleLookupTable),
        4 => Ok(BufferRole::Auxiliary),
        _ => Err(archive_error("buffer role tag is invalid")),
    }
}
fn device_type_tag(value: GpuDeviceType) -> u8 {
    match value {
        GpuDeviceType::Other => 0,
        GpuDeviceType::Integrated => 1,
        GpuDeviceType::Discrete => 2,
        GpuDeviceType::Virtual => 3,
        GpuDeviceType::Cpu => 4,
    }
}
fn device_type_from_tag(value: u8) -> Result<GpuDeviceType, GpuError> {
    match value {
        0 => Ok(GpuDeviceType::Other),
        1 => Ok(GpuDeviceType::Integrated),
        2 => Ok(GpuDeviceType::Discrete),
        3 => Ok(GpuDeviceType::Virtual),
        4 => Ok(GpuDeviceType::Cpu),
        _ => Err(archive_error("device type tag is invalid")),
    }
}
fn archive_error(message: &str) -> GpuError {
    GpuError::InvalidShaderArchive(message.to_string())
}
fn incompatible(component: &str) -> GpuError {
    GpuError::IncompatibleShaders(format!("{component} does not match"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_shader() -> CompiledShader {
        CompiledShader(VulkanSpirvShader {
            entry_point: "main",
            words: vec![0x0723_0203, 0, 0, 0, 0],
            scalar: ScalarType::F32,
            sequence_len: 8,
            batch_count: 2,
            workgroup_size: WorkgroupSize { x: 8, y: 1, z: 1 },
            dispatch: DispatchGeometry { x: 2, y: 1, z: 1 },
            descriptors: vec![VulkanDescriptorBinding {
                set: 0,
                binding: 0,
                descriptor_type: VulkanDescriptorType::StorageBuffer,
                access: BufferAccess::ReadOnly,
                role: BufferRole::Input,
            }],
            required_shared_memory_bytes: 0,
            required_subgroup_size: None,
            require_full_subgroups: false,
        })
    }

    fn sample_transform(batch_count: usize) -> TransformShaders {
        let mut shader = sample_shader();
        shader.0.batch_count = batch_count;
        TransformShaders {
            batch_count,
            forward_fft: vec![shader.clone()],
            remap: shader.clone(),
            inverse_fft: vec![shader.clone()],
            overlap: OverlapShaders {
                start: shader.clone(),
                normal: shader.clone(),
                end: shader,
            },
        }
    }

    fn sample_archive() -> GpuShaders {
        GpuShaders {
            archive_version: ARCHIVE_VERSION,
            shader_abi_version: SHADER_ABI_VERSION,
            planner_commit: current_planner_commit(),
            gpu_info: GpuInfo {
                id: GpuDeviceId { device_uuid: [6; 16] },
                pipeline_cache_id: GpuPipelineCacheId {
                    vendor_id: 1,
                    device_id: 2,
                    pipeline_cache_uuid: [3; 16],
                },
                device_type: GpuDeviceType::Discrete,
                api_version: 4,
                driver_version: 5,
                shader_float64: true,
                max_storage_buffer_range: 1 << 20,
                max_compute_workgroup_count: [8, 8, 8],
                max_compute_workgroup_size: [64, 1, 1],
                max_compute_work_group_invocations: 64,
                device_name: "test device".to_string(),
            },
            geometry: GpuShaderGeometry {
                scalar: scalar_tag(ScalarType::F32),
                channels: 2,
                group_chunks: 4,
                input_fft_size: 16,
                output_fft_size: 32,
                input_chunk_frames: 8,
                output_chunk_frames: 16,
                direction_up: true,
                lower_nyquist_bin: 8,
                reflect_start_bin: 7,
                nyquist_fold_bits: 1.0f64.to_bits(),
                phase_enabled: false,
            },
            single: sample_transform(2),
            grouped: sample_transform(8),
            pipeline_cache: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn archive_round_trip_preserves_all_fields() {
        let shaders = sample_archive();
        assert_eq!(GpuShaders::decode(&shaders.to_bytes()).unwrap(), shaders);
    }

    #[test]
    fn rejects_truncated_archive() {
        assert!(matches!(
            GpuShaders::decode(MAGIC),
            Err(GpuError::InvalidShaderArchive(_))
        ));
    }

    #[test]
    fn rejects_bad_checksum_before_allocating_payloads() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&ARCHIVE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.push(1);
        assert!(matches!(
            GpuShaders::decode(&bytes),
            Err(GpuError::InvalidShaderArchive(_))
        ));
    }

    #[test]
    fn rejects_corruption_and_unknown_schema() {
        let mut corrupt = sample_archive().to_bytes();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(matches!(
            GpuShaders::decode(&corrupt),
            Err(GpuError::InvalidShaderArchive(_))
        ));

        let mut unknown = sample_archive().to_bytes();
        unknown[8..12].copy_from_slice(&(ARCHIVE_VERSION + 1).to_le_bytes());
        assert!(matches!(
            GpuShaders::decode(&unknown),
            Err(GpuError::InvalidShaderArchive(_))
        ));
    }
}
