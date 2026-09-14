use std::sync::Arc;

use ash::vk;
use num_traits::Float;
use vkfft_rs::backend::vulkan::{VulkanGlslBackend, VulkanSpirvShader};
use vkfft_rs::{
    Complex64, FftConfig, Precision, ProgramIr, ProgramResourceInitialization, ProgramResourceKind, RealFftIr,
    RealFftKind, TransformKind,
};

use super::buffer::{GpuBuffer, GpuScalar};
use super::context::{GpuComputePipeline, GpuDevice, RecordingCommandBuffer};
use super::error::GpuError;
use super::shaders::CompiledShader;

fn vkfft_err(context: &'static str, err: impl std::fmt::Display) -> GpuError {
    GpuError::PlanCreationFailed(format!("{context}: {err}"))
}

/// Casts an `f64` (`vkfft-rs`'s canonical LUT/init-value precision) down to this buffer's
/// scalar type. Identity for `f64`, a narrowing cast for `f32`.
pub(crate) trait FromF64: Sized {
    fn from_f64(value: f64) -> Self;
}
impl FromF64 for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}
impl FromF64 for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

/// Packs `real` (a real-valued window) into `dst` as `Complex<T>` (`im = 0`), at batch index
/// `batch` of a buffer laid out `stride` complex slots per batch -- the packing convention every
/// [`GpuRealFft`] input buffer expects (`batch` = `window_index * channels + channel`, see
/// `crate::gpu::gpu_core`).
pub(crate) fn pack_complex_batch<T: Float>(dst: &mut [T], batch: usize, stride: usize, real: &[T]) {
    let base = batch * stride * 2;
    for (i, &value) in real.iter().enumerate() {
        dst[base + i * 2] = value;
        dst[base + i * 2 + 1] = T::zero();
    }
}

fn flatten_complex64<T: FromF64>(values: &[Complex64]) -> Vec<T> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.push(T::from_f64(value.re));
        out.push(T::from_f64(value.im));
    }
    out
}

/// A planned, GPU-resident real FFT (R2C or C2R) for a fixed `(length, batch_count)`, reused
/// unchanged across every chunk.
///
/// Built entirely from `vkfft-rs`'s public planner/shader-lowering API (`RealFftIr`,
/// `ProgramIr`, `VulkanGlslBackend::lower_real_fft`, `VulkanComputePipeline`) against buffers
/// and pipelines this type owns -- never `vkfft-rs`'s own hidden `VulkanExecutionContext`.
/// Buffers are allocated once, pipelines are built once, and every pipeline's descriptor set
/// is bound to its persistent buffer once at construction; per chunk, only [`GpuRealFft::record`]
/// runs (real data is written into/read out of [`GpuRealFft::input_buffer`]/
/// [`GpuRealFft::output_buffer`] by the caller, outside this type). This is what lets a GPU
/// core append a forward FFT, a custom spectral-remap dispatch, and an inverse FFT into one
/// command buffer with one submission -- satisfying `gpu_plan.md`'s "never transfer the
/// spectrum back to the CPU between forward and inverse FFT" rule using only `vkfft-rs`'s
/// stable public surface.
pub(crate) struct GpuRealFft<T> {
    /// One persistent `GpuBuffer<T>` per `ProgramAllocation`, indexed by `ProgramAllocationId`.
    allocations: Vec<GpuBuffer<T>>,
    /// One pipeline per `ProgramIr` pass, in `program.passes` order, with its descriptor set
    /// already bound to the matching entries in `allocations`.
    pipelines: Vec<GpuComputePipeline>,
    input_allocation: usize,
    output_allocation: usize,
    /// Complex elements per batch row in [`GpuRealFft::input_buffer`] -- may exceed
    /// the transform length (R2C) or half-spectrum length (C2R) if the planner padded each
    /// batch's physical storage.
    input_stride: usize,
    /// Complex elements per batch row in [`GpuRealFft::output_buffer`] -- may exceed
    /// the half-spectrum length (R2C) or transform length (C2R).
    output_stride: usize,
}

pub(crate) struct CompiledFft {
    pub(crate) shaders: Vec<VulkanSpirvShader>,
    pub(crate) input_stride: usize,
    pub(crate) output_stride: usize,
}

pub(crate) fn compile_fft(
    context: &GpuDevice,
    kind: RealFftKind,
    length: usize,
    batch_count: usize,
    precision: Precision,
    normalize_inverse: bool,
) -> Result<CompiledFft, GpuError> {
    let transform = match kind {
        RealFftKind::RealToComplex => TransformKind::RealToComplex,
        RealFftKind::ComplexToReal => TransformKind::ComplexToReal,
    };
    let config = FftConfig::new(vec![length])
        .with_transform(transform)
        .with_batch_count(batch_count)
        .with_precision(precision)
        .with_inverse_normalization(normalize_inverse);
    let plan = vkfft_rs::FftPlan::build(config).map_err(|err| vkfft_err("FFT planning failed", err))?;
    let ir = RealFftIr::build(&plan, context.device_profile())
        .map_err(|err| vkfft_err("real FFT IR construction failed", err))?;
    ir.validate()
        .map_err(|err| vkfft_err("real FFT IR failed validation", err))?;
    let shaders = VulkanGlslBackend
        .lower_real_fft(&ir)
        .map_err(|err| vkfft_err("failed to lower real FFT IR to Vulkan shaders", err))?
        .into_iter()
        .map(|shader| {
            shader
                .compile_spirv()
                .map_err(|err| vkfft_err("failed to compile Vulkan shader to SPIR-V", err))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let program = ProgramIr::real_fft(&ir).map_err(|err| vkfft_err("failed to build real FFT program IR", err))?;
    let input = program
        .resources
        .iter()
        .find(|resource| resource.kind == ProgramResourceKind::Input)
        .and_then(|resource| resource.external_layout)
        .ok_or_else(|| GpuError::PlanCreationFailed("real FFT input resource has no external layout".to_string()))?;
    let output = program
        .resources
        .iter()
        .find(|resource| resource.kind == ProgramResourceKind::Output)
        .and_then(|resource| resource.external_layout)
        .ok_or_else(|| GpuError::PlanCreationFailed("real FFT output resource has no external layout".to_string()))?;
    Ok(CompiledFft {
        shaders,
        input_stride: input.physical_stride,
        output_stride: output.physical_stride,
    })
}

impl<T: GpuScalar + FromF64> GpuRealFft<T> {
    /// Plans and allocates a real FFT of `kind` for `length` real samples per transform,
    /// batched `batch_count` at a time (one batch per channel for streaming, or per
    /// chunk-channel pair for batch mode).
    /// `normalize_inverse` selects whether a `ComplexToReal` build divides its output by
    /// `length` (matching `realfft`/CPU-core convention -- see `CpuCore::transform_chunk`'s
    /// manual `normalize` scale); meaningless (and ignored) for `RealToComplex`.
    pub(crate) fn build(
        context: &Arc<GpuDevice>,
        kind: RealFftKind,
        length: usize,
        batch_count: usize,
        precision: Precision,
        normalize_inverse: bool,
        shaders: &[CompiledShader],
    ) -> Result<Self, GpuError> {
        let transform = match kind {
            RealFftKind::RealToComplex => TransformKind::RealToComplex,
            RealFftKind::ComplexToReal => TransformKind::ComplexToReal,
        };
        let config = FftConfig::new(vec![length])
            .with_transform(transform)
            .with_batch_count(batch_count)
            .with_precision(precision)
            .with_inverse_normalization(normalize_inverse);
        let plan = vkfft_rs::FftPlan::build(config).map_err(|err| vkfft_err("FFT planning failed", err))?;

        let ir = RealFftIr::build(&plan, context.device_profile())
            .map_err(|err| vkfft_err("real FFT IR construction failed", err))?;
        ir.validate()
            .map_err(|err| vkfft_err("real FFT IR failed validation", err))?;

        let program = ProgramIr::real_fft(&ir).map_err(|err| vkfft_err("failed to build real FFT program IR", err))?;
        if shaders.len() != program.passes.len() {
            return Err(GpuError::IncompatibleShaders(format!(
                "FFT pass count differs: archive has {}, planner requires {}",
                shaders.len(),
                program.passes.len()
            )));
        }
        let memory_plan = program
            .memory_plan()
            .map_err(|err| vkfft_err("failed to plan real FFT program memory", err))?;

        let mut allocations = Vec::with_capacity(memory_plan.allocations.len());
        for allocation in &memory_plan.allocations {
            // Complex packing: every program allocation (real data included -- real input is
            // packed as `Complex<T>` with `im = 0`, matching `vkfft-rs`'s own convention) is
            // `elements` complex values, i.e. `elements * 2` scalar `T`s.
            let buffer = GpuBuffer::<T>::new(context, allocation.elements * 2, vk::BufferUsageFlags::empty())?;
            let zeros = vec![T::default(); buffer.len()];
            buffer.upload(&zeros)?;
            allocations.push(buffer);
        }

        for resource in &program.resources {
            let allocation_id = memory_plan
                .allocation_for(resource.id)
                .map_err(|err| vkfft_err("real FFT resource has no allocation", err))?;
            match &resource.initialization {
                ProgramResourceInitialization::ExternalInput | ProgramResourceInitialization::Zeroed => {}
                ProgramResourceInitialization::Complex64(values) => {
                    let flat = flatten_complex64::<T>(values);
                    allocations[allocation_id.0].upload(&flat)?;
                }
                ProgramResourceInitialization::StockhamUnitRoots { len } => {
                    let values = vkfft_rs::lut::stockham_root_table(*len)
                        .map_err(|err| vkfft_err("failed to build Stockham root table", err))?;
                    let flat = flatten_complex64::<T>(&values);
                    allocations[allocation_id.0].upload(&flat)?;
                }
                ProgramResourceInitialization::ComplexDoubleDouble(_) => {
                    return Err(GpuError::F128UnsupportedOnGpu);
                }
            }
        }

        let mut pipelines = Vec::with_capacity(program.passes.len());
        for (pass, shader) in program.passes.iter().zip(shaders) {
            let shader = shader.as_spirv();
            let planned_bindings = pass.bindings.iter().map(|binding| binding.binding).collect::<Vec<_>>();
            let archived_bindings = shader
                .descriptors
                .iter()
                .map(|descriptor| (descriptor.set, descriptor.binding))
                .collect::<Vec<_>>();
            if shader.scalar != T::scalar_type()
                || archived_bindings != planned_bindings.iter().map(|&binding| (0, binding)).collect::<Vec<_>>()
            {
                return Err(GpuError::IncompatibleShaders(format!(
                    "FFT pass descriptor contract differs from regenerated program IR \
                     (planned bindings {planned_bindings:?}, archived bindings {archived_bindings:?}, \
                     planned scalar {:?}, archived scalar {:?})",
                    T::scalar_type(),
                    shader.scalar
                )));
            }
            let mut pipeline = context.create_compute_pipeline(shader)?;

            let mut bindings = Vec::with_capacity(pass.bindings.len());
            for binding in &pass.bindings {
                let allocation_id = memory_plan
                    .allocation_for(binding.resource)
                    .map_err(|err| vkfft_err("FFT pass binding has no allocation", err))?;
                bindings.push((binding.binding, allocations[allocation_id.0].storage_binding()?));
            }
            pipeline
                .bind_storage_buffers(bindings)
                .map_err(|err| vkfft_err("failed to bind FFT pass buffers", err))?;

            pipelines.push(pipeline);
        }

        let input_resource = program
            .resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Input)
            .ok_or_else(|| GpuError::PlanCreationFailed("real FFT program has no input resource".to_string()))?;
        let output_resource = program
            .resources
            .iter()
            .find(|resource| resource.kind == ProgramResourceKind::Output)
            .ok_or_else(|| GpuError::PlanCreationFailed("real FFT program has no output resource".to_string()))?;
        let input_allocation = memory_plan
            .allocation_for(input_resource.id)
            .map_err(|err| vkfft_err("real FFT input resource has no allocation", err))?
            .0;
        let output_allocation = memory_plan
            .allocation_for(output_resource.id)
            .map_err(|err| vkfft_err("real FFT output resource has no allocation", err))?
            .0;
        let input_stride = input_resource
            .external_layout
            .ok_or_else(|| GpuError::PlanCreationFailed("real FFT input resource has no external layout".to_string()))?
            .physical_stride;
        let output_stride = output_resource
            .external_layout
            .ok_or_else(|| GpuError::PlanCreationFailed("real FFT output resource has no external layout".to_string()))?
            .physical_stride;

        Ok(Self {
            allocations,
            pipelines,
            input_allocation,
            output_allocation,
            input_stride,
            output_stride,
        })
    }

    /// The buffer real (R2C) or complex-spectrum (C2R) input data must be written into before
    /// [`GpuRealFft::record`] runs, packed as `Complex<T>` (`im = 0` for real data) per
    /// `vkfft-rs`'s convention, batched contiguously (`length` complex slots per batch for R2C,
    /// `half_spectrum_len` for C2R).
    pub(crate) fn input_buffer(&self) -> &GpuBuffer<T> {
        &self.allocations[self.input_allocation]
    }

    /// The buffer this FFT's result is available in after [`GpuRealFft::record`] runs:
    /// complex-spectrum for R2C (`half_spectrum_len` complex slots per batch), or real output
    /// packed as `Complex<T>` (`im` always `0`) for C2R (`length` complex slots per batch).
    pub(crate) fn output_buffer(&self) -> &GpuBuffer<T> {
        &self.allocations[self.output_allocation]
    }

    /// Complex elements per batch row in [`GpuRealFft::input_buffer`] (see the field doc).
    pub(crate) fn input_stride(&self) -> usize {
        self.input_stride
    }

    /// Complex elements per batch row in [`GpuRealFft::output_buffer`] (see the field doc).
    pub(crate) fn output_stride(&self) -> usize {
        self.output_stride
    }

    /// Records every pass of this FFT, with a full compute barrier after each dispatch.
    pub(crate) fn record(&self, command: &RecordingCommandBuffer<'_>) {
        for pipeline in &self.pipelines {
            command.dispatch(pipeline);
            command.compute_barrier();
        }
    }
}
