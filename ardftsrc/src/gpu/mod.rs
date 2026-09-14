//! Vulkan/VkFFT GPU backend.
//!
//! [`GpuDevice`] owns shareable Vulkan infrastructure. [`GpuContext`] owns one validated
//! resampling configuration, its derived geometry, chunk-group parallelism, and lazily compiled
//! [`GpuShaders`]. [`GpuCore`] adds only ring-buffer depth and mutable streaming state. Compiled
//! shaders can be persisted with [`GpuShaders::to_bytes`] and restored with
//! [`GpuContext::load_shaders`].
//!
//! Both `f32` and `f64` are targeted; `f128` is out of scope on the GPU backend
//! ([`GpuError::F128UnsupportedOnGpu`]), and `f64` execution itself requires a device that
//! reports `shaderFloat64` support ([`GpuDevice::require_f64`]) -- notably, this is never
//! true on Apple GPUs, since Metal (and therefore MoltenVK) has no double-precision shader
//! type.

mod batch_overlap_shader;
mod buffer;
mod context;
mod error;
mod fft_program;
mod gpu_core;
mod interleaved_gpu_resampler;
mod overlap_shader;
mod planar_gpu_resampler;
mod remap_shader;
mod shaders;
mod transform_pipeline;

pub use context::{GpuContext, GpuDevice, GpuDeviceId, GpuDeviceType, GpuInfo, GpuPipelineCacheId};
pub use error::GpuError;
pub use gpu_core::GpuCore;
pub use interleaved_gpu_resampler::InterleavedGpuResampler;
pub use planar_gpu_resampler::PlanarGpuResampler;
pub use shaders::GpuShaders;
