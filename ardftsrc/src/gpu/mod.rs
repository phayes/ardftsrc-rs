//! Vulkan/VkFFT GPU backend infrastructure.
//!
//! This module owns infrastructure shared by [`GpuStreamingCore`] and [`GpuBatchCore`]: long-lived
//! Vulkan objects, device capability reporting, and GPU-specific error types. It intentionally
//! does not implement any DSP itself -- the CPU core (`crate::cpu_core::CpuCore`) remains the
//! reference implementation, and GPU cores are built independently on top of this module rather
//! than sharing execution logic with it.
//!
//! Both `f32` and `f64` are targeted; `dd_fft` is out of scope on the GPU backend
//! ([`GpuError::DdFftUnsupportedOnGpu`]), and `f64` execution itself requires a device that
//! reports `shaderFloat64` support ([`GpuContext::require_f64`]) -- notably, this is never
//! true on Apple GPUs, since Metal (and therefore MoltenVK) has no double-precision shader
//! type.

mod batch_core;
mod batch_overlap_shader;
mod buffer;
mod context;
mod error;
mod fft_program;
mod overlap_shader;
mod remap_shader;
mod streaming_core;
mod transform_pipeline;

pub use batch_core::GpuBatchCore;
pub use context::{GpuCapabilities, GpuContext};
pub use error::GpuError;
pub use streaming_core::GpuStreamingCore;
