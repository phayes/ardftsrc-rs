/// Errors from the GPU (Vulkan/VkFFT) backend.
///
/// This is intentionally separate from [`crate::Error`]: the GPU backend is optional
/// (`gpu` feature), experimental, and its failure modes (driver/loader problems, missing
/// device capabilities) are meaningfully different from the CPU core's.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GpuError {
    /// Vulkan instance/loader initialization failed (no Vulkan loader present, no ICD, etc).
    #[error("Vulkan initialization failed: {0}")]
    VulkanInitFailed(String),

    /// No physical device exposed a compute-capable queue family.
    #[error("no compatible Vulkan device found")]
    NoCompatibleDevice,

    /// `f64` GPU execution was requested but the selected device cannot run `shaderFloat64`.
    ///
    /// Notably, this is always the case on Apple GPUs (Metal has no double-precision type,
    /// so MoltenVK never reports `shaderFloat64` support).
    #[error("GPU device does not support 64-bit floating point shaders (shaderFloat64)")]
    Fp64Unsupported,

    /// Logical device creation failed after a physical device was selected.
    #[error("Vulkan device creation failed: {0}")]
    DeviceCreationFailed(String),

    /// VkFFT plan creation failed for the requested transform configuration.
    #[error("VkFFT plan creation failed: {0}")]
    PlanCreationFailed(String),

    /// A VkFFT or compute shader submission failed to execute.
    #[error("GPU execution failed: {0}")]
    ExecutionFailed(String),

    /// A GPU buffer/memory allocation failed.
    #[error("GPU allocation failed: {0}")]
    AllocationFailed(String),

    /// GPU pre-decimation was requested but is not yet supported.
    #[error("GPU decimation is not yet supported")]
    DecimationUnsupported,

    /// `dd_fft` (double-double precision) was requested on the GPU backend, which never
    /// supports it.
    #[error("dd_fft is not supported on the GPU backend")]
    DdFftUnsupportedOnGpu,

    /// A GPU core method was called in a state that does not allow it (for example, polling
    /// output on a slot that was never submitted).
    #[error("invalid GPU submission state: {0}")]
    InvalidSubmissionState(String),

    /// Output for a submitted chunk was polled before the GPU finished processing it.
    #[error("GPU output is not ready yet")]
    OutputNotReady,

    /// The [`crate::Config`] given to a GPU core failed CPU-side validation/derivation.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}
