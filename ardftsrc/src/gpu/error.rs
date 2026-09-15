/// Errors from the GPU (Vulkan/VkFFT) backend.
///
/// This is intentionally separate from [`crate::Error`]: the GPU backend is optional
/// (`gpu` feature), experimental, and its failure modes (driver/loader problems, missing
/// device capabilities) are meaningfully different from the CPU core's.
use super::context::GpuDeviceId;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GpuError {
    /// Vulkan instance/loader initialization failed (no Vulkan loader present, no ICD, etc).
    #[error("Vulkan initialization failed: {0}")]
    VulkanInitFailed(String),

    /// No physical device exposed a compute-capable queue family.
    #[error("no compatible Vulkan device found")]
    NoCompatibleDevice,

    /// The requested physical-device UUID was not present among compatible devices.
    #[error("requested Vulkan GPU device was not found: {0:?}")]
    DeviceNotFound(GpuDeviceId),

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

    /// A high-precision FFT backend was requested on the GPU backend, which never supports it.
    #[error("high_precision is not supported on the GPU backend")]
    HighPrecisionUnsupportedOnGpu,

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

    /// A serialized shader archive is malformed or uses an unsupported format.
    #[error("invalid GPU shader archive: {0}")]
    InvalidShaderArchive(String),

    /// Compiled shaders do not match the context's device or resampling geometry.
    #[error("GPU shaders are incompatible with this context: {0}")]
    IncompatibleShaders(String),

    /// Vulkan pipeline-cache data could not be imported or exported.
    #[error("Vulkan pipeline cache failed: {0}")]
    PipelineCacheFailed(String),

    /// Wrong number of channel slices/streams were provided.
    #[error("expected {expected} channels, got {actual}")]
    WrongChannelCount { expected: usize, actual: usize },

    /// Chunk length does not match the expected stream chunk length (or exceeds it for a final chunk).
    #[error("expected {expected} frames, got {actual}")]
    WrongFrameCount { expected: usize, actual: usize },

    /// Interleaved buffer length is not evenly divisible by channel count.
    #[error("interleaved input length {samples} is not divisible by channel count {channels}")]
    MalformedInputLength { channels: usize, samples: usize },

    /// Provided output buffer is smaller than required for produced samples.
    #[error("output buffer can hold {actual} samples, but {expected} samples are required")]
    InsufficientOutputBuffer { expected: usize, actual: usize },

    /// Additional input (or another finalize) was submitted after the stream was already finalized.
    #[error("stream has already been finalized")]
    AlreadyFinalized,
}
