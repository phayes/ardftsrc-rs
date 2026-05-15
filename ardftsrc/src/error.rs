/// Ardftsrc error types.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// Input sample rate was not set or is zero before an operation that requires it.
    #[error("input sample rate must be set (non-zero); use with_input_rate()")]
    MustSetInputSampleRate,

    /// Output sample rate was not set or is zero before an operation that requires it.
    #[error("output sample rate must be set (non-zero); use with_output_rate()")]
    MustSetOutputSampleRate,

    /// Channel count is zero.
    #[error("channel count must be greater than zero, got {0}")]
    InvalidChannels(usize),

    /// Quality setting is zero.
    #[error("quality must be greater than zero, got {0}")]
    InvalidQuality(usize),

    /// `f32` samples cannot use arbitrarily large FFT geometries derived from quality.
    #[error("quality greater than 8192 is not supported for f32. Use f64 instead.")]
    QualityTooHighForF32,

    /// Bandwidth is outside the valid normalized range `[0.0, 1.0]`.
    #[error("bandwidth must be in the range 0.0..=1.0, got {0}")]
    InvalidBandwidth(f32),

    /// Phase is outside the valid normalized range `[-1.0, 1.0]`.
    #[error("phase must be finite and in the range -1.0..=1.0, got {0}")]
    InvalidPhase(f32),

    /// Phase intensity is outside the valid range `[0.0, 100.0]`.
    #[error("phase intensity must be finite and in the range 0.0..=100.0, got {0}")]
    InvalidPhaseIntensity(f32),

    /// Cosine taper alpha is non-finite or not greater than zero.
    #[error("taper alpha must be finite and greater than zero, got {0}")]
    InvalidAlpha(f32),

    /// Beta taper beta is non-finite or not greater than zero.
    #[error("taper beta must be finite and greater than zero, got {0}")]
    InvalidBeta(f32),

    /// Wrong channel count.
    #[error("expected {expected} channels, got {actual}")]
    WrongChannelCount { expected: usize, actual: usize },

    /// Wrong frame count.
    #[error("expected {expected} frames, got {actual}")]
    WrongFrameCount { expected: usize, actual: usize },

    /// Interleaved input length is not divisible by channel count.
    #[error("interleaved input length {samples} is not divisible by channel count {channels}")]
    MalformedInputLength { channels: usize, samples: usize },

    /// Finalization was requested with an incomplete interleaved frame still buffered.
    #[error("cannot finalize with dangling partial frame: {samples} buffered samples for {channels} channels")]
    DanglingPartialFrame { channels: usize, samples: usize },

    /// Chunk size does not match the expected stream chunk length.
    #[error("expected {expected} samples in this chunk, got {actual}")]
    WrongChunkLength { expected: usize, actual: usize },

    /// Provided output buffer is smaller than required for produced samples.
    #[error("output buffer can hold {actual} samples, but {expected} samples are required")]
    InsufficientOutputBuffer { expected: usize, actual: usize },

    /// Additional input was submitted after final input was already provided.
    #[error("stream has already been finalized")]
    AlreadyFinalized,

    /// FFT backend reported an internal failure.
    #[error("FFT backend error: {0}")]
    Fft(String),

    /// A preset config was used before required sample rates and channel count were set.
    #[error(
        "preset config must be configured before creating a stream. Use with_input_rate(), with_output_rate(), and with_channels() to configure the preset."
    )]
    PresetNotConfigured,
}
