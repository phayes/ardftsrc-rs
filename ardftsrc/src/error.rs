#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// Input or output sample rate is zero.
    #[error("input and output sample rates must both be greater than zero, got {input} and {output}")]
    InvalidSampleRate { input: usize, output: usize },

    /// Channel count is zero.
    #[error("channel count must be greater than zero, got {0}")]
    InvalidChannels(usize),

    /// Quality setting is zero.
    #[error("quality must be greater than zero, got {0}")]
    InvalidQuality(usize),

    /// Bandwidth is outside the valid normalized range `[0.0, 1.0]`.
    #[error("bandwidth must be in the range 0.0..=1.0, got {0}")]
    InvalidBandwidth(f32),

    /// Cosine taper alpha is non-finite or not greater than zero.
    #[error("alpha must be finite and greater than zero, got {0}")]
    InvalidAlpha(f32),

    /// Wrong channel count.
    #[error("expected {expected} channels, got {actual}")]
    WrongChannelCount { expected: usize, actual: usize },

    /// Wrong frame count.
    #[error("expected {expected} frames, got {actual}")]
    WrongFrameCount { expected: usize, actual: usize },

    /// Interleaved input length is not divisible by channel count.
    #[error("interleaved input length {samples} is not divisible by channel count {channels}")]
    MalformedInputLength { channels: usize, samples: usize },

    /// Chunk size does not match the expected stream chunk length.
    #[error("expected {expected} samples in this chunk, got {actual}")]
    WrongChunkLength { expected: usize, actual: usize },

    /// Provided output buffer is smaller than required for produced samples.
    #[error("output buffer can hold {actual} samples, but {expected} samples are required")]
    InsufficientOutputBuffer { expected: usize, actual: usize },

    /// Additional input was submitted after final input was already provided.
    #[error("stream has already been finalized")]
    StreamAlreadyFinalized,

    /// Flush was requested more than once.
    #[error("stream has already been flushed")]
    AlreadyFlushed,

    /// FFT backend reported an internal failure.
    #[error("FFT backend error: {0}")]
    Fft(String),

    /// A preset config was used before required sample rates and channel count were set.
    #[error(
        "preset config must be configured before creating a stream. Use with_input_rate(), with_output_rate(), and with_channels() to configure the preset."
    )]
    PresetNotConfigured,
}
