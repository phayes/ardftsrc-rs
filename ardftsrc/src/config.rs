use num_traits::Float;
use realfft::num_complex::Complex;

/// Low-latency, lower-quality preset.
///
/// You may prefer using a sinc resampler (eg. rubato) instead.
///
/// **HydrogenAudio SRC quality results f32**
/// - Overall Score: 67.98%
/// - Spectrogram Score: 50.66%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 89.1%
/// - Intermodulation Distortion Score: 35.66%
/// - Impulse Frequency Score: 50.76%
/// - Pre-ringing Score: 26.67%
/// - Gapless Sine Score: 45.58%
/// - Link with more details: <https://src.hydrogenaudio.org/compareresults?id1=c527356d-3566-46f8-8dea-dc2065b11e46&id2=0>
///
/// **HydrogenAudio SRC quality results f64**
/// - Overall Score: 92.69%
/// - Spectrogram Score: 91.34%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 87.96%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 96.28%
/// - Pre-ringing Score: 26.79%
/// - Gapless Sine Score: 55.61%
/// - Link with more details: <https://src.hydrogenaudio.org/compareresults?id1=8e59a5bd-8147-470c-9501-44ab81718b8f&id2=0>
///
/// # Example
///
/// ```rust
/// let config = ardftsrc::PRESET_FAST
///     .with_input_rate(44_100)
///     .with_output_rate(48_000)
///     .with_channels(2);
/// ```
pub const PRESET_FAST: Config = Config {
    input_sample_rate: 0,
    output_sample_rate: 0,
    channels: 0,
    quality: 512,
    bandwidth: 0.8323,
    taper_type: TaperType::Cosine(3.4375),
    ..Config::DEFAULT
};

/// Balanced preset for good realtime quality. ***You should probably use this one.***
///
/// **HydrogenAudio SRC quality results f64**
/// - Overall Score: 97.45%
/// - Spectrogram Score: 95.61%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 93.59%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 99.5%
/// - Pre-ringing Score: 21.24%
/// - Gapless Sine Score: 100%
/// - Link with more details: <https://src.hydrogenaudio.org/compareresults?id1=e12d7fe0-dfa2-4c49-bbdd-51c16a931cb5&id2=0>
///
/// # Example
/// ```rust
/// let config = ardftsrc::PRESET_GOOD
///     .with_input_rate(44_100)
///     .with_output_rate(48_000)
///     .with_channels(2);
/// ```
pub const PRESET_GOOD: Config = Config {
    input_sample_rate: 0,
    output_sample_rate: 0,
    channels: 0,
    quality: 1878,
    bandwidth: 0.9114534,
    taper_type: TaperType::Cosine(3.4375),
    ..Config::DEFAULT
};

/// High quality preset suitable for offline processing or realtime applications where quality is critical.
///
/// **HydrogenAudio SRC quality results f64**
/// - Overall Score: 99.26%
/// - Spectrogram Score: 99.41%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 99.08%
/// - Intermodulation Distortion Score: 98.41%
/// - Impulse Frequency Score: 98.74%
/// - Pre-ringing Score: 17.87%
/// - Gapless Sine Score: 100%
/// - Link with more details: <https://src.hydrogenaudio.org/compareresults?id1=43a72723-7f35-4318-bbd1-44cdfaa6df88&id2=0>
///
/// # Example
/// ```rust
/// let config = ardftsrc::PRESET_HIGH
///     .with_input_rate(44_100)
///     .with_output_rate(48_000)
///     .with_channels(2);
/// ```
pub const PRESET_HIGH: Config = Config {
    input_sample_rate: 0,
    output_sample_rate: 0,
    channels: 0,
    quality: 73622,
    bandwidth: 0.9873534,
    taper_type: TaperType::Cosine(3.4375),
    ..Config::DEFAULT
};

/// Maximum quality preset, optimized for offline processing. Not recommended for realtime applications.
///
/// **HydrogenAudio SRC quality results f64**
/// - Overall Score: 99.70%
/// - Spectrogram Score: 99.64%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 99.64%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 99.03%
/// - Pre-ringing Score: 17.74%
/// - Gapless Sine Score: 100%
/// - Link with more details: <https://src.hydrogenaudio.org/compareresults?id1=dbdbdd66-d8b8-4b8b-b217-b71162cb1f2f&id2=0>
///
/// # Example
/// ```rust
/// let config = ardftsrc::PRESET_EXTREME
///     .with_input_rate(44_100)
///     .with_output_rate(48_000)
///     .with_channels(2);
/// ```
pub const PRESET_EXTREME: Config = Config {
    input_sample_rate: 0,
    output_sample_rate: 0,
    channels: 0,
    quality: 524514,
    bandwidth: 0.9952346,
    taper_type: TaperType::Cosine(3.4375),
    ..Config::DEFAULT
};

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Transition profile used to shape the cutoff edge of the frequency mask.
pub enum TaperType {
    /// Uses a Planck-taper transition
    Planck,

    /// Uses a sigmoid-warped cosine transition.
    ///
    /// `alpha` controls the sharpness of the transition.
    ///
    /// Value guide for `Cosine(alpha)`:
    /// - `1.5`: Very smooth transition; may increase near-Nyquist artifacts.
    /// - `2.5`: Smooth and less aggressive shaping.
    /// - `3.5`: Good balance between smoothness and selectivity.
    /// - `4.0`: Sharper shaping; trades smoothness for selectivity.
    Cosine(f32),
}

impl Default for TaperType {
    fn default() -> Self {
        Self::Cosine(3.4375)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Configures the ardftsrc resampler.
/// 
/// # Example
///
/// ```rust
/// let config = ardftsrc::Config::new(44_100, 48_000, 2).with_phase(-0.5);
/// ```
pub struct Config {
    /// Input audio sample rate in Hz.
    pub input_sample_rate: usize,

    /// Output audio sample rate in Hz.
    pub output_sample_rate: usize,

    /// Number of interleaved audio channels.
    pub channels: usize,

    /// Set the overall "quality" of the resampler.
    ///  
    /// Quality roughly sets the spectral resolution scale (and therefore FFT bin count),
    /// but this mapping is not exactly 1:1 (exact bin count depends on rate ratio and quantization).
    ///
    /// Default value is 1878 (same quality as PRESET_GOOD).
    ///
    /// Value guide:
    ///  - `512` (PRESET_FAST):       Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    ///  - `1878` (PRESET_GOOD):      Good balanced quality - you should probably use this. (Default)
    ///  - `73622` (PRESET_HIGH):     High quality, good for offline resampling, also marginally appropriate for realtime applications where quality is critical.
    ///  - `524514` (PRESET_EXTREME): Extreme quality, good for offline resampling, very high quality but also very slow. Not recommended for realtime applications.
    pub quality: usize,

    /// Normalized filter bandwidth in the range `[0.0, 1.0]`.
    ///
    /// Higher values preserve more high-frequency content but shorten the transition band.
    ///
    /// Value guide:
    /// - `0.82`: Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    /// - `0.95`: Balanced high-end retention for most cases.
    /// - `0.97`: More aggressive high-end retention; Use with a higher "quality" setting.
    /// - `0.99`: Very aggressive high-end retention; Only recommended when using a very high "quality" setting.
    pub bandwidth: f32,

    /// Frequency taper profile used around the cutoff region.
    ///
    /// - `Planck`: Uses a Planck taper transition.
    /// - `Cosine(alpha)`: Uses a sigmoid-warped cosine transition.
    ///
    /// Default value is `Cosine(3.4375)`, which was arrived at through testing
    /// various values on the HydrogenAudio SRC test suite.
    ///
    /// Lower `alpha` values result in a smoother transition, while higher values
    /// produce a sharper transition.
    ///
    /// Value guide for `Cosine(alpha)`:
    /// - `1.5`: Very smooth transition; may increase audible near-Nyquist artifacts.
    /// - `2.5`: Smooth and less aggressive shaping.
    /// - `3.5`: Good balance between smoothness and selectivity.
    /// - `4.0`: Sharper shaping; can trade smoothness for selectivity.
    pub taper_type: TaperType,

    /// Phase: Frequency-dependent phase rotation in the range `[-1.0, 1.0]`.
    ///
    /// Positive values rotate higher bins forward; negative values apply the conjugate rotation.
    /// `0.0` disables phase rotation.
    ///
    /// Setting a negative phase value can help with pre-ringing artifacts.
    ///
    /// Default value is `0.0`.
    pub phase: f32,

    /// Scales the phase rotation angle in the range `[0.0, 100.0]`.
    ///
    /// `0.0` disables phase rotation. The default value is `50.0`.
    pub phase_intensity: f32,

    /// The range of input sample rates that the realtime resampler will support.
    ///
    /// This is used in combination with `realtime_max_channels` to set the size of the ringer buffers for off-thread realtime resampling.
    /// The default value (22.05KHz - 192KHz at 7.1 surround) is very generous in what it supports, but uses about 8MB of memory.
    /// To reduce memory usage, you can set this to a narrower range, or reduce the number of channels supported.
    ///
    /// Because this range sets the buffer size and is not a hard limit, it will often support values outside the range (eg using the default config, 22.05KHz -> 384Khz stereo will work fine).
    /// Going above the range (eg using the default config, but resampling 22.05KHz -> 384Khz 13.1 surround) will not error, but will cause underruns and crackling (negative-zero samples on `read_sample()``).
    ///
    /// This setting is only for realtime resamplers (`RealtimeResampler`, `RodioResampler`), it has no effect on chunk resamplers (`InterleavedResampler` and `PlanarResampler`).
    #[cfg(feature = "realtime")]
    pub realtime_input_range: std::ops::Range<usize>,

    /// The maximum number of channels that the realtime resampler will support. See `realtime_input_range`.
    ///
    /// This setting is only for realtime resamplers (`RealtimeResampler`, `RodioResampler`), it has no effect on chunk resamplers (`InterleavedResampler` and `PlanarResampler`).\
    #[cfg(feature = "realtime")]
    pub realtime_max_channels: usize,

    /// For `RodioResampler`, this setting controls whether to use a fast start mode.
    ///
    /// Fast start mode will prime the resampler with initial samples to get it up to speed, and avoid start-up silence.
    /// This is only appropriate to use when the inner sounce can handle rapid calls to `next()`. For example, this will
    /// generally work on buffered streams or audio files, but not on live microphones.
    ///
    ///   - Set to "true" if the inner source is something like a buffered stream or audio file.
    ///   - Set to "false" if the inner source is very realtime (e.g. a live microphone).
    ///
    /// If set to `true` for an inner source that cannot handle this, you will experience crackling at the start of the stream as the inner source fails to keep up.
    ///
    /// This setting is only for `RodioResampler`, it has no effect on other resamplers.
    #[cfg(feature = "rodio")]
    pub rodio_fast_start: bool,
}

impl Config {
    pub const DEFAULT: Self = Self {
        input_sample_rate: 0,
        output_sample_rate: 0,
        channels: 2,
        quality: 1878,
        bandwidth: 0.9114534,
        taper_type: TaperType::Cosine(3.4375),
        phase: 0.0,
        phase_intensity: 50.0,
        #[cfg(feature = "realtime")]
        realtime_input_range: 22_050..192_000,
        #[cfg(feature = "realtime")]
        realtime_max_channels: 8,
        #[cfg(feature = "rodio")]
        rodio_fast_start: false,
    };

    /// Builds a config with explicit sample rates/channel count and default (PRESET_GOOD) quality settings.
    #[must_use]
    pub fn new(input_sample_rate: usize, output_sample_rate: usize, channels: usize) -> Self {
        Self {
            input_sample_rate,
            output_sample_rate,
            channels,
            ..Self::default()
        }
    }

    /// Input audio sample rate in Hz.
    #[must_use]
    pub fn with_input_rate(mut self, input_sample_rate: usize) -> Self {
        self.input_sample_rate = input_sample_rate;
        self
    }

    /// Output audio sample rate in Hz.
    #[must_use]
    pub fn with_output_rate(mut self, output_sample_rate: usize) -> Self {
        self.output_sample_rate = output_sample_rate;
        self
    }

    /// Number of interleaved audio channels.
    #[must_use]
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    /// Set the overall "quality" of the resampler.
    /// 
    /// Quality roughly sets the spectral resolution scale (and therefore FFT bin count),
    /// but this mapping is not exactly 1:1 (exact bin count depends on rate ratio and quantization).
    ///
    /// Default value is 1878 (same quality as PRESET_GOOD).
    ///
    /// Value guide:
    ///  - `512` (PRESET_FAST):       Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    ///  - `1878` (PRESET_GOOD):      Good balanced quality - you should probably use this. (Default)
    ///  - `73622` (PRESET_HIGH):     High quality, good for offline resampling, also marginally appropriate for realtime applications where quality is critical.
    ///  - `524514` (PRESET_EXTREME): Extreme quality, good for offline resampling, very high quality but also very slow. Not recommended for realtime applications.
    #[must_use]
    pub fn with_quality(mut self, quality: usize) -> Self {
        self.quality = quality;
        self
    }

    /// Normalized filter bandwidth in the range `[0.0, 1.0]`.
    ///
    /// Higher values preserve more high-frequency content but shorten the transition band.
    ///
    /// Value guide:
    /// - `0.82`: Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    /// - `0.95`: Balanced high-end retention for most cases.
    /// - `0.97`: More aggressive high-end retention; Use with a higher "quality" setting.
    /// - `0.99`: Very aggressive high-end retention; Only recommended when using a very high "quality" setting.
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: f32) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Frequency taper profile used around the cutoff region.
    ///
    /// - `Planck`: Uses a Planck taper transition.
    /// - `Cosine(alpha)`: Uses a sigmoid-warped cosine transition.
    ///
    /// Default value is `Cosine(3.4375)`, which was arrived at through testing
    /// various values on the HydrogenAudio SRC test suite.
    ///
    /// Lower `alpha` values result in a smoother transition, while higher values
    /// produce a sharper transition.
    ///
    /// Value guide for `Cosine(alpha)`:
    /// - `1.5`: Very smooth transition; may increase audible near-Nyquist artifacts.
    /// - `2.5`: Smooth and less aggressive shaping.
    /// - `3.5`: Good balance between smoothness and selectivity.
    /// - `4.0`: Sharper shaping; can trade smoothness for selectivity.
    #[must_use]
    pub fn with_taper_type(mut self, taper_type: TaperType) -> Self {
        self.taper_type = taper_type;
        self
    }

    /// Phase: Frequency-dependent phase rotation in the range `[-1.0, 1.0]`.
    ///
    /// Positive values rotate higher bins forward; negative values apply the conjugate rotation.
    /// `0.0` disables phase rotation.
    ///
    /// Setting a negative phase value can help with pre-ringing artifacts.
    ///
    /// Default value is `0.0`.
    #[must_use]
    pub fn with_phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }

    /// Scales the phase rotation angle in the range `[0.0, 100.0]`.
    ///
    /// `0.0` disables phase rotation. The default value is `50.0`.
    #[must_use]
    pub fn with_phase_intensity(mut self, phase_intensity: f32) -> Self {
        self.phase_intensity = phase_intensity;
        self
    }

    /// The range of input sample rates that the realtime resampler will support.
    ///
    /// This is used in combination with `realtime_max_channels` to set the size of the ringer buffers for off-thread realtime resampling.
    /// The default value (22.05KHz - 192KHz at 7.1 surround) is very generous in what it supports, but uses about 8MB of memory.
    /// To reduce memory usage, you can set this to a narrower range, or reduce the number of channels supported.
    ///
    /// Because this range sets the buffer size and is not a hard limit, it will often support values outside the range (eg using the default config, 22.05KHz -> 384Khz stereo will work fine).
    /// Going above the range (eg using the default config, but resampling 22.05KHz -> 384Khz 13.1 surround) will not error, but will cause underruns and crackling (negative-zero samples on `read_sample()``).
    ///
    /// This setting is only for realtime resamplers (`RealtimeResampler`, `RodioResampler`), it has no effect on chunk resamplers (`InterleavedResampler` and `PlanarResampler`).
    #[must_use]
    #[cfg(feature = "realtime")]
    pub fn with_realtime_input_range(mut self, realtime_input_range: std::ops::Range<usize>) -> Self {
        self.realtime_input_range = realtime_input_range;
        self
    }

    /// The maximum number of channels that the realtime resampler will support. See `realtime_input_range`.
    ///
    /// This setting is only for realtime resamplers (`RealtimeResampler`, `RodioResampler`), it has no effect on chunk resamplers (`InterleavedResampler` and `PlanarResampler`).
    #[must_use]
    #[cfg(feature = "realtime")]
    pub fn with_realtime_max_channels(mut self, realtime_max_channels: usize) -> Self {
        self.realtime_max_channels = realtime_max_channels;
        self
    }

    /// For `RodioResampler`, this setting controls whether to use a fast start mode.
    ///
    /// Fast start mode will prime the resampler with initial samples to get it up to speed, and avoid start-up silence.
    /// This is only appropriate to use when the inner sounce can handle rapid calls to `next()`. For example, this will
    /// generally work on buffered streams or audio files, but not on live microphones.
    ///
    ///   - Set to "true" if the inner source is something like a buffered stream or audio file.
    ///   - Set to "false" if the inner source is very realtime (e.g. a live microphone).
    ///
    /// If set to `true` for an inner source that cannot handle this, you will experience crackling at the start of the stream as the inner source fails to keep up.
    ///
    /// This setting is only for `RodioResampler`, it has no effect on other resamplers.
    #[must_use]
    #[cfg(feature = "rodio")]
    pub fn with_rodio_fast_start(mut self, rodio_fast_start: bool) -> Self {
        self.rodio_fast_start = rodio_fast_start;
        self
    }

    /// Validates all user-facing configuration fields.
    ///
    /// Returns `Ok(())` when all values are in range, or a specific `Error` describing the first
    /// invalid field encountered.
    pub fn validate(&self) -> Result<(), Error> {
        // Special "you didnt configure your preset message"
        if self.input_sample_rate == 0 && self.output_sample_rate == 0 && self.channels == 0 {
            return Err(Error::PresetNotConfigured);
        }

        if self.input_sample_rate == 0 {
            return Err(Error::MustSetInputSampleRate);
        }
        if self.output_sample_rate == 0 {
            return Err(Error::MustSetOutputSampleRate);
        }

        if self.channels == 0 {
            return Err(Error::InvalidChannels(self.channels));
        }

        if self.quality == 0 {
            return Err(Error::InvalidQuality(self.quality));
        }

        if !(0.0..=1.0).contains(&self.bandwidth) || !self.bandwidth.is_finite() {
            return Err(Error::InvalidBandwidth(self.bandwidth));
        }

        if !(-1.0..=1.0).contains(&self.phase) || !self.phase.is_finite() {
            return Err(Error::InvalidPhase(self.phase));
        }

        if !(0.0..=100.0).contains(&self.phase_intensity) || !self.phase_intensity.is_finite() {
            return Err(Error::InvalidPhaseIntensity(self.phase_intensity));
        }

        if let TaperType::Cosine(alpha) = self.taper_type
            && (alpha <= 0.0 || !alpha.is_finite())
        {
            return Err(Error::InvalidAlpha(alpha));
        }

        Ok(())
    }

    /// Computes derived FFT/chunk geometry from validated user configuration.
    ///
    /// Returns `DerivedConfig` for processing, or an error if validation fails.
    pub(crate) fn derive_config<T>(&self) -> Result<DerivedConfig<T>, Error>
    where
        T: Float,
    {
        self.validate()?;

        // Detect `T == f32` without specialization: only `f32` shares IEEE single max with `f32::MAX`.
        if let Some(f32_max) = num_traits::NumCast::from(f32::MAX) {
            if <T as Float>::max_value() == f32_max && self.quality > 8192 {
                return Err(Error::QualityTooHighForF32);
            }
        }

        Ok(DerivedConfig::from_config(self))
    }

    /// Returns the required input and output buffer sizes for realtime resampling.
    ///
    /// Returns (input_buffer_size, output_buffer_size)
    #[cfg(feature = "realtime")]
    pub(crate) fn realtime_buffer_sizes(&self) -> (usize, usize) {
        let quality = self.quality;
        let fixed_output_rate = self.output_sample_rate;
        let min_input_rate = self.realtime_input_range.start;
        let max_input_rate = self.realtime_input_range.end;

        let mut a = max_input_rate;
        let mut b = fixed_output_rate;
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        debug_assert!(a != 0, "expected non-zero gcd for input/output rates");
        let input_common_divisor = a;
        let mut input_chunk_frames = max_input_rate / input_common_divisor;
        let output_chunk_frames_for_input = fixed_output_rate / input_common_divisor;
        let input_denominator = input_chunk_frames.min(output_chunk_frames_for_input).max(1);
        let input_factor = quality.div_ceil(input_denominator).next_multiple_of(2);
        input_chunk_frames *= input_factor;

        let mut a = min_input_rate;
        let mut b = fixed_output_rate;
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        debug_assert!(a != 0, "expected non-zero gcd for input/output rates");
        let output_common_divisor = a;
        let input_chunk_frames_for_output = min_input_rate / output_common_divisor;
        let mut output_chunk_frames = fixed_output_rate / output_common_divisor;
        let output_denominator = input_chunk_frames_for_output.min(output_chunk_frames).max(1);
        let output_factor = quality.div_ceil(output_denominator).next_multiple_of(2);
        output_chunk_frames *= output_factor;

        (
            input_chunk_frames * self.realtime_max_channels,
            output_chunk_frames * self.realtime_max_channels,
        )
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DerivedConfig<T> {
    pub(crate) input_sample_rate: usize,
    pub(crate) output_sample_rate: usize,
    pub(crate) input_chunk_frames: usize,
    pub(crate) output_chunk_frames: usize,
    pub(crate) input_fft_size: usize,
    pub(crate) output_fft_size: usize,
    pub(crate) input_offset: usize,
    pub(crate) output_offset: usize,
    pub(crate) cutoff_bins: usize,
    pub(crate) taper_bins: usize,
    pub(crate) taper: Vec<T>,
    pub(crate) phase: Vec<Complex<T>>,
    pub(crate) phase_enabled: bool,
}

impl<T> DerivedConfig<T>
where
    T: Float,
{
    /// Expands user-facing configuration into internal chunk/FFT dimensions.
    ///
    /// Returns a fully populated `DerivedConfig` with rate-reduced chunk sizes and offsets.
    fn from_config(config: &Config) -> Self {
        let common_divisor = gcd(config.input_sample_rate, config.output_sample_rate);
        let mut input_chunk_frames = config.input_sample_rate / common_divisor;
        let mut output_chunk_frames = config.output_sample_rate / common_divisor;

        let denominator = input_chunk_frames.min(output_chunk_frames);
        let max_chunk_frames = 2 * input_chunk_frames.max(output_chunk_frames);
        let max_factor = i32::MAX as usize / max_chunk_frames;
        let mut factor = config.quality.div_ceil(denominator).min(max_factor);
        factor += factor & 1;
        input_chunk_frames *= factor;
        output_chunk_frames *= factor;

        let input_fft_size = input_chunk_frames * 2;
        let output_fft_size = output_chunk_frames * 2;
        let input_offset = (input_fft_size - input_chunk_frames) / 2;
        let output_offset = (output_fft_size - output_chunk_frames) / 2;
        let cutoff_bins = input_chunk_frames.min(output_chunk_frames) + 1;
        let taper_bins = (cutoff_bins as f64 * (1.0 - f64::from(config.bandwidth))).ceil() as usize;
        let is_passthrough = config.input_sample_rate == config.output_sample_rate;
        let taper = Self::build_taper(
            input_fft_size,
            cutoff_bins,
            taper_bins,
            is_passthrough,
            config.taper_type,
        );

        let phase_value = T::from(config.phase).unwrap_or_else(T::zero);
        let phase_intensity = T::from(config.phase_intensity).unwrap_or_else(T::zero);
        let phase = Self::build_phase(cutoff_bins, phase_value, phase_intensity);
        let phase_enabled = !phase_value.is_zero() && !phase_intensity.is_zero();

        Self {
            input_sample_rate: config.input_sample_rate,
            output_sample_rate: config.output_sample_rate,
            input_chunk_frames,
            output_chunk_frames,
            input_fft_size,
            output_fft_size,
            input_offset,
            output_offset,
            cutoff_bins,
            taper_bins,
            taper,
            phase,
            phase_enabled,
        }
    }

    /// Builds the per-bin unit complex phase rotation used before tapering.
    fn build_phase(bins: usize, phase: T, phase_intensity: T) -> Vec<Complex<T>> {
        if bins == 0 {
            return Vec::new();
        }

        let magnitude = phase.abs();
        let sign = if phase < T::zero() { -T::one() } else { T::one() };
        let denominator = T::from(bins).unwrap_or_else(T::one);

        (0..bins)
            .map(|idx| {
                let x = T::from(idx).unwrap_or_else(T::zero) / denominator;
                let angle = (magnitude * x).asin() * phase_intensity * sign;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect()
    }

    fn build_taper(
        input_fft_size: usize,
        cutoff_bin: usize,
        taper_bins: usize,
        is_passthrough: bool,
        taper_type: TaperType,
    ) -> Vec<T> {
        match taper_type {
            TaperType::Planck => Self::build_planck_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough),
            TaperType::Cosine(alpha) => {
                Self::build_cosine_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough, alpha)
            }
        }
    }

    /// Builds a Planck-taper frequency mask.
    ///
    /// Returns passband unity bins, a Planck-taper transition, and stopband zeros.
    fn build_planck_taper(input_fft_size: usize, cutoff_bin: usize, taper_bins: usize, is_passthrough: bool) -> Vec<T> {
        let mut taper = vec![T::zero(); input_fft_size / 2 + 1];

        if is_passthrough {
            taper.fill(T::one());
            return taper;
        }

        let transition = if taper_bins == 0 {
            Vec::new()
        } else if taper_bins == 1 {
            vec![T::one()]
        } else {
            let denom = T::from(taper_bins).unwrap() - T::one();

            let raw: Vec<T> = (0..taper_bins)
                .map(|idx| {
                    if idx == 0 {
                        return T::one();
                    }

                    if idx == taper_bins - 1 {
                        return T::zero();
                    }

                    let x = T::from(idx).unwrap_or_else(T::zero) / denom;

                    // Descending Planck taper
                    let z = T::one() / x - T::one() / (T::one() - x);
                    let rising = T::one() / (z.exp() + T::one());

                    let value = T::one() - rising;

                    if value.is_normal() {
                        value
                    } else if value >= T::one() {
                        T::one()
                    } else {
                        T::zero()
                    }
                })
                .collect();

            let trim_start = raw.iter().position(|value| *value < T::one()).unwrap_or(raw.len());

            let trim_stop = raw
                .iter()
                .rposition(|value| *value > T::zero())
                .map_or(0, |idx| raw.len() - idx - 1);

            let active_end = raw.len().saturating_sub(trim_stop);

            raw[trim_start..active_end].to_vec()
        };

        let taper_start = cutoff_bin.saturating_sub(transition.len());

        for (idx, value) in taper.iter_mut().enumerate() {
            if idx < taper_start {
                *value = T::one();
            } else if idx < cutoff_bin {
                *value = transition[idx - taper_start];
            } else {
                *value = T::zero();
            }
        }

        taper
    }

    /// Builds a sigmoid-warped cosine frequency taper.
    ///
    /// Returns passband unity bins, a trimmed warped-cosine transition, and stopband zeros.
    fn build_cosine_taper(
        input_fft_size: usize,
        cutoff_bin: usize,
        taper_bins: usize,
        is_passthrough: bool,
        alpha: f32,
    ) -> Vec<T> {
        let mut taper = vec![T::zero(); input_fft_size / 2 + 1];

        if is_passthrough {
            taper.fill(T::one());
            return taper;
        }

        let transition = if taper_bins == 0 {
            Vec::new()
        } else if taper_bins == 1 {
            vec![T::one()]
        } else {
            let pi = T::from(std::f64::consts::PI).unwrap_or_else(T::zero);
            let two = T::one() + T::one();
            let alpha = T::from(alpha).unwrap_or_else(T::one);
            let denom = T::from(taper_bins).unwrap() - T::one();

            let raw: Vec<T> = (0..taper_bins)
                .map(|idx| {
                    let x = T::from(idx).unwrap_or_else(T::zero) / denom;

                    // Powered sigmoid warp:
                    //
                    //     x_warped = x^a / (x^a + (1 - x)^a)
                    //
                    // This preserves endpoints but concentrates most of the transition
                    // around the middle, making the cosine behave more like the
                    // trimmed logistic taper.
                    let a = x.powf(alpha);
                    let b = (T::one() - x).powf(alpha);
                    let warped = a / (a + b);

                    let value = (T::one() + (pi * warped).cos()) / two;

                    if value.is_normal() {
                        value
                    } else if value == T::one() {
                        T::one()
                    } else {
                        T::zero()
                    }
                })
                .collect();

            let trim_start = raw.iter().position(|value| *value < T::one()).unwrap_or(raw.len());

            let trim_stop = raw
                .iter()
                .rposition(|value| *value > T::zero())
                .map_or(0, |idx| raw.len() - idx - 1);

            let active_end = raw.len().saturating_sub(trim_stop);

            raw[trim_start..active_end].to_vec()
        };

        let taper_start = cutoff_bin.saturating_sub(transition.len());

        for (idx, value) in taper.iter_mut().enumerate() {
            if idx < taper_start {
                *value = T::one();
            } else if idx < cutoff_bin {
                *value = transition[idx - taper_start];
            } else {
                *value = T::zero();
            }
        }

        taper
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_no_nans;
    #[test]
    fn derives_chunk_sizes_from_reduced_rates() {
        let config = Config::new(44_100, 48_000, 2);
        let derived = config.derive_config::<f32>().unwrap();

        assert_eq!(derived.input_sample_rate, 44_100);
        assert_eq!(derived.output_sample_rate, 48_000);
        assert_eq!(derived.input_chunk_frames, 2058);
        assert_eq!(derived.output_chunk_frames, 2240);
        assert_eq!(derived.input_fft_size, 4116);
        assert_eq!(derived.output_fft_size, 4480);
        assert_eq!(derived.input_offset, 1029);
        assert_eq!(derived.output_offset, 1120);
        assert_eq!(derived.cutoff_bins, 2059);
        assert_eq!(derived.taper_bins, 183);
    }

    #[test]
    fn derives_chunk_sizes_with_c_filter_factor_rule() {
        let config = Config::new(44_100, 96_000, 2);
        let derived = config.derive_config::<f32>().unwrap();

        assert_eq!(derived.input_sample_rate, 44_100);
        assert_eq!(derived.output_sample_rate, 96_000);
        assert_eq!(derived.input_chunk_frames, 2058);
        assert_eq!(derived.output_chunk_frames, 4480);
        assert_eq!(derived.input_fft_size, 4116);
        assert_eq!(derived.output_fft_size, 8960);
        assert_eq!(derived.input_offset, 1029);
        assert_eq!(derived.output_offset, 2240);
        assert_eq!(derived.cutoff_bins, 2059);
        assert_eq!(derived.taper_bins, 183);
    }

    #[test]
    fn taper_has_expected_rolloff_shape() {
        for taper_type in [TaperType::Cosine(3.45), TaperType::Planck] {
            let config = Config {
                input_sample_rate: 48_000,
                output_sample_rate: 44_100,
                channels: 1,
                quality: 64,
                bandwidth: 0.95,
                taper_type,
                ..Config::default()
            };
            let derived = config.derive_config::<f32>().unwrap();
            let taper = &derived.taper;
            assert_no_nans(taper, "config::taper_has_expected_rolloff_shape taper");
            let taper_bins = derived.taper_bins.max(1);
            let transition_start = derived.cutoff_bins.saturating_sub(taper_bins);

            assert!(taper.iter().all(|value| *value >= 0.0 && *value <= 1.0));
            assert_eq!(taper[transition_start - 1], 1.0);
            assert_eq!(taper[derived.cutoff_bins], 0.0);
            assert!(taper[..transition_start].iter().all(|value| *value == 1.0));
            assert!(taper[derived.cutoff_bins..].iter().all(|value| *value == 0.0));
            assert!(
                taper[transition_start..derived.cutoff_bins]
                    .windows(2)
                    .all(|pair| pair[0] >= pair[1])
            );
        }
    }

    #[test]
    fn passthrough_taper_is_all_ones() {
        for taper_type in [TaperType::Cosine(3.45), TaperType::Planck] {
            let config = Config {
                input_sample_rate: 48_000,
                output_sample_rate: 48_000,
                channels: 1,
                quality: 64,
                bandwidth: 0.75,
                taper_type,
                ..Config::default()
            };
            let derived = config.derive_config::<f32>().unwrap();
            assert_no_nans(&derived.taper, "config::passthrough_taper_is_all_ones taper");

            assert_eq!(derived.taper.len(), derived.input_fft_size / 2 + 1);
            assert!(derived.taper.iter().all(|value| *value == 1.0));
        }
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(matches!(
            Config::new(0, 48_000, 2).validate(),
            Err(Error::MustSetInputSampleRate)
        ));

        assert!(matches!(
            Config::new(48_000, 0, 2).validate(),
            Err(Error::MustSetOutputSampleRate)
        ));

        assert!(matches!(
            Config::new(44_100, 48_000, 0).validate(),
            Err(Error::InvalidChannels(0))
        ));

        let config = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            bandwidth: f32::NAN,
            ..Config::default()
        };
        assert!(matches!(config.validate(), Err(Error::InvalidBandwidth(_))));

        let zero_alpha = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            taper_type: TaperType::Cosine(0.0),
            ..Config::default()
        };
        assert!(matches!(zero_alpha.validate(), Err(Error::InvalidAlpha(0.0))));

        let negative_alpha = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            taper_type: TaperType::Cosine(-1.0),
            ..Config::default()
        };
        assert!(matches!(negative_alpha.validate(), Err(Error::InvalidAlpha(-1.0))));

        let non_finite_alpha = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            taper_type: TaperType::Cosine(f32::NAN),
            ..Config::default()
        };
        assert!(matches!(
            non_finite_alpha.validate(),
            Err(Error::InvalidAlpha(alpha)) if alpha.is_nan()
        ));

        for phase in [-1.0, 0.0, 1.0] {
            let config = Config {
                input_sample_rate: 48_000,
                output_sample_rate: 48_000,
                phase,
                ..Config::default()
            };
            assert!(config.validate().is_ok());
        }

        for phase in [-1.0001, 1.0001] {
            let config = Config {
                input_sample_rate: 48_000,
                output_sample_rate: 48_000,
                phase,
                ..Config::default()
            };
            assert!(matches!(config.validate(), Err(Error::InvalidPhase(value)) if value == phase));
        }

        let non_finite_phase = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            phase: f32::NAN,
            ..Config::default()
        };
        assert!(matches!(
            non_finite_phase.validate(),
            Err(Error::InvalidPhase(phase)) if phase.is_nan()
        ));

        for phase_intensity in [0.0, Config::DEFAULT.phase_intensity, 100.0] {
            let config = Config {
                input_sample_rate: 48_000,
                output_sample_rate: 48_000,
                phase_intensity,
                ..Config::default()
            };
            assert!(config.validate().is_ok());
        }

        for phase_intensity in [-0.0001, 100.0001] {
            let config = Config {
                input_sample_rate: 48_000,
                output_sample_rate: 48_000,
                phase_intensity,
                ..Config::default()
            };
            assert!(matches!(config.validate(), Err(Error::InvalidPhaseIntensity(value)) if value == phase_intensity));
        }

        let non_finite_phase_intensity = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            phase_intensity: f32::NAN,
            ..Config::default()
        };
        assert!(matches!(
            non_finite_phase_intensity.validate(),
            Err(Error::InvalidPhaseIntensity(phase_intensity)) if phase_intensity.is_nan()
        ));
    }

    #[test]
    fn phase_table_uses_c_rotation_formula() {
        let phase_intensity = Config::DEFAULT.phase_intensity;
        let identity = DerivedConfig::<f32>::build_phase(4, 0.0, phase_intensity);
        assert_eq!(identity.len(), 4);
        assert!(identity.iter().all(|phase| phase.re == 1.0 && phase.im == 0.0));

        let zero_intensity = DerivedConfig::<f32>::build_phase(4, 0.5, 0.0);
        assert!(zero_intensity.iter().all(|phase| phase.re == 1.0 && phase.im == 0.0));

        let positive = DerivedConfig::<f32>::build_phase(4, 0.5, phase_intensity);
        let negative = DerivedConfig::<f32>::build_phase(4, -0.5, phase_intensity);

        assert_eq!(positive[0].re, 1.0);
        assert_eq!(positive[0].im, 0.0);

        let expected_angle = (0.5f32 * (1.0 / 4.0)).asin() * phase_intensity;
        assert!((positive[1].re - expected_angle.cos()).abs() < 1e-6);
        assert!((positive[1].im - expected_angle.sin()).abs() < 1e-6);

        for (pos, neg) in positive.iter().zip(negative.iter()) {
            let magnitude = (pos.re * pos.re + pos.im * pos.im).sqrt();
            assert!(pos.re.is_finite());
            assert!(pos.im.is_finite());
            assert!((magnitude - 1.0).abs() < 1e-6);
            assert!((pos.re - neg.re).abs() < 1e-6);
            assert!((pos.im + neg.im).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_quality_above_8192_for_f32_derived_config() {
        let config = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            quality: 8193,
            ..Config::default()
        };
        assert!(matches!(
            config.derive_config::<f32>(),
            Err(Error::QualityTooHighForF32)
        ));
    }

    #[test]
    fn allows_quality_8192_for_f32_derived_config() {
        let config = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            quality: 8192,
            ..Config::default()
        };
        assert!(config.derive_config::<f32>().is_ok());
    }

    #[test]
    fn allows_high_quality_for_f64_derived_config() {
        let config = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            quality: 65_536,
            ..Config::default()
        };
        assert!(config.derive_config::<f64>().is_ok());
    }

    #[test]
    fn taper_is_all_ones_when_passthrough() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 8, 4, true, taper_type);
            assert_no_nans(&taper, "config::taper_is_all_ones_when_passthrough taper");

            assert_eq!(taper.len(), 9);
            assert!(taper.iter().all(|v| *v == 1.0));
        }
    }

    #[test]
    fn taper_has_expected_passband_transition_and_stopband() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 6, 4, false, taper_type);
            assert_no_nans(
                &taper,
                "config::taper_has_expected_passband_transition_and_stopband taper",
            );

            assert_eq!(taper.len(), 9);

            // cutoff_bin = 6, transition length should occupy bins before it.
            // So bins >= 6 are stopband.
            assert_eq!(taper[6], 0.0);
            assert_eq!(taper[7], 0.0);
            assert_eq!(taper[8], 0.0);

            // Early bins should be passband.
            assert_eq!(taper[0], 1.0);
            assert_eq!(taper[1], 1.0);

            // Transition should be descending.
            assert!(taper[2] >= taper[3]);
            assert!(taper[3] >= taper[4]);
            assert!(taper[4] >= taper[5]);

            assert!(taper[2] <= 1.0);
            assert!(taper[5] >= 0.0);
        }
    }

    #[test]
    fn transition_is_descending_and_bounded() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let cutoff_bin = 24;
            let taper = DerivedConfig::<f32>::build_taper(64, cutoff_bin, 16, false, taper_type);
            assert_no_nans(&taper, "config::transition_is_descending_and_bounded taper");
            let transition_start = taper
                .iter()
                .position(|value| *value < 1.0)
                .expect("expected transition start");
            let transition = &taper[transition_start..cutoff_bin];

            assert!(!transition.is_empty());

            for value in transition {
                assert!(*value >= 0.0);
                assert!(*value <= 1.0);
            }

            for pair in transition.windows(2) {
                assert!(pair[0] >= pair[1]);
            }

            assert!(transition.first().unwrap() < &1.0);
            assert!(transition.last().unwrap() > &0.0);
        }
    }

    #[test]
    fn zero_taper_bins_produces_hard_cutoff() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 6, 0, false, taper_type);
            assert_no_nans(&taper, "config::zero_taper_bins_produces_hard_cutoff taper");

            assert_eq!(taper.len(), 9);

            for idx in 0..6 {
                assert_eq!(taper[idx], 1.0);
            }

            for idx in 6..taper.len() {
                assert_eq!(taper[idx], 0.0);
            }
        }
    }

    #[test]
    fn one_taper_bin_keeps_single_unity_transition_bin() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 6, 1, false, taper_type);
            assert_no_nans(&taper, "config::one_taper_bin_keeps_single_unity_transition_bin taper");

            assert_eq!(taper[5], 1.0);
            assert_eq!(taper[6], 0.0);
        }
    }

    #[test]
    fn taper_handles_cutoff_smaller_than_transition_width() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 2, 8, false, taper_type);

            assert_eq!(taper.len(), 9);

            // No panic, and cutoff still respected.
            for idx in 2..taper.len() {
                assert_eq!(taper[idx], 0.0);
            }

            assert!(taper[0] <= 1.0);
            assert!(taper[0] >= 0.0);
            assert!(taper[1] <= 1.0);
            assert!(taper[1] >= 0.0);
        }
    }
}
