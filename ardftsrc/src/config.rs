use crate::TaperType;
use crate::dsp_float::DspFloat;
use crate::extrapolation::Extrapolation;
use crate::spectral::SpectralPlan;

/// Low-latency, lower-quality preset.
///
/// You may prefer using a sinc resampler (eg. rubato) instead.
///
/// Quality reports: [f64](https://github.com/phayes/ardftsrc-rs/blob/master/reports/report_fast.md),
/// [f32](https://github.com/phayes/ardftsrc-rs/blob/master/reports/report_fast_f32.md).
///
/// **[HydrogenAudio Test Suite (f32)](https://src.hydrogenaudio.org/compareresults?id1=4b0b2ab6-5a10-4f5a-80fa-81e3d131269d&id2=0)**
/// - Overall Score: 71.83%
/// - Spectrogram Score: 51.73%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 93.95%
/// - Intermodulation Distortion Score: 35.62%
/// - Impulse Frequency Score: 50.69%
/// - Pre-ringing Score: 21.63%
/// - Gapless Sine Score: 100%
///
/// **[HydrogenAudio Test Suite (f64)](https://src.hydrogenaudio.org/compareresults?id1=072838ff-9072-46d3-aca3-ebe6f765f2dd&id2=0)**
/// - Overall Score: 97.69%
/// - Spectrogram Score: 96.22%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 93.95%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 99.62%
/// - Pre-ringing Score: 21.63%
/// - Gapless Sine Score: 100%
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
    bandwidth: 0.92,
    taper_type: TaperType::Kaiser(30.0),
    ..Config::DEFAULT
};

/// Balanced preset for good realtime quality. ***You should probably use this one.***
///
/// Quality reports: [f64](https://github.com/phayes/ardftsrc-rs/blob/master/reports/report_good.md).
///
/// **[HydrogenAudio Test Suite (f64)](https://src.hydrogenaudio.org/compareresults?id1=607244af-95ac-4005-a745-cff648b9298f&id2=0)**
/// - Overall Score: 99.26%
/// - Spectrogram Score: 98.91%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 98.14%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 99.54%
/// - Pre-ringing Score: 18.28%
/// - Gapless Sine Score: 100%
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
    quality: 2304,
    bandwidth: 0.975,
    taper_type: TaperType::Kaiser(30.0),
    ..Config::DEFAULT
};

/// High quality preset suitable for realtime applications where quality is critical, or for offline processing.
///
/// The alias floor is `0` dB, so the transition is centred on the lower Nyquist: only content in the
/// top ~0.4% of the band folds back, at -6 dB or less.
///
/// Quality reports: [f64](https://github.com/phayes/ardftsrc-rs/blob/master/reports/report_high.md).
///
/// **[HydrogenAudio Test Suite (f64)](https://src.hydrogenaudio.org/compareresults?id1=25aa9380-2193-44d1-a452-1b9b488f9735&id2=0)**
/// - Overall Score: 99.86%
/// - Spectrogram Score: 99.66%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 100%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 99.57%
/// - Pre-ringing Score: 17.74%
/// - Gapless Sine Score: 100%
///
/// **[HydrogenAudio Test Suite (`DoubleDouble`)](https://src.hydrogenaudio.org/compareresults?id1=b91d0f01-0e50-4967-90f4-68a86faa2312&id2=0)**
/// - Overall Score: 99.92%
/// - Spectrogram Score: 99.66%
/// - Aliasing Score: 100%
/// - Nyquist Filter Score: 100%
/// - Intermodulation Distortion Score: 100%
/// - Impulse Frequency Score: 100%
/// - Pre-ringing Score: 17.74%
/// - Gapless Sine Score: 100%
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
    quality: 7056,
    bandwidth: 0.996,
    taper_type: TaperType::Kaiser(34.0),
    alias_floor: AliasFloor::Decibels(0.0),
    ..Config::DEFAULT
};

use crate::Error;

/// Lowest frequency that the resampler may fold (downsampling) or image (upsampling) energy
/// onto. See [`Config::alias_floor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AliasFloor {
    /// Fraction of the lower Nyquist frequency in `[bandwidth, 1.0]`, in the same units as
    /// [`Config::bandwidth`]. `1.0` disables aliasing.
    Fraction(f32),

    /// Fold/image only down to the frequency where the final filter response is `db` dB
    /// (must be at most `0.0`, e.g. `-3.0`; `0.0` is equivalent to `Fraction(bandwidth)`).
    /// Resolved when the resampler is constructed, using the configured bandwidth and taper.
    Decibels(f32),
}

impl Default for AliasFloor {
    fn default() -> Self {
        Self::Fraction(1.0)
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
    /// Default value is 2304 (same quality as PRESET_GOOD).
    ///
    /// Value guide:
    ///  - `512` (PRESET_FAST):       Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    ///  - `2304` (PRESET_GOOD):      Good balanced quality - you should probably use this. (Default)
    ///  - `7056` (PRESET_HIGH):      High quality, good for realtime applications where quality is critical, or for offline resampling. Alias floor is `0` dB.
    pub quality: usize,

    /// Normalized filter bandwidth in the range `[0.0, 1.0]`.
    ///
    /// Higher values preserve more high-frequency content but shorten the transition band.
    ///
    /// Default value is `0.975` (same bandwidth as [`PRESET_GOOD`]).
    ///
    /// Value guide:
    /// - `0.92`: Fast and low quality, great for realtime applications ([`PRESET_FAST`]). At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    /// - `0.975`: Balanced high-end retention for most cases ([`PRESET_GOOD`]).
    /// - `0.99`: Very aggressive high-end retention; Only recommended when using a higher "quality" setting.
    pub bandwidth: f32,

    /// Frequency taper profile used around the cutoff region.
    ///
    /// - `Planck`: Uses a Planck taper transition.
    /// - `Kaiser(beta)`: Exactly integrated Kaiser-window CDF; -6 dB at the midpoint (Kaiser `beta` at most 300).
    /// - `Tanh(alpha)`: Endpoint-normalized hyperbolic tangent transition.
    /// - `Cosine(alpha)`: Uses a sigmoid-warped cosine transition.
    /// - `BetaCdf(alpha, beta)`: Beta-CDF taper from the regularized lower incomplete beta function.
    ///
    /// Default value is `Kaiser(30.0)` (same taper as [`PRESET_GOOD`]).
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

    /// Alias floor: permits controlled aliasing/imaging inside the low-pass transition band.
    ///
    /// By default the transition ends at the lower Nyquist frequency and content beyond it is
    /// suppressed. Lowering the floor extends the transition past Nyquist (mirrored about it), so
    /// it is wider and rings less. Energy in the extended region is folded back (downsampling) or
    /// imaged (upsampling), but never below the floor, and the floor never goes below the
    /// passband edge set by [`bandwidth`](Config::bandwidth).
    ///
    /// This is not the same as removing the low-pass filter: the passband is unchanged and
    /// content above the extended stopband is still suppressed.
    ///
    /// - `AliasFloor::Fraction(1.0)`: no aliasing (default).
    /// - `AliasFloor::Decibels(-3.0)`: similar to SoX `rate -a`.
    /// - `AliasFloor::Fraction(bandwidth)`: widest transition, centred on the lower Nyquist so
    ///   each pair of bins mirrored about it is complementary.
    ///
    /// Pre-decimation stages (see [`decimate`](Config::decimate)) always stay strict.
    pub alias_floor: AliasFloor,

    /// EXPERIMENTAL: Enables an optional 2:1 pre-decimation stage ahead of the FFT resampler for very large
    /// downsampling ratios (e.g. 192kHz -> 48kHz).
    ///
    /// When enabled, it inserts one or more cheap
    /// time-domain 2:1 decimation stages before the FFT resampler, chosen so the FFT stage still
    /// performs at least one genuine 2:1 reduction of its own. Each decimation stage halves the
    /// working sample rate, which shrinks the FFT chunk/window size (and therefore streaming
    /// buffer requirements and algorithmic latency) roughly in proportion.
    ///
    /// Each decimation stage reuses [`bandwidth`](Config::bandwidth) to decide how much guard
    /// band to keep below its own post-decimation Nyquist frequency, so it makes the same
    /// quality tradeoff already implied by that setting.
    /// Decimation only ever engages when downsampling by at least 4x.
    /// For smaller ratios it has no effect.
    ///
    /// Default value is `false`.
    pub decimate: bool,

    /// Strategy used to synthesize missing start/stop-edge samples whenever real `pre`/`post`
    /// context (set via a core's `pre()`/`post()` methods) doesn't cover everything a window
    /// needs.
    ///
    /// Default value is [`Extrapolation::Lpc`].
    pub extrapolation: Extrapolation,

    /// For [`RodioResampler`](crate::RodioResampler), this setting controls whether to use a fast start mode.
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
    /// When the inner source changes sample rate or channel count, the new span is primed the same way, so the format change
    /// plays without a gap.
    ///
    /// This setting is only for [`RodioResampler`](crate::RodioResampler), it has no effect on other resamplers.
    #[cfg(feature = "rodio")]
    pub rodio_fast_start: bool,
}

impl Config {
    pub const DEFAULT: Self = Self {
        input_sample_rate: 0,
        output_sample_rate: 0,
        channels: 2,
        quality: 2304,
        bandwidth: 0.975,
        taper_type: TaperType::Kaiser(30.0),
        phase: 0.0,
        phase_intensity: 50.0,
        alias_floor: AliasFloor::Fraction(1.0),
        decimate: false,
        extrapolation: Extrapolation::Lpc,
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
    /// Default value is 2304 (same quality as PRESET_GOOD).
    ///
    /// Value guide:
    ///  - `512` (PRESET_FAST):       Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    ///  - `2304` (PRESET_GOOD):      Good balanced quality - you should probably use this. (Default)
    ///  - `7056` (PRESET_HIGH):      High quality, good for realtime applications where quality is critical, or for offline resampling. Alias floor is `0` dB.
    #[must_use]
    pub fn with_quality(mut self, quality: usize) -> Self {
        self.quality = quality;
        self
    }

    /// Normalized filter bandwidth in the range `[0.0, 1.0]`.
    ///
    /// Higher values preserve more high-frequency content but shorten the transition band.
    ///
    /// Default value is `0.975` (same bandwidth as [`PRESET_GOOD`]).
    ///
    /// Value guide:
    /// - `0.92`: Fast and low quality, great for realtime applications ([`PRESET_FAST`]). At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    /// - `0.975`: Balanced high-end retention for most cases ([`PRESET_GOOD`]).
    /// - `0.99`: Very aggressive high-end retention; Only recommended when using a higher "quality" setting.
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: f32) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Frequency taper profile used around the cutoff region.
    ///
    /// - `Planck`: Uses a Planck taper transition.
    /// - `Kaiser(beta)`: Exactly integrated Kaiser-window CDF; -6 dB at the midpoint (Kaiser `beta` at most 300).
    /// - `Tanh(alpha)`: Endpoint-normalized hyperbolic tangent transition.
    /// - `Cosine(alpha)`: Uses a sigmoid-warped cosine transition.
    /// - `BetaCdf(alpha, beta)`: Beta-CDF taper from the regularized lower incomplete beta function.
    ///
    /// Default value is `Kaiser(30.0)` (same taper as [`PRESET_GOOD`]).
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

    /// Sets the alias floor as a fraction of the lower Nyquist frequency in `[bandwidth, 1.0]`.
    ///
    /// `1.0` (default) disables aliasing. See [`alias_floor`](Config::alias_floor).
    #[must_use]
    pub fn with_alias_floor(mut self, fraction: f32) -> Self {
        self.alias_floor = AliasFloor::Fraction(fraction);
        self
    }

    /// Sets the alias floor to the frequency where the final filter response is `db` dB
    /// (must be at most `0.0`). `-3.0` is similar to SoX `rate -a`.
    ///
    /// Resolved when the resampler is constructed, so it always reflects the final bandwidth and
    /// taper regardless of builder call order. See [`alias_floor`](Config::alias_floor).
    #[must_use]
    pub fn with_alias_floor_db(mut self, db: f32) -> Self {
        self.alias_floor = AliasFloor::Decibels(db);
        self
    }

    /// For [`RodioResampler`](crate::RodioResampler), this setting controls whether to use a fast start mode.
    ///
    /// Fast start mode will prime the resampler with initial samples to get it up to speed, and avoid start-up silence.
    /// This is only appropriate to use when the inner sounce can handle rapid calls to `next()`. For example, this will
    /// generally work on buffered streams or audio files, but not on live microphones.
    ///
    ///   - Set to `true` if the inner source is something like a buffered stream or audio file.
    ///   - Set to `false` if the inner source is very realtime (e.g. a live microphone).
    ///
    /// If set to `true` for an inner source that cannot handle this, you will experience crackling at the start of the stream as the inner source fails to keep up.
    ///
    /// When the inner source changes sample rate or channel count, the new span is primed the same way, so the format change
    /// plays without a gap.
    ///
    /// This setting is only for [`RodioResampler`](crate::RodioResampler), it has no effect on other resamplers.
    #[must_use]
    #[cfg(feature = "rodio")]
    pub fn with_rodio_fast_start(mut self, rodio_fast_start: bool) -> Self {
        self.rodio_fast_start = rodio_fast_start;
        self
    }

    /// EXPERIMENTAL: Enables an optional 2:1 pre-decimation stage ahead of the FFT resampler for very large
    /// downsampling ratios (e.g. 192kHz -> 48kHz).
    ///
    /// When enabled, it inserts one or more cheap
    /// time-domain 2:1 decimation stages before the FFT resampler, chosen so the FFT stage still
    /// performs at least one genuine 2:1 reduction of its own. Each decimation stage halves the
    /// working sample rate, which shrinks the FFT chunk/window size (and therefore streaming
    /// buffer requirements and algorithmic latency) roughly in proportion.
    ///
    /// Each decimation stage reuses [`bandwidth`](Config::bandwidth) to decide how much guard
    /// band to keep below its own post-decimation Nyquist frequency, so it makes the same
    /// quality tradeoff already implied by that setting.
    /// Decimation only ever engages when downsampling by at least 4x.
    /// For smaller ratios it has no effect.
    #[must_use]
    pub fn with_decimate(mut self, decimate: bool) -> Self {
        self.decimate = decimate;
        self
    }

    /// Strategy used to synthesize missing start/stop-edge samples whenever real `pre`/`post`
    /// context doesn't cover everything a window needs.
    ///
    /// Default value is [`Extrapolation::Lpc`].
    #[must_use]
    pub fn with_extrapolation(mut self, extrapolation: Extrapolation) -> Self {
        self.extrapolation = extrapolation;
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

        match self.alias_floor {
            AliasFloor::Fraction(fraction) => {
                if !fraction.is_finite() || fraction < self.bandwidth || fraction > 1.0 {
                    return Err(Error::InvalidAliasFloor(fraction));
                }
            }
            AliasFloor::Decibels(db) => {
                if !db.is_finite() || db > 0.0 {
                    return Err(Error::InvalidAliasFloorDb(db));
                }
            }
        }

        // Validate the taper type
        self.taper_type.validate()?;

        Ok(())
    }

    /// Computes derived FFT/chunk geometry from validated user configuration.
    ///
    /// Returns [`DerivedConfig`] for processing, or an error if validation fails.
    pub(crate) fn derive_config<T>(&self) -> Result<DerivedConfig<T>, Error>
    where
        T: DspFloat,
    {
        self.validate()?;

        // Single precision can't resolve the FFT sizes above this quality; anything at least as
        // precise as `f64` can.
        if T::machine_epsilon() > f64::EPSILON && self.quality > 8192 {
            return Err(Error::QualityTooHighForF32);
        }

        Ok(DerivedConfig::from_config(self))
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
    /// Precomputed filter geometry, gain, phase, and bin mapping for the FFT stage.
    pub(crate) spectral: SpectralPlan<T>,
    /// Number of cascaded 2:1 decimation stages to run ahead of the FFT resampler. Zero when
    /// [`Config::decimate`] is disabled or the rate ratio doesn't warrant it.
    pub(crate) decimation_stages: usize,
    /// FIR coefficients shared by every decimation stage (empty when `decimation_stages == 0`).
    pub(crate) decimation_taps: Vec<T>,
    /// The decimation cascade's group delay in whole output samples, rounded up; trimmed on top
    /// of `output_offset` at stream start. The rounding remainder is compensated by a matching
    /// fractional delay in `spectral`, so the two together are exact. Zero without decimation.
    pub(crate) decimator_output_delay: usize,
    /// Strategy used to synthesize missing start/stop-edge samples; see [`Config::extrapolation`].
    pub(crate) extrapolation: Extrapolation,
}

impl<T> DerivedConfig<T> {
    /// Returns the raw (pre-decimation) input chunk length in frames -- the number of frames a
    /// caller must supply per streaming chunk. Equals `input_chunk_frames` when decimation is
    /// disabled.
    #[inline]
    pub(crate) fn raw_input_chunk_frames(&self) -> usize {
        self.input_chunk_frames << self.decimation_stages
    }
}

impl<T> DerivedConfig<T>
where
    T: DspFloat,
{
    /// Expands user-facing configuration into internal chunk/FFT dimensions.
    ///
    /// Returns a fully populated [`DerivedConfig`] with rate-reduced chunk sizes and offsets.
    fn from_config(config: &Config) -> Self {
        let decimation_stages = if config.decimate {
            crate::decimate::decimation_stage_count(config.input_sample_rate, config.output_sample_rate)
        } else {
            0
        };
        // FFT geometry is derived from the *decimated* working rate; `input_sample_rate` below
        // keeps reporting the true (raw, pre-decimation) rate, since that's the domain callers
        // and `is_passthrough()` operate in.
        let effective_input_rate = config.input_sample_rate >> decimation_stages;

        let common_divisor = gcd(effective_input_rate, config.output_sample_rate);
        let mut input_chunk_frames = effective_input_rate / common_divisor;
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
        let decimation_taps = if decimation_stages > 0 {
            // Cap each stage's group delay to roughly the (decimated-domain) chunk size, so
            // flushing the cascade's trailing state at end-of-stream stays effectively lossless
            // (see `decimate::design_decimation_taps` and `CpuCore`'s finalize handling).
            crate::decimate::design_decimation_taps(config.bandwidth, input_chunk_frames)
        } else {
            Vec::new()
        };

        // The decimation cascade's group delay, converted to output samples at the true
        // (pre-decimation) rate ratio, is generally fractional. Startup trimming can only drop
        // whole samples, so it rounds up and the FFT stage delays the output by the remainder.
        let raw_decimator_delay = crate::decimate::raw_group_delay(decimation_stages, decimation_taps.len());
        let scaled_decimator_delay = raw_decimator_delay * config.output_sample_rate;
        let decimator_output_delay = scaled_decimator_delay.div_ceil(config.input_sample_rate);
        let decimator_delay_remainder = (decimator_output_delay * config.input_sample_rate - scaled_decimator_delay)
            as f64
            / config.input_sample_rate as f64;

        let spectral = SpectralPlan::new(
            input_chunk_frames,
            output_chunk_frames,
            config.bandwidth,
            &config.taper_type,
            f64::from(config.phase),
            f64::from(config.phase_intensity),
            crate::spectral::resolve_alias_floor(config.alias_floor, config.bandwidth, &config.taper_type),
            decimator_delay_remainder,
        );

        Self {
            input_sample_rate: config.input_sample_rate,
            output_sample_rate: config.output_sample_rate,
            input_chunk_frames,
            output_chunk_frames,
            input_fft_size,
            output_fft_size,
            input_offset,
            output_offset,
            spectral,
            decimation_stages,
            decimation_taps,
            decimator_output_delay,
            extrapolation: config.extrapolation,
        }
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
        assert_eq!(derived.input_chunk_frames, 2352);
        assert_eq!(derived.output_chunk_frames, 2560);
        assert_eq!(derived.input_fft_size, 4704);
        assert_eq!(derived.output_fft_size, 5120);
        assert_eq!(derived.input_offset, 1176);
        assert_eq!(derived.output_offset, 1280);
        assert_eq!(derived.spectral.geometry.stopband_end_bin, 2353);
        assert_eq!(derived.spectral.geometry.transition_bins(), 59);
    }

    #[test]
    fn derives_chunk_sizes_with_c_filter_factor_rule() {
        let config = Config::new(44_100, 96_000, 2);
        let derived = config.derive_config::<f32>().unwrap();

        assert_eq!(derived.input_sample_rate, 44_100);
        assert_eq!(derived.output_sample_rate, 96_000);
        assert_eq!(derived.input_chunk_frames, 2352);
        assert_eq!(derived.output_chunk_frames, 5120);
        assert_eq!(derived.input_fft_size, 4704);
        assert_eq!(derived.output_fft_size, 10240);
        assert_eq!(derived.input_offset, 1176);
        assert_eq!(derived.output_offset, 2560);
        assert_eq!(derived.spectral.geometry.stopband_end_bin, 2353);
        assert_eq!(derived.spectral.geometry.transition_bins(), 59);
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
            let taper = &derived.spectral.gain;
            assert_no_nans(taper, "config::taper_has_expected_rolloff_shape taper");
            let cutoff_bins = derived.spectral.geometry.stopband_end_bin;
            let taper_bins = derived.spectral.geometry.transition_bins().max(1);
            let transition_start = cutoff_bins.saturating_sub(taper_bins);

            assert!(taper.iter().all(|value| *value >= 0.0 && *value <= 1.0));
            assert_eq!(taper[transition_start - 1], 1.0);
            assert_eq!(taper[cutoff_bins], 0.0);
            assert!(taper[..transition_start].iter().all(|value| *value == 1.0));
            assert!(taper[cutoff_bins..].iter().all(|value| *value == 0.0));
            assert!(
                taper[transition_start..cutoff_bins]
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
            assert_no_nans(&derived.spectral.gain, "config::passthrough_taper_is_all_ones taper");

            assert_eq!(derived.spectral.gain.len(), derived.input_fft_size / 2 + 1);
            assert!(derived.spectral.gain.iter().all(|value| *value == 1.0));
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
    fn validates_alias_floor() {
        let base = Config::new(96_000, 44_100, 1);
        assert_eq!(base.alias_floor, AliasFloor::Fraction(1.0));
        assert!(base.validate().is_ok());
        assert!(base.clone().with_alias_floor(base.bandwidth).validate().is_ok());
        assert!(base.clone().with_alias_floor_db(-3.0).validate().is_ok());
        assert!(base.clone().with_alias_floor_db(0.0).validate().is_ok());

        for fraction in [base.bandwidth - 0.01, 1.01, f32::NAN] {
            assert!(matches!(
                base.clone().with_alias_floor(fraction).validate(),
                Err(Error::InvalidAliasFloor(_))
            ));
        }
        for db in [3.0, f32::NAN, f32::NEG_INFINITY] {
            assert!(matches!(
                base.clone().with_alias_floor_db(db).validate(),
                Err(Error::InvalidAliasFloorDb(_))
            ));
        }
    }

    #[test]
    fn alias_floor_db_is_resolved_against_final_bandwidth() {
        let db_first = Config::new(96_000, 44_100, 1)
            .with_alias_floor_db(-6.0)
            .with_bandwidth(0.95);
        let db_last = Config::new(96_000, 44_100, 1)
            .with_bandwidth(0.95)
            .with_alias_floor_db(-6.0);
        let strict = Config::new(96_000, 44_100, 1).with_bandwidth(0.95);

        let derived = db_first.derive_config::<f64>().unwrap();
        assert_eq!(derived, db_last.derive_config::<f64>().unwrap());

        let strict = strict.derive_config::<f64>().unwrap().spectral.geometry;
        assert_eq!(derived.spectral.geometry.passband_end_bin, strict.passband_end_bin);
        assert!(derived.spectral.geometry.stopband_end_bin > strict.stopband_end_bin);
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

    #[cfg(feature = "high_precision")]
    #[test]
    fn allows_high_quality_for_high_precision_derived_config() {
        let config = Config {
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            quality: 65_536,
            ..Config::default()
        };
        assert!(config.derive_config::<crate::DoubleDouble>().is_ok());
        assert!(config.derive_config::<crate::F128>().is_ok());
        assert!(config.derive_config::<crate::F256>().is_ok());
    }

    #[test]
    fn taper_is_all_ones_when_passthrough() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper: Vec<f32> = taper_type.build_taper(16, 8, 4, true, false);
            assert_no_nans(&taper, "config::taper_is_all_ones_when_passthrough taper");

            assert_eq!(taper.len(), 9);
            assert!(taper.iter().all(|v| *v == 1.0));
        }
    }

    #[test]
    fn taper_has_expected_passband_transition_and_stopband() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper: Vec<f32> = taper_type.build_taper(16, 6, 4, false, false);
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
            let taper: Vec<f32> = taper_type.build_taper(64, cutoff_bin, 16, false, false);
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
            let taper: Vec<f32> = taper_type.build_taper(16, 6, 0, false, false);
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
            let taper: Vec<f32> = taper_type.build_taper(16, 6, 1, false, false);
            assert_no_nans(&taper, "config::one_taper_bin_keeps_single_unity_transition_bin taper");

            assert_eq!(taper[5], 1.0);
            assert_eq!(taper[6], 0.0);
        }
    }

    #[test]
    fn taper_handles_cutoff_smaller_than_transition_width() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper: Vec<f32> = taper_type.build_taper(16, 2, 8, false, false);

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
