use num_traits::Float;

/// Low-latency preset tuned for realtime workloads.
///
/// You may prefer using a sinc resampler (eg. rubato) instead.
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
};

/// Balanced preset for good realtime quality.
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
    quality: 2048,
    bandwidth: 0.95,
    taper_type: TaperType::Cosine(3.4375),
};

/// High quality preset suitable for offline processing or realtime applications where quality is critical.
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
    quality: 65536,
    bandwidth: 0.97,
    taper_type: TaperType::Cosine(3.4375),
};

/// Maximum quality preset, optimized for offline processing. Not recommended for realtime applications.
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
    quality: 524288,
    bandwidth: 0.9932,
    taper_type: TaperType::Cosine(3.4375),
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
pub struct Config {
    /// Input audio sample rate in Hz.
    pub input_sample_rate: usize,

    /// Output audio sample rate in Hz.
    pub output_sample_rate: usize,

    /// Number of interleaved audio channels.
    pub channels: usize,

    // Quality roughly sets the spectral resolution scale (and therefore FFT bin count),
    // but this mapping is not exactly 1:1 (exact bin count depends on rate ratio and quantization).
    //
    // Default value is 2048.
    //
    // Value guide:
    //  - `512`:    Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    //  - `2048`:   Good quality, appropriate for realtime applications where high quality is desired. (Default)
    //  - `65536`:  High quality, good for offline resampling, also marginally appropriate for realtime applications where quality is critical.
    //  - `524288`: Extreme quality, good for offline resampling, very high quality but also very slow. Not recommended for realtime applications.
    pub quality: usize,

    /// Normalized filter bandwidth in the range `[0.0, 1.0]`.
    ///
    /// Higher values preserve more high-frequency content but shorten the transition band.
    ///
    /// Value guide:
    /// - `0.82`:  Fast and low quality, great for realtime applications. At this quality you may prefer using a sinc resampler (eg. rubato) instead.
    /// - `0.95`:  Balanced high-end retention for most cases.
    /// - `0.97`:  More aggressive high-end retention; Use with a higher "quality" setting.
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
}

impl Config {
    /// Builds a config with explicit sample rates/channel count and default quality settings.
    ///
    /// Returns a new `Config` value; semantic validation is deferred to `validate`.
    #[must_use] 
    pub fn new(input_sample_rate: usize, output_sample_rate: usize, channels: usize) -> Self {
        Self {
            input_sample_rate,
            output_sample_rate,
            channels,
            ..Self::default()
        }
    }

    /// Sets the input sample rate.
    ///
    /// Useful for completing a preset configuration before validation/stream creation.
    #[must_use] 
    pub fn with_input_rate(mut self, input_sample_rate: usize) -> Self {
        self.input_sample_rate = input_sample_rate;
        self
    }

    /// Sets the output sample rate.
    ///
    /// Useful for completing a preset configuration before validation/stream creation.
    #[must_use] 
    pub fn with_output_rate(mut self, output_sample_rate: usize) -> Self {
        self.output_sample_rate = output_sample_rate;
        self
    }

    /// Sets the number of interleaved channels.
    ///
    /// Useful for completing a preset configuration before validation/stream creation.
    #[must_use] 
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    /// Validates all user-facing configuration fields.
    ///
    /// Returns `Ok(())` when all values are in range, or a specific `Error` describing the first
    /// invalid field encountered.
    pub fn validate(&self) -> Result<(), Error> {
        if self.input_sample_rate == 0 || self.output_sample_rate == 0 {
            return Err(Error::InvalidSampleRate {
                input: self.input_sample_rate,
                output: self.output_sample_rate,
            });
        }

        // Special "you didnt configure your preset message"
        if self.input_sample_rate == 0 && self.output_sample_rate == 0 && self.channels == 0 {
            return Err(Error::PresetNotConfigured);
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

        if let TaperType::Cosine(alpha) = self.taper_type
            && (alpha <= 0.0 || !alpha.is_finite()) {
                return Err(Error::InvalidAlpha(alpha));
            }

        Ok(())
    }

    /// Computes derived FFT/chunk geometry from validated user configuration.
    ///
    /// Returns `DerivedConfig` for processing, or an error if validation fails.
    pub fn derive_config<T>(&self) -> Result<DerivedConfig<T>, Error>
    where
        T: Float,
    {
        self.validate()?;
        Ok(DerivedConfig::from_config(self))
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_sample_rate: 44_100,
            output_sample_rate: 44_100,
            channels: 2,
            quality: 2048,
            bandwidth: 0.95,
            taper_type: TaperType::default(),
        }
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
        }
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
        assert_eq!(derived.taper_bins, 103);
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
        assert_eq!(derived.taper_bins, 103);
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
            };
            let derived = config.derive_config::<f32>().unwrap();
            let taper = &derived.taper;
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
            };
            let derived = config.derive_config::<f32>().unwrap();

            assert_eq!(derived.taper.len(), derived.input_fft_size / 2 + 1);
            assert!(derived.taper.iter().all(|value| *value == 1.0));
        }
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(matches!(
            Config::new(0, 48_000, 2).validate(),
            Err(Error::InvalidSampleRate { .. })
        ));

        assert!(matches!(
            Config::new(44_100, 48_000, 0).validate(),
            Err(Error::InvalidChannels(0))
        ));

        let config = Config {
            bandwidth: f32::NAN,
            ..Config::default()
        };
        assert!(matches!(config.validate(), Err(Error::InvalidBandwidth(_))));

        let zero_alpha = Config {
            taper_type: TaperType::Cosine(0.0),
            ..Config::default()
        };
        assert!(matches!(
            zero_alpha.validate(),
            Err(Error::InvalidAlpha(0.0))
        ));

        let negative_alpha = Config {
            taper_type: TaperType::Cosine(-1.0),
            ..Config::default()
        };
        assert!(matches!(
            negative_alpha.validate(),
            Err(Error::InvalidAlpha(-1.0))
        ));

        let non_finite_alpha = Config {
            taper_type: TaperType::Cosine(f32::NAN),
            ..Config::default()
        };
        assert!(matches!(
            non_finite_alpha.validate(),
            Err(Error::InvalidAlpha(alpha)) if alpha.is_nan()
        ));
    }

    #[test]
    fn taper_is_all_ones_when_passthrough() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 8, 4, true, taper_type);

            assert_eq!(taper.len(), 9);
            assert!(taper.iter().all(|v| *v == 1.0));
        }
    }

    #[test]
    fn taper_has_expected_passband_transition_and_stopband() {
        for taper_type in [TaperType::Cosine(3.5), TaperType::Planck] {
            let taper = DerivedConfig::<f32>::build_taper(16, 6, 4, false, taper_type);

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
