use num_traits::Float;
use realfft::num_complex::Complex;

use crate::{AliasFloor, TaperType};

/// Resolves a user-facing [`AliasFloor`] into a fraction of the lower Nyquist frequency.
///
/// For [`AliasFloor::Decibels`], the floor `F` is placed exactly where the *final* (stretched)
/// transition reaches the requested level. With the transition spanning `[bw, 2 - F]` and the
/// level reached at normalized position `x`, `F = bw + x (2 - F - bw)`, which solves to
/// `F = (bw + x (2 - bw)) / (1 + x)`.
pub(crate) fn resolve_alias_floor(alias_floor: AliasFloor, bandwidth: f32, taper_type: &TaperType) -> f64 {
    let bw = f64::from(bandwidth);
    let floor = match alias_floor {
        AliasFloor::Fraction(fraction) => f64::from(fraction),
        AliasFloor::Decibels(db) => {
            let x = taper_type.transition_position_at_gain(10f64.powf(f64::from(db) / 20.0));
            (bw + x * (2.0 - bw)) / (1.0 + x)
        }
    };
    floor.clamp(bw.min(1.0), 1.0)
}

/// Bin-domain low-pass filter geometry.
///
/// Input and output FFTs share the same Hz-per-bin spacing (chunk sizes come from the reduced
/// rate ratio), so every boundary here is valid in both spectra.
///
/// ```text
/// 0 ---- passband ---- pass_end ---- floor ---- N ---- stop_end ---- U
///                          |<------- transition ------->|
///                                      |<-- folded -->|
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FilterGeometry {
    /// Nyquist bin of the lower-rate spectrum (`N`).
    pub(crate) lower_nyquist_bin: usize,
    /// Nyquist bin of the higher-rate spectrum (`U`).
    pub(crate) upper_nyquist_bin: usize,
    /// Bins below this are passed at unity gain. Never moves with the alias floor.
    pub(crate) passband_end_bin: usize,
    /// Lowest bin that may receive folded (downsampling) or imaged (upsampling) energy.
    /// Equals `lower_nyquist_bin` when aliasing is disabled.
    pub(crate) floor_bin: usize,
    /// One past the last bin with non-zero gain.
    pub(crate) stopband_end_bin: usize,
}

impl FilterGeometry {
    /// Derives the geometry from chunk sizes, `bandwidth`, and an alias floor expressed as a
    /// fraction of the lower Nyquist frequency (`1.0` disables aliasing).
    pub(crate) fn new(input_chunk_frames: usize, output_chunk_frames: usize, bandwidth: f32, alias_floor: f64) -> Self {
        let lower_nyquist_bin = input_chunk_frames.min(output_chunk_frames);
        let upper_nyquist_bin = input_chunk_frames.max(output_chunk_frames);

        let taper_bins = ((lower_nyquist_bin + 1) as f64 * (1.0 - f64::from(bandwidth))).ceil() as usize;
        let passband_end_bin = (lower_nyquist_bin + 1).saturating_sub(taper_bins);

        // The floor can never reach into the passband, and never includes DC.
        let min_floor_bin = passband_end_bin.max(1).min(lower_nyquist_bin);
        let floor_bin =
            ((alias_floor * lower_nyquist_bin as f64).round() as usize).clamp(min_floor_bin, lower_nyquist_bin);

        // Mirror the floor about the lower Nyquist, bounded by the larger spectrum.
        let stopband_end_bin = (2 * lower_nyquist_bin + 1 - floor_bin).min(upper_nyquist_bin + 1);

        Self {
            lower_nyquist_bin,
            upper_nyquist_bin,
            passband_end_bin,
            floor_bin,
            stopband_end_bin,
        }
    }

    /// Width of the transition in bins.
    #[inline]
    pub(crate) fn transition_bins(&self) -> usize {
        self.stopband_end_bin - self.passband_end_bin
    }

    /// First lower-spectrum bin `k` whose mirror `2N - k` has non-zero gain. Bins in
    /// `reflect_start_bin()..lower_nyquist_bin` take part in folding/imaging; the range is empty
    /// when aliasing is disabled.
    #[inline]
    pub(crate) fn reflect_start_bin(&self) -> usize {
        let n = self.lower_nyquist_bin;
        self.floor_bin
            .max((2 * n + 1).saturating_sub(self.stopband_end_bin))
            .min(n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Equal chunk sizes (phase-only processing).
    Equal,
    /// Input spectrum is larger than output spectrum.
    Down,
    /// Input spectrum is smaller than output spectrum.
    Up,
}

/// Precomputed mapping from the forward (input) spectrum to the inverse (output) spectrum.
///
/// Holds everything the hot path needs: gain per absolute bin, phase rotation per output
/// baseband bin, and the reflection range. Phase is always applied in the destination domain so
/// direct and folded/imaged energy share the same output-frequency phase response.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpectralPlan<T> {
    direction: Direction,
    pub(crate) geometry: FilterGeometry,
    reflect_start_bin: usize,
    /// Gain indexed by absolute bin frequency; `upper_nyquist_bin + 1` entries.
    pub(crate) gain: Vec<T>,
    /// Unit phase rotation per lower-spectrum bin; `lower_nyquist_bin + 1` entries.
    pub(crate) phase: Vec<Complex<T>>,
    pub(crate) phase_enabled: bool,
}

impl<T> SpectralPlan<T>
where
    T: Float,
{
    pub(crate) fn new(
        input_chunk_frames: usize,
        output_chunk_frames: usize,
        bandwidth: f32,
        taper_type: &TaperType,
        phase: T,
        phase_intensity: T,
        alias_floor: f64,
    ) -> Self {
        let geometry = FilterGeometry::new(input_chunk_frames, output_chunk_frames, bandwidth, alias_floor);
        let direction = match input_chunk_frames.cmp(&output_chunk_frames) {
            std::cmp::Ordering::Equal => Direction::Equal,
            std::cmp::Ordering::Greater => Direction::Down,
            std::cmp::Ordering::Less => Direction::Up,
        };

        let gain = taper_type.build_taper(
            geometry.upper_nyquist_bin * 2,
            geometry.stopband_end_bin,
            geometry.transition_bins(),
            direction == Direction::Equal,
        );

        Self {
            direction,
            geometry,
            reflect_start_bin: geometry.reflect_start_bin(),
            gain,
            phase: build_phase(geometry.lower_nyquist_bin + 1, phase, phase_intensity),
            phase_enabled: !phase.is_zero() && !phase_intensity.is_zero(),
        }
    }

    /// Maps the forward spectrum `src` into the inverse spectrum `dst`, filling every `dst` bin.
    ///
    /// DC and output Nyquist bins are left real-valued, as required by `realfft`.
    pub(crate) fn apply(&self, src: &[Complex<T>], dst: &mut [Complex<T>]) {
        let n = self.geometry.lower_nyquist_bin;
        let r0 = self.reflect_start_bin;
        let gain = &self.gain;
        let phase = self.phase_enabled.then_some(self.phase.as_slice());

        match self.direction {
            Direction::Up => {
                // Baseband, including the input Nyquist bin: it already carries both ±N halves,
                // so it is not imaged separately.
                map_direct(&mut dst[..=n], &src[..=n], &gain[..=n], phase);
                dst[n + 1..].fill(Complex::new(T::zero(), T::zero()));

                // First spectral image above the input Nyquist: output bin 2N - k mirrors k.
                for k in r0..n {
                    let j = 2 * n - k;
                    dst[j] = rotate(src[k], phase, k).conj() * gain[j];
                }
            }
            Direction::Down | Direction::Equal => {
                map_direct(&mut dst[..r0], &src[..r0], &gain[..r0], phase);

                // Fold input bin 2N - k (above the output Nyquist) onto output bin k. A positive
                // frequency above Nyquist aliases to a negative one, hence the conjugate.
                for k in r0..n {
                    let j = 2 * n - k;
                    dst[k] = rotate(src[k] * gain[k] + src[j].conj() * gain[j], phase, k);
                }

                // The output Nyquist bin holds both +N and -N. When downsampling, input bin N is
                // only the +N half, so it folds onto itself (x2). With equal spectra it already
                // holds both halves.
                let fold = if self.direction == Direction::Down {
                    T::one() + T::one()
                } else {
                    T::one()
                };
                dst[n] = Complex::new(rotate(src[n], phase, n).re * gain[n] * fold, T::zero());
            }
        }

        if let Some(dc_bin) = dst.get_mut(0) {
            dc_bin.im = T::zero();
        }
        if dst.len() > 1 {
            let nyquist_bin = dst.len() - 1;
            dst[nyquist_bin].im = T::zero();
        }
    }
}

/// Copies `src` into `dst` with per-bin gain and optional destination-domain phase rotation.
#[inline]
fn map_direct<T: Float>(dst: &mut [Complex<T>], src: &[Complex<T>], gain: &[T], phase: Option<&[Complex<T>]>) {
    match phase {
        Some(phase) => {
            for (((dst, src), gain), phase) in dst.iter_mut().zip(src).zip(gain).zip(phase) {
                *dst = *src * *gain * *phase;
            }
        }
        None => {
            for ((dst, src), gain) in dst.iter_mut().zip(src).zip(gain) {
                *dst = *src * *gain;
            }
        }
    }
}

#[inline(always)]
fn rotate<T: Float>(value: Complex<T>, phase: Option<&[Complex<T>]>, bin: usize) -> Complex<T> {
    match phase {
        Some(phase) => value * phase[bin],
        None => value,
    }
}

/// Builds the per-bin unit complex phase rotation.
pub(crate) fn build_phase<T: Float>(bins: usize, phase: T, phase_intensity: T) -> Vec<Complex<T>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    #[test]
    fn phase_table_uses_c_rotation_formula() {
        let phase_intensity = Config::DEFAULT.phase_intensity;
        let identity = build_phase::<f32>(4, 0.0, phase_intensity);
        assert_eq!(identity.len(), 4);
        assert!(identity.iter().all(|phase| phase.re == 1.0 && phase.im == 0.0));

        let zero_intensity = build_phase::<f32>(4, 0.5, 0.0);
        assert!(zero_intensity.iter().all(|phase| phase.re == 1.0 && phase.im == 0.0));

        let positive = build_phase::<f32>(4, 0.5, phase_intensity);
        let negative = build_phase::<f32>(4, -0.5, phase_intensity);

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
    fn default_geometry_matches_strict_cutoff() {
        // 48k -> 44.1k at default quality: N = 2058, U = 2240.
        let geometry = FilterGeometry::new(2240, 2058, 0.9114534, 1.0);
        assert_eq!(geometry.lower_nyquist_bin, 2058);
        assert_eq!(geometry.upper_nyquist_bin, 2240);
        assert_eq!(geometry.stopband_end_bin, 2059);
        assert_eq!(geometry.transition_bins(), 183);
        assert_eq!(geometry.floor_bin, 2058);
        assert_eq!(geometry.reflect_start_bin(), 2058);
    }

    const BW: f32 = 0.9114534;

    fn tapers() -> Vec<TaperType> {
        vec![
            TaperType::Cosine(3.4375),
            TaperType::Planck,
            TaperType::BetaCdf {
                alpha: 10.0,
                beta: 10.0,
            },
            #[cfg(feature = "bessel")]
            TaperType::Bessel(6.0),
        ]
    }

    #[test]
    fn lowering_floor_widens_stopband_without_moving_passband() {
        // 96k -> 44.1k, 48k -> 44.1k (clamps at the upper spectrum), and 44.1k -> 96k.
        for (input, output) in [(4480, 2058), (2240, 2058), (2058, 4480)] {
            let strict = FilterGeometry::new(input, output, BW, 1.0);
            let mut previous_stop = strict.stopband_end_bin;
            for step in 0..=20 {
                let floor = 1.0 - (1.0 - f64::from(BW)) * f64::from(step) / 20.0;
                let geometry = FilterGeometry::new(input, output, BW, floor);
                assert_eq!(geometry.passband_end_bin, strict.passband_end_bin);
                assert!(geometry.stopband_end_bin >= previous_stop);
                assert!(geometry.stopband_end_bin <= geometry.upper_nyquist_bin + 1);
                assert!(geometry.floor_bin >= geometry.passband_end_bin.max(1));
                assert!(geometry.reflect_start_bin() >= geometry.floor_bin);
                assert!(geometry.reflect_start_bin() <= geometry.lower_nyquist_bin);
                previous_stop = geometry.stopband_end_bin;
            }
        }
    }

    #[test]
    fn floor_below_bandwidth_clamps_to_passband_edge() {
        let geometry = FilterGeometry::new(4480, 2058, BW, 0.0);
        assert_eq!(geometry.floor_bin, geometry.passband_end_bin);
    }

    #[test]
    fn widest_floor_mirrors_transition_about_nyquist() {
        let geometry = FilterGeometry::new(4480, 2058, BW, f64::from(BW));
        let n = geometry.lower_nyquist_bin;
        // `round(bw * N)` and `N + 1 - ceil((N + 1)(1 - bw))` may differ by a bin.
        assert!(geometry.floor_bin.abs_diff(geometry.passband_end_bin) <= 2);
        assert_eq!(geometry.stopband_end_bin, 2 * n + 1 - geometry.floor_bin);
        assert_eq!(geometry.reflect_start_bin(), geometry.floor_bin);
    }

    #[test]
    fn stopband_clamps_to_upper_spectrum() {
        let geometry = FilterGeometry::new(2240, 2058, BW, 0.8);
        assert_eq!(geometry.stopband_end_bin, 2241);
        // Mirrors of bins below this would lie beyond the input spectrum.
        assert_eq!(geometry.reflect_start_bin(), 2 * 2058 + 1 - 2241);
    }

    #[test]
    fn transition_position_is_monotone_in_gain() {
        for taper in tapers() {
            assert_eq!(taper.transition_position_at_gain(1.0), 0.0);
            assert_eq!(taper.transition_position_at_gain(0.0), 1.0);
            let mut previous = 0.0;
            for db in [-0.1, -1.0, -3.0, -6.0, -20.0, -60.0] {
                let position = taper.transition_position_at_gain(10f64.powf(db / 20.0));
                assert!(
                    position >= previous && position <= 1.0,
                    "{taper:?}: {db} dB -> {position}"
                );
                previous = position;
            }
        }
    }

    #[test]
    fn fraction_floor_passes_through() {
        let floor = resolve_alias_floor(AliasFloor::Fraction(0.95), BW, &TaperType::Planck);
        assert_eq!(floor, f64::from(0.95f32));
        assert_eq!(
            resolve_alias_floor(AliasFloor::Fraction(1.0), BW, &TaperType::Planck),
            1.0
        );
    }

    #[test]
    fn decibel_floor_lands_at_requested_level() {
        for taper in tapers() {
            for bandwidth in [0.8f32, BW, 0.97] {
                let mut previous_floor = f64::from(bandwidth);
                for db in [-3.0f32, -6.0, -20.0, -40.0] {
                    let floor = resolve_alias_floor(AliasFloor::Decibels(db), bandwidth, &taper);
                    assert!(
                        floor >= previous_floor && floor <= 1.0,
                        "{taper:?} bw={bandwidth} {db} dB"
                    );
                    previous_floor = floor;

                    let plan = SpectralPlan::<f64>::new(4480, 2058, bandwidth, &taper, 0.0, 0.0, floor);
                    let target = 10f64.powf(f64::from(db) / 20.0);
                    // The level is located on a high-resolution sampling of the shape; short real
                    // transitions (and the Bessel taper's size-dependent normalization) shift it
                    // by a few bins.
                    let bin = plan.geometry.floor_bin;
                    assert!(
                        plan.gain[bin - 3] >= target && plan.gain[bin + 3] <= target,
                        "{taper:?} bw={bandwidth} {db} dB: gain around floor bin {bin} = {:?}, target {target}",
                        &plan.gain[bin - 3..=bin + 3]
                    );
                }
            }
        }
    }
}

/// End-to-end tone tests through the core resampler.
#[cfg(test)]
mod tone_tests {
    use super::*;
    use crate::Config;
    use crate::cpu_core::CpuCore;

    const AMP: f64 = 0.5;
    const BW: f32 = 0.9114534;
    /// Strict-mode suppression expected for content outside the passband/transition.
    const SUPPRESSED_DB: f64 = -100.0;

    fn tone(rate: usize, hz: f64, phase: f64) -> Vec<f64> {
        (0..rate)
            .map(|i| AMP * (2.0 * std::f64::consts::PI * hz * i as f64 / rate as f64 + phase).cos())
            .collect()
    }

    fn resample(config: Config, input: &[f64]) -> Vec<f64> {
        let mut core = CpuCore::<f64>::new(config.derive_config::<f64>().unwrap());
        let output = core.process_all(input).unwrap();
        assert!(output.iter().all(|sample| sample.is_finite()), "non-finite output");
        output
    }

    fn config(input_rate: usize, output_rate: usize, alias_floor: AliasFloor) -> Config {
        Config {
            alias_floor,
            ..Config::new(input_rate, output_rate, 1)
        }
    }

    /// Level of the `hz` component in dB relative to `AMP`, Blackman-windowed over the
    /// steady-state middle half of `signal`.
    fn level_db(signal: &[f64], rate: usize, hz: f64) -> f64 {
        use std::f64::consts::PI;

        let body = &signal[signal.len() / 4..signal.len() * 3 / 4];
        let last = (body.len() - 1) as f64;
        let (mut re, mut im, mut window_sum) = (0.0, 0.0, 0.0);
        for (i, sample) in body.iter().enumerate() {
            let x = i as f64 / last;
            let window = 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos();
            let angle = 2.0 * PI * hz * i as f64 / rate as f64;
            re += sample * window * angle.cos();
            im -= sample * window * angle.sin();
            window_sum += window;
        }
        let magnitude = 2.0 * (re * re + im * im).sqrt() / window_sum;
        20.0 * (magnitude / AMP).max(1e-15).log10()
    }

    #[test]
    fn downsampling_folds_at_mirrored_gain_only_when_floor_is_lowered() {
        for alias_floor in [AliasFloor::Decibels(-3.0), AliasFloor::Fraction(BW)] {
            let aliased = config(96_000, 44_100, alias_floor);
            let spectral = aliased.derive_config::<f64>().unwrap().spectral;
            let n = spectral.geometry.lower_nyquist_bin;
            let bin_hz = 22_050.0 / n as f64;

            // Source bin a quarter of the way from Nyquist to the extended stopband edge, which
            // folds onto output bin 2N - j with gain[j].
            let j = n + (spectral.geometry.stopband_end_bin - n) / 4;
            let alias_hz = (2 * n - j) as f64 * bin_hz;
            let input = tone(96_000, j as f64 * bin_hz, 0.3);

            let expected = 20.0 * spectral.gain[j].log10();
            let level = level_db(&resample(aliased, &input), 44_100, alias_hz);
            assert!(
                (level - expected).abs() < 0.5,
                "{alias_floor:?}: alias level {level} dB, expected {expected} dB"
            );

            let strict = level_db(
                &resample(config(96_000, 44_100, AliasFloor::Fraction(1.0)), &input),
                44_100,
                alias_hz,
            );
            assert!(strict < SUPPRESSED_DB, "strict alias level {strict} dB");
        }
    }

    #[test]
    fn content_above_extended_stopband_stays_suppressed() {
        // Widest floor at 96k -> 44.1k stops at ~24.0 kHz.
        for hz in [24_500.0, 26_000.0, 30_000.0] {
            let output = resample(config(96_000, 44_100, AliasFloor::Fraction(BW)), &tone(96_000, hz, 0.3));
            let alias = level_db(&output, 44_100, 44_100.0 - hz);
            assert!(alias < SUPPRESSED_DB, "{hz} Hz alias level {alias} dB");
        }
    }

    #[test]
    fn folding_never_reaches_below_floor() {
        let config = config(96_000, 44_100, AliasFloor::Decibels(-3.0));
        let geometry = config.derive_config::<f64>().unwrap().spectral.geometry;
        let bin_hz = 22_050.0 / geometry.lower_nyquist_bin as f64;
        let floor_hz = geometry.floor_bin as f64 * bin_hz;

        // Tones whose alias would land 50..1000 Hz below the floor.
        for offset in [50.0, 200.0, 500.0, 1000.0] {
            let alias_hz = floor_hz - offset;
            let output = resample(config.clone(), &tone(96_000, 44_100.0 - alias_hz, 0.3));
            let level = level_db(&output, 44_100, alias_hz);
            assert!(
                level < SUPPRESSED_DB,
                "alias {alias_hz} Hz (floor {floor_hz} Hz) level {level} dB"
            );
        }
    }

    #[test]
    fn passband_is_unchanged_by_alias_floor() {
        for hz in [1_000.0, 15_000.0, 19_500.0] {
            let input = tone(96_000, hz, 0.3);
            let strict = level_db(
                &resample(config(96_000, 44_100, AliasFloor::Fraction(1.0)), &input),
                44_100,
                hz,
            );
            let widest = level_db(
                &resample(config(96_000, 44_100, AliasFloor::Fraction(BW)), &input),
                44_100,
                hz,
            );
            assert!(strict.abs() < 0.01, "{hz} Hz strict level {strict} dB");
            assert!(
                (strict - widest).abs() < 0.001,
                "{hz} Hz: strict {strict} dB, widest {widest} dB"
            );
        }
    }

    #[test]
    fn upsampling_images_only_when_floor_is_lowered() {
        for phase in [0.0f32, -0.5] {
            let make = |alias_floor| Config {
                phase,
                ..config(44_100, 96_000, alias_floor)
            };
            let input = tone(44_100, 21_800.0, 0.3);
            let strict = level_db(&resample(make(AliasFloor::Fraction(1.0)), &input), 96_000, 22_300.0);
            let widest = level_db(&resample(make(AliasFloor::Fraction(BW)), &input), 96_000, 22_300.0);

            assert!(strict < SUPPRESSED_DB, "phase {phase}: strict image level {strict} dB");
            assert!(widest > -20.0, "phase {phase}: widest image level {widest} dB");
        }
    }

    #[test]
    fn downsampled_nyquist_tone_folds_onto_itself() {
        // A tone exactly at the output Nyquist folds onto itself as g * A * cos(phase) * (-1)^n,
        // where g is the gain at Nyquist. The probe reads a Nyquist component at twice its
        // amplitude.
        let phase: f64 = 0.6;
        let input = tone(96_000, 22_050.0, phase);
        for alias_floor in [AliasFloor::Fraction(1.0), AliasFloor::Fraction(BW)] {
            let config = config(96_000, 44_100, alias_floor);
            let spectral = config.derive_config::<f64>().unwrap().spectral;
            let gain = spectral.gain[spectral.geometry.lower_nyquist_bin];

            let expected = 20.0 * (2.0 * gain * phase.cos()).max(1e-15).log10();
            let level = level_db(&resample(config, &input), 44_100, 22_050.0);
            if expected < SUPPRESSED_DB {
                assert!(level < SUPPRESSED_DB, "{alias_floor:?}: Nyquist level {level} dB");
            } else {
                assert!(
                    (level - expected).abs() < 0.5,
                    "{alias_floor:?}: Nyquist level {level} dB, expected {expected} dB"
                );
            }
        }
    }

    #[test]
    fn alias_level_is_phase_independent_across_tapers() {
        let mut tapers = vec![
            TaperType::Cosine(3.4375),
            TaperType::Planck,
            TaperType::BetaCdf {
                alpha: 10.0,
                beta: 10.0,
            },
        ];
        #[cfg(feature = "bessel")]
        tapers.push(TaperType::Bessel(6.0));

        let input = tone(96_000, 22_300.0, 0.3);
        for taper_type in tapers {
            let levels: Vec<f64> = [-1.0f32, -0.5, 0.0, 0.5, 1.0]
                .into_iter()
                .map(|phase| {
                    let config = Config {
                        phase,
                        taper_type,
                        ..config(96_000, 44_100, AliasFloor::Fraction(BW))
                    };
                    level_db(&resample(config, &input), 44_100, 21_800.0)
                })
                .collect();
            let (min, max) = levels.iter().fold((f64::MAX, f64::MIN), |(min, max), level| {
                (min.min(*level), max.max(*level))
            });
            assert!(min > -40.0, "{taper_type:?}: alias levels {levels:?}");
            assert!(
                max - min < 0.5,
                "{taper_type:?}: alias levels vary with phase {levels:?}"
            );
        }
    }
}
