use crate::Error;
use crate::beta_reg::beta_reg;
use crate::dsp_float::DspFloat;
use crate::kaiser::{KaiserSeries, MAX_KAISER_BETA};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Transition profile used to shape the cutoff edge of the frequency mask.
pub enum TaperType {
    /// Uses a Planck-taper transition
    Planck,

    /// Cumulative distribution of a Kaiser window.
    ///
    /// `beta` suggested default: `30.0`.
    Kaiser(f32),

    /// Endpoint-normalized hyperbolic tangent transition.
    ///
    /// `alpha` suggested default: `3.0`.
    Tanh(f32),

    /// Uses a sigmoid-warped cosine transition.
    ///
    /// `alpha` controls the sharpness of the transition. Suggested default: `3.5`.
    ///
    /// Value guide for `Cosine(alpha)`:
    /// - `1.5`: Very smooth transition; may increase near-Nyquist artifacts.
    /// - `2.5`: Smooth and less aggressive shaping.
    /// - `3.5`: Good balance between smoothness and selectivity.
    /// - `4.0`: Sharper shaping; trades smoothness for selectivity.
    Cosine(f32),

    /// Beta-CDF taper.
    ///
    /// `alpha` and `beta` are the two Beta distribution shape parameters.
    /// Symmetric suggested default:
    ///     BetaCdf { alpha: 10.0, beta: 10.0 }
    ///
    /// Asymmetric:
    ///     BetaCdf { alpha: 8.0, beta: 10.0 }
    ///     BetaCdf { alpha: 10.0, beta: 8.0 }
    BetaCdf { alpha: f32, beta: f32 },
}

impl Default for TaperType {
    fn default() -> Self {
        Self::Kaiser(30.0)
    }
}

impl TaperType {
    /// Builds a full frequency mask of `spectrum_fft_size / 2 + 1` bins: unity passband, a
    /// descending transition over the `taper_bins` bins before `cutoff_bin`, and zero stopband.
    ///
    /// With `straddle_nyquist` (the transition extends past the lower Nyquist and is mirrored
    /// about it), the transition is laid out exactly as sampled, so a symmetric profile stays
    /// centred on its span and mirrored bin pairs stay complementary. Otherwise leading unity
    /// and trailing zero samples are trimmed and the rest ends just before `cutoff_bin`, so
    /// every transition bin carries a non-zero gain. The Kaiser taper samples fixed bin positions
    /// and is never trimmed.
    ///
    /// Profiles are computed and trimmed in `f64` and converted last, so the layout is the same
    /// for every `T`.
    pub(crate) fn build_taper<T: DspFloat>(
        &self,
        spectrum_fft_size: usize,
        cutoff_bin: usize,
        taper_bins: usize,
        is_passthrough: bool,
        straddle_nyquist: bool,
    ) -> Vec<T> {
        let len = spectrum_fft_size / 2 + 1;
        if is_passthrough {
            return vec![T::one(); len];
        }
        let raw = self.build_transition(taper_bins, straddle_nyquist);
        let transition = if straddle_nyquist || matches!(self, TaperType::Kaiser(_)) {
            &raw[..]
        } else {
            trim_transition(&raw)
        };
        let transition: Vec<T> = transition.iter().map(|&value| T::cast_f64(value)).collect();
        place_transition(len, cutoff_bin, &transition)
    }

    /// Builds the normalized descending transition shape sampled over `taper_bins` bins,
    /// untrimmed. `straddle_nyquist` only changes the sampling of the Kaiser taper.
    pub(crate) fn build_transition(&self, taper_bins: usize, straddle_nyquist: bool) -> Vec<f64> {
        match self {
            TaperType::Planck => planck_transition(taper_bins),
            TaperType::Kaiser(beta) => kaiser_transition(taper_bins, *beta, straddle_nyquist),
            TaperType::Tanh(alpha) => tanh_transition(taper_bins, *alpha),
            TaperType::Cosine(alpha) => cosine_transition(taper_bins, *alpha),
            TaperType::BetaCdf { alpha, beta } => beta_cdf_transition(taper_bins, *alpha, *beta),
        }
    }

    /// Returns the normalized position in `[0.0, 1.0]` across the transition (`0.0` = passband
    /// edge, `1.0` = stopband edge) where the descending gain first falls to `gain`.
    ///
    /// Used to place lowered alias floors, so the shape is sampled at high resolution with the
    /// same builder and the straddling layout used for those masks. The result is independent
    /// of FFT size.
    pub(crate) fn transition_position_at_gain(&self, gain: f64) -> f64 {
        const RESOLUTION: usize = 65_536;

        if gain >= 1.0 {
            return 0.0;
        }
        if gain <= 0.0 {
            return 1.0;
        }

        let transition = self.build_transition(RESOLUTION, true);

        // Monotone non-increasing: find the first sample at or below `gain` and interpolate.
        let idx = transition.partition_point(|value| *value > gain);
        let position = if idx == 0 {
            0.0
        } else if idx == transition.len() {
            transition.len() as f64
        } else {
            let (above, below) = (transition[idx - 1], transition[idx]);
            (idx - 1) as f64 + (above - gain) / (above - below)
        };

        (position / RESOLUTION as f64).clamp(0.0, 1.0)
    }

    /// Validates taper parameters and returns an error for invalid values.
    pub fn validate(&self) -> Result<(), Error> {
        let check = |value: f32, max: f32, error: fn(f32) -> Error| {
            if value > 0.0 && value.is_finite() && value <= max {
                Ok(())
            } else {
                Err(error(value))
            }
        };
        match self {
            TaperType::Planck => Ok(()),
            TaperType::Kaiser(beta) => check(*beta, MAX_KAISER_BETA, Error::InvalidKaiserBeta),
            TaperType::Tanh(alpha) | TaperType::Cosine(alpha) => check(*alpha, f32::MAX, Error::InvalidAlpha),
            TaperType::BetaCdf { alpha, beta } => {
                check(*alpha, f32::MAX, Error::InvalidAlpha).and(check(*beta, f32::MAX, Error::InvalidBeta))
            }
        }
    }
}

/// Lays out a mask of `len` bins: unity below the transition, `transition` ending just before
/// `cutoff_bin`, and zeros from `cutoff_bin` on.
fn place_transition<T: DspFloat>(len: usize, cutoff_bin: usize, transition: &[T]) -> Vec<T> {
    let taper_start = cutoff_bin.saturating_sub(transition.len());

    (0..len)
        .map(|idx| {
            if idx < taper_start {
                T::one()
            } else if idx < cutoff_bin {
                transition[idx - taper_start]
            } else {
                T::zero()
            }
        })
        .collect()
}

/// Trims leading unity and trailing zero samples from a raw descending transition.
fn trim_transition(raw: &[f64]) -> &[f64] {
    let trim_start = raw.iter().position(|value| *value < 1.0).unwrap_or(raw.len());
    let trim_stop = raw
        .iter()
        .rposition(|value| *value > 0.0)
        .map_or(0, |idx| raw.len() - idx - 1);
    let active_end = raw.len().saturating_sub(trim_stop);

    &raw[trim_start..active_end]
}

/// Kaiser-window CDF from the exact integral of the window, sampled at the `taper_bins` bin
/// centres, or at `taper_bins` points running from unity to zero inclusive when the transition
/// straddles Nyquist.
fn kaiser_transition(taper_bins: usize, beta: f32, straddle_nyquist: bool) -> Vec<f64> {
    let npts = taper_bins;
    let m = if straddle_nyquist { npts.saturating_sub(1) } else { npts };
    let mut taper = vec![0.0; npts];
    if m == 0 {
        return taper;
    }

    let series = KaiserSeries::new(f64::from(beta));
    let half_area = series.half_integral(1.0);

    // Evaluate the passband half and mirror it, so the profile is exactly complementary.
    let half = npts / 2;
    for i in 0..half {
        let x = if straddle_nyquist {
            i as f64 / m as f64
        } else {
            (i as f64 + 0.5) / m as f64
        };
        let u = 1.0 - 2.0 * x;
        let lo = (0.5 * (1.0 - series.half_integral(u) / half_area)).clamp(0.0, 0.5);
        taper[i] = 1.0 - lo;
        taper[npts - 1 - i] = lo;
    }
    if npts % 2 == 1 {
        taper[half] = 0.5;
    }

    // Monotonicity repair for rounding noise.
    for i in 1..npts {
        if taper[i] > taper[i - 1] {
            taper[i] = taper[i - 1];
        }
    }
    taper
}

/// (1 - tanh(alpha * (2*x-1)) / tanh(alpha)) / 2.
fn tanh_transition(n: usize, alpha: f32) -> Vec<f64> {
    let alpha = f64::from(alpha);
    sampled_transition(n, |x| 0.5 * (1.0 - (alpha * (2.0 * x - 1.0)).tanh() / alpha.tanh()))
}

/// Sample an endpoint-normalized profile; a single transition bin uses its midpoint.
fn sampled_transition(n: usize, profile: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if n == 1 {
                profile(0.5)
            } else if i == 0 {
                1.0
            } else if i == n - 1 {
                0.0
            } else {
                profile(i as f64 / (n - 1) as f64).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Builds a Planck-taper transition.
fn planck_transition(taper_bins: usize) -> Vec<f64> {
    if taper_bins == 0 {
        return Vec::new();
    }
    if taper_bins == 1 {
        return vec![1.0];
    }

    let denom = taper_bins as f64 - 1.0;

    (0..taper_bins)
        .map(|idx| {
            if idx == 0 {
                return 1.0;
            }

            if idx == taper_bins - 1 {
                return 0.0;
            }

            let x = idx as f64 / denom;

            // Descending Planck taper
            let z = 1.0 / x - 1.0 / (1.0 - x);
            let rising = 1.0 / (z.exp() + 1.0);

            let value = 1.0 - rising;

            if value.is_normal() {
                value
            } else if value >= 1.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Builds a sigmoid-warped cosine transition.
fn cosine_transition(taper_bins: usize, alpha: f32) -> Vec<f64> {
    if taper_bins == 0 {
        return Vec::new();
    }
    if taper_bins == 1 {
        return vec![1.0];
    }

    let pi = std::f64::consts::PI;
    let alpha = f64::from(alpha);
    let denom = taper_bins as f64 - 1.0;

    (0..taper_bins)
        .map(|idx| {
            let x = idx as f64 / denom;

            // Powered sigmoid warp:
            //
            //     x_warped = x^a / (x^a + (1 - x)^a)
            //
            // This preserves endpoints but concentrates most of the transition
            // around the middle, making the cosine behave more like the
            // trimmed logistic taper.
            let a = x.powf(alpha);
            let b = (1.0 - x).powf(alpha);
            let warped = a / (a + b);

            let value = (1.0 + (pi * warped).cos()) / 2.0;

            if value.is_normal() {
                value
            } else if value == 1.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Builds a Beta-CDF transition from the regularized lower incomplete beta function.
fn beta_cdf_transition(taper_bins: usize, alpha: f32, beta: f32) -> Vec<f64> {
    if taper_bins == 0 {
        return Vec::new();
    }
    if taper_bins == 1 {
        return vec![1.0];
    }

    let denom = taper_bins as f64 - 1.0;

    (0..taper_bins)
        .map(|idx| {
            if idx == 0 {
                return 1.0;
            }

            if idx == taper_bins - 1 {
                return 0.0;
            }

            let x = (idx as f64 / denom).clamp(0.0, 1.0);
            let value = 1.0 - beta_reg(alpha as f64, beta as f64, x);

            if value.is_normal() {
                value
            } else if value >= 1.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaiser_taper_is_descending_and_bounded() {
        let taper = TaperType::Kaiser(18.0).build_taper::<f64>(64, 24, 16, false, false);
        let transition_start = taper
            .iter()
            .position(|value| *value < 1.0)
            .expect("expected transition start");
        let transition = &taper[transition_start..24];

        assert_eq!(taper.len(), 33);
        assert!(!transition.is_empty());
        assert!(taper[..transition_start].iter().all(|value| *value == 1.0));
        assert!(taper[24..].iter().all(|value| *value == 0.0));

        for value in transition {
            assert!(*value >= 0.0);
            assert!(*value <= 1.0);
        }
        for pair in transition.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }

    #[test]
    fn kaiser_passthrough_is_all_ones() {
        let taper = TaperType::Kaiser(18.0).build_taper::<f32>(16, 8, 4, true, false);

        assert_eq!(taper.len(), 9);
        assert!(taper.iter().all(|value| *value == 1.0));
    }
}

#[cfg(test)]
mod new_taper_tests {
    use super::*;

    #[test]
    fn new_profiles_are_finite_monotone_and_validated() {
        let kaiser_betas = [f32::MIN_POSITIVE, 0.1, 3.0, 18.0, MAX_KAISER_BETA];
        let tapers = [f32::MIN_POSITIVE, 0.1, 3.0, 6.0, 200.0, f32::MAX]
            .map(TaperType::Tanh)
            .into_iter()
            .chain(kaiser_betas.map(TaperType::Kaiser));
        for taper in tapers {
            assert!(taper.validate().is_ok(), "{taper:?}");
            for n in [0, 1, 2, 3, 16, 63] {
                for straddle in [false, true] {
                    let mask = taper.build_taper::<f64>(256, 100, n, false, straddle);
                    assert!(
                        mask.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                        "{taper:?}, n={n}, straddle={straddle}"
                    );
                    assert!(
                        mask.windows(2).all(|v| v[0] >= v[1]),
                        "{taper:?}, n={n}, straddle={straddle}"
                    );
                    assert_eq!(mask[0], 1.0);
                    assert_eq!(mask[100], 0.0);
                    assert!(
                        taper
                            .build_taper::<f32>(256, 100, n, true, straddle)
                            .iter()
                            .all(|v| *v == 1.0)
                    );
                }
            }
        }
        for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(TaperType::Tanh(alpha).validate(), Err(Error::InvalidAlpha(_))));
        }
        for beta in [0.0, -1.0, f32::NAN, f32::INFINITY, MAX_KAISER_BETA * 1.01] {
            assert!(
                matches!(TaperType::Kaiser(beta).validate(), Err(Error::InvalidKaiserBeta(_))),
                "{beta}"
            );
        }
    }

    #[test]
    fn tanh_has_midpoint_and_complementary_symmetry() {
        let shape = tanh_transition(9, 3.0);
        assert_eq!(shape[shape.len() / 2], 0.5);
        for (a, b) in shape.iter().zip(shape.iter().rev()) {
            assert!((a + b - 1.0).abs() < 1e-14);
        }
    }

    /// Modified Bessel function I0 by its power series.
    fn i0_series(x: f64) -> f64 {
        let (mut sum, mut term) = (1.0, 1.0);
        for k in 1..500 {
            term *= (0.5 * x / k as f64).powi(2);
            sum += term;
            if term < sum * 1e-18 {
                break;
            }
        }
        sum
    }

    /// Kaiser CDF at normalized position `x` across the transition, by composite Simpson
    /// integration of the window: the descending reference for `Kaiser`.
    fn kaiser_cdf_reference(beta: f64, x: f64) -> f64 {
        let window = |t: f64| i0_series(beta * (1.0 - (2.0 * t - 1.0).powi(2)).max(0.0).sqrt());
        let simpson = |a: f64, b: f64| {
            let steps = 20_000;
            let h = (b - a) / steps as f64;
            let inner: f64 = (1..steps)
                .map(|i| window(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
                .sum();
            (window(a) + window(b) + inner) * h / 3.0
        };
        simpson(x, 1.0) / simpson(0.0, 1.0)
    }

    #[test]
    fn kaiser_matches_integrated_window() {
        for beta in [2.0f32, 12.0, 25.0] {
            for n in [5, 8, 33] {
                let centres = kaiser_transition(n, beta, false);
                let edges = kaiser_transition(n, beta, true);
                for i in 0..n {
                    let at_centre = kaiser_cdf_reference(f64::from(beta), (i as f64 + 0.5) / n as f64);
                    let at_edge = kaiser_cdf_reference(f64::from(beta), i as f64 / (n - 1) as f64);
                    assert!((centres[i] - at_centre).abs() < 1e-10, "beta={beta} n={n} i={i}");
                    assert!((edges[i] - at_edge).abs() < 1e-10, "beta={beta} n={n} i={i}");
                }
            }
        }
    }

    #[test]
    fn kaiser_is_exactly_complementary() {
        for straddle in [false, true] {
            for n in [1, 2, 7, 8, 101] {
                if straddle && n == 1 {
                    continue;
                }
                let shape = kaiser_transition(n, 18.0, straddle);
                assert_eq!(shape.len(), n);
                for (a, b) in shape.iter().zip(shape.iter().rev()) {
                    assert_eq!(a + b, 1.0, "n={n}, straddle={straddle}");
                }
                if n % 2 == 1 {
                    assert_eq!(shape[n / 2], 0.5);
                }
            }
        }
        // A single straddling sample spans no transition, leaving a hard cutoff.
        assert_eq!(kaiser_transition(1, 18.0, true), vec![0.0]);
        let edges = kaiser_transition(9, 18.0, true);
        assert_eq!((edges[0], edges[8]), (1.0, 0.0));
    }

    #[test]
    fn layout_does_not_depend_on_sample_type() {
        let tapers = [
            TaperType::Planck,
            TaperType::Cosine(3.4375),
            TaperType::Tanh(3.0),
            TaperType::BetaCdf {
                alpha: 18.0,
                beta: 18.0,
            },
            TaperType::Kaiser(18.0),
        ];
        for taper in tapers {
            for straddle in [false, true] {
                let wide = taper.build_taper::<f64>(12_000, 5_500, 5_000, false, straddle);
                let narrow = taper.build_taper::<f32>(12_000, 5_500, 5_000, false, straddle);
                for (bin, (w, n)) in wide.iter().zip(&narrow).enumerate() {
                    assert_eq!(*w as f32, *n, "{taper:?}, straddle={straddle}, bin {bin}");
                }
            }
        }
    }
}
