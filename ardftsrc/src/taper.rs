use crate::Error;
use crate::beta_reg::beta_reg;
use num_traits::Float;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Transition profile used to shape the cutoff edge of the frequency mask.
pub enum TaperType {
    /// Uses a Planck-taper transition
    Planck,

    /// Uses a cumulative Bessel-I0 taper transition.
    ///
    /// `alpha` controls the steepness of the transition.
    #[cfg(feature = "bessel")]
    Bessel(f32),

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

    /// Beta-CDF taper.
    ///
    /// `alpha` and `beta` are the two Beta distribution shape parameters.
    /// Symmetric:
    ///     BetaCdf { alpha: 10.0, beta: 10.0 }
    ///
    /// Asymmetric:
    ///     BetaCdf { alpha: 8.0, beta: 10.0 }
    ///     BetaCdf { alpha: 10.0, beta: 8.0 }
    BetaCdf { alpha: f32, beta: f32 },
}

impl Default for TaperType {
    fn default() -> Self {
        Self::Cosine(3.4375)
    }
}

impl TaperType {
    /// Builds a full frequency mask of `spectrum_fft_size / 2 + 1` bins: unity passband, a
    /// descending transition ending just before `cutoff_bin`, and zero stopband.
    pub(crate) fn build_taper<T: Float>(
        &self,
        spectrum_fft_size: usize,
        cutoff_bin: usize,
        taper_bins: usize,
        is_passthrough: bool,
    ) -> Vec<T> {
        let len = spectrum_fft_size / 2 + 1;
        if is_passthrough {
            return vec![T::one(); len];
        }
        let transition = self.build_transition::<T>(taper_bins);
        place_transition(len, cutoff_bin, &transition)
    }

    /// Builds only the normalized descending transition shape sampled over `taper_bins` bins,
    /// with leading unity and trailing zero samples trimmed.
    pub(crate) fn build_transition<T: Float>(&self, taper_bins: usize) -> Vec<T> {
        match self {
            TaperType::Planck => planck_transition(taper_bins),
            #[cfg(feature = "bessel")]
            TaperType::Bessel(alpha) => cumulative_bessel_i0_transition(taper_bins, *alpha),
            TaperType::Cosine(alpha) => cosine_transition(taper_bins, *alpha),
            TaperType::BetaCdf { alpha, beta } => beta_cdf_transition(taper_bins, *alpha, *beta),
        }
    }

    /// Returns the normalized position in `[0.0, 1.0]` across the transition (`0.0` = passband
    /// edge, `1.0` = stopband edge) where the descending gain first falls to `gain`.
    ///
    /// The shape is sampled at high resolution with the same builder (and trimming/placement)
    /// used for real masks, so the result is independent of FFT size.
    pub(crate) fn transition_position_at_gain(&self, gain: f64) -> f64 {
        const RESOLUTION: usize = 65_536;

        if gain >= 1.0 {
            return 0.0;
        }
        if gain <= 0.0 {
            return 1.0;
        }

        let transition = self.build_transition::<f64>(RESOLUTION);
        // Trimmed transitions are placed so they end at the stopband edge.
        let offset = (RESOLUTION - transition.len()) as f64;

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

        ((offset + position) / RESOLUTION as f64).clamp(0.0, 1.0)
    }

    /// Validates taper parameters and returns an error for invalid values.
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            TaperType::Planck => Ok(()),
            #[cfg(feature = "bessel")]
            TaperType::Bessel(alpha) => {
                if *alpha <= 0.0 || !alpha.is_finite() {
                    return Err(Error::InvalidAlpha(*alpha));
                } else {
                    Ok(())
                }
            }
            TaperType::Cosine(alpha) => {
                if *alpha <= 0.0 || !alpha.is_finite() {
                    return Err(Error::InvalidAlpha(*alpha));
                } else {
                    Ok(())
                }
            }
            TaperType::BetaCdf { alpha, beta } => {
                if *alpha <= 0.0 || !alpha.is_finite() {
                    return Err(Error::InvalidAlpha(*alpha));
                } else if *beta <= 0.0 || !beta.is_finite() {
                    return Err(Error::InvalidBeta(*beta));
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Lays out a mask of `len` bins: unity below the transition, `transition` ending just before
/// `cutoff_bin`, and zeros from `cutoff_bin` on.
fn place_transition<T: Float>(len: usize, cutoff_bin: usize, transition: &[T]) -> Vec<T> {
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
fn trim_transition<T: Float>(raw: &[T]) -> &[T] {
    let trim_start = raw.iter().position(|value| *value < T::one()).unwrap_or(raw.len());
    let trim_stop = raw
        .iter()
        .rposition(|value| *value > T::zero())
        .map_or(0, |idx| raw.len() - idx - 1);
    let active_end = raw.len().saturating_sub(trim_stop);

    &raw[trim_start..active_end]
}

/// Builds a cumulative Bessel-I0 transition.
#[cfg(feature = "bessel")]
fn cumulative_bessel_i0_transition<T: Float>(taper_bins: usize, alpha: f32) -> Vec<T> {
    if taper_bins == 0 {
        return Vec::new();
    }

    let alpha = f64::from(alpha);
    let n = taper_bins as f64;
    let alpha2 = 4.0 * (alpha * std::f64::consts::PI / n).powi(2);
    let mut raw = vec![0.0; taper_bins];
    let mut scale = 0.0;

    for idx in (0..taper_bins).rev() {
        let idx_f = idx as f64;
        let tmp = idx_f * (n - idx_f) * alpha2;
        raw[idx] = pxfm::f_i0(tmp.sqrt());
        scale += raw[idx];
    }

    let scale = 1.0 / (scale + 1.0);
    let mut sum = 0.0;
    for idx in (0..taper_bins).rev() {
        sum += raw[idx];
        raw[idx] = sum * scale;
    }

    trim_transition(&raw)
        .iter()
        .map(|value| T::from(*value).expect("T should be f64 or f32 and be able to convert from f64"))
        .collect()
}

/// Builds a Planck-taper transition.
fn planck_transition<T: Float>(taper_bins: usize) -> Vec<T> {
    if taper_bins == 0 {
        return Vec::new();
    }
    if taper_bins == 1 {
        return vec![T::one()];
    }

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

    trim_transition(&raw).to_vec()
}

/// Builds a sigmoid-warped cosine transition.
fn cosine_transition<T: Float>(taper_bins: usize, alpha: f32) -> Vec<T> {
    if taper_bins == 0 {
        return Vec::new();
    }
    if taper_bins == 1 {
        return vec![T::one()];
    }

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

    trim_transition(&raw).to_vec()
}

/// Builds a Beta-CDF transition from the regularized lower incomplete beta function.
fn beta_cdf_transition<T: Float>(taper_bins: usize, alpha: f32, beta: f32) -> Vec<T> {
    if taper_bins == 0 {
        return Vec::new();
    }
    if taper_bins == 1 {
        return vec![T::one()];
    }

    let denom = T::from(taper_bins).unwrap() - T::one();

    let raw: Vec<T> = (0..taper_bins)
        .map(|idx| {
            if idx == 0 {
                return T::one();
            }

            if idx == taper_bins - 1 {
                return T::zero();
            }

            let x_t = T::from(idx).unwrap_or_else(T::zero) / denom;
            let x = x_t.to_f64().unwrap_or(0.0).clamp(0.0, 1.0);
            let cdf = beta_reg(alpha as f64, beta as f64, x);
            let value = T::from(1.0 - cdf).expect("T should be f64 or f32 and be able to convert from f64");

            if value.is_normal() {
                value
            } else if value >= T::one() {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect();

    trim_transition(&raw).to_vec()
}

#[cfg(all(test, feature = "bessel"))]
mod tests {
    use super::*;

    #[test]
    fn cumulative_bessel_i0_taper_is_descending_and_bounded() {
        let taper = TaperType::Bessel(6.0).build_taper::<f64>(64, 24, 16, false);
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
    fn cumulative_bessel_i0_passthrough_is_all_ones() {
        let taper = TaperType::Bessel(6.0).build_taper::<f32>(16, 8, 4, true);

        assert_eq!(taper.len(), 9);
        assert!(taper.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn bessel_i0_matches_known_values() {
        assert!((pxfm::f_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((pxfm::f_i0(1.0) - 1.266_065_877_752_008_2).abs() < 1e-15);
        assert!((pxfm::f_i0(2.0) - 2.279_585_302_336_067_3).abs() < 1e-15);
    }
}
