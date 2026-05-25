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
    pub(crate) fn build_taper<T: Float>(
        &self,
        input_fft_size: usize,
        cutoff_bin: usize,
        taper_bins: usize,
        is_passthrough: bool,
    ) -> Vec<T> {
        match self {
            TaperType::Planck => build_planck_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough),
            #[cfg(feature = "bessel")]
            TaperType::Bessel(alpha) => {
                build_cumulative_bessel_i0_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough, *alpha)
            }
            TaperType::Cosine(alpha) => {
                build_cosine_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough, *alpha)
            }
            TaperType::BetaCdf { alpha, beta } => {
                build_beta_cdf_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough, *alpha, *beta)
            }
        }
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

/// Builds a cumulative Bessel-I0 frequency taper.
///
/// Returns passband unity bins, a trimmed descending transition, and stopband zeros.
#[cfg(feature = "bessel")]
fn build_cumulative_bessel_i0_taper<T: Float>(
    input_fft_size: usize,
    cutoff_bin: usize,
    taper_bins: usize,
    is_passthrough: bool,
    alpha: f32,
) -> Vec<T> {
    let mut taper = vec![T::zero(); input_fft_size / 2 + 1];
    let alpha = f64::from(alpha);

    if is_passthrough {
        taper.fill(T::one());
        return taper;
    }

    let transition = if taper_bins == 0 {
        Vec::new()
    } else {
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

        let trim_start = raw.iter().position(|value| *value < 1.0).unwrap_or(raw.len());
        let trim_stop = raw
            .iter()
            .rposition(|value| *value > 0.0)
            .map_or(0, |idx| raw.len() - idx - 1);
        let active_end = raw.len().saturating_sub(trim_stop);

        raw[trim_start..active_end]
            .iter()
            .map(|value| T::from(*value).expect("T should be f64 or f32 and be able to convert from f64"))
            .collect()
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

/// Builds a Planck-taper frequency mask.
///
/// Returns passband unity bins, a Planck-taper transition, and stopband zeros.
fn build_planck_taper<T: Float>(
    input_fft_size: usize,
    cutoff_bin: usize,
    taper_bins: usize,
    is_passthrough: bool,
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
fn build_cosine_taper<T: Float>(
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

/// Builds a Beta-CDF frequency taper from the regularized lower incomplete beta function.
///
/// Returns passband unity bins, a trimmed descending Beta-CDF transition,
/// and stopband zeros.
fn build_beta_cdf_taper<T: Float>(
    input_fft_size: usize,
    cutoff_bin: usize,
    taper_bins: usize,
    is_passthrough: bool,
    alpha: f32,
    beta: f32,
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
