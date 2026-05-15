use crate::Error;
use num_traits::Float;
use crate::beta_reg::beta_reg;

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
            TaperType::Cosine(alpha) => {
                build_cosine_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough, *alpha)
            }
            TaperType::BetaCdf { alpha, beta } => {
                build_beta_cdf_taper(input_fft_size, cutoff_bin, taper_bins, is_passthrough, *alpha, *beta)
            }
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        match self {
            TaperType::Planck => Ok(()),
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

/// Builds a Beta-CDF frequency taper.
///
/// Returns passband unity bins, a trimmed descending Beta-CDF transition,
/// and stopband zeros.
///
/// `order` controls the shape of the transition:
///
/// - `order = 1.0` gives a linear-ish transition.
/// - `order > 1.0` gives an S-shaped transition.
/// - `order = 24.0` gives a very flat-at-the-edges, steep-in-the-middle
///   transition equivalent to `1 - BetaCDF(x; 24, 24)`.
///
/// TODO: Validate alpha and beta are positive.
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
