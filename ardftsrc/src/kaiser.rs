/// Largest `beta` accepted by the Kaiser taper.
pub(crate) const MAX_KAISER_BETA: f32 = 300.0;

/// Power series of the Kaiser window's I0 for one `beta`:
/// `I0(beta * sqrt(w)) = sum_k c_k w^k` with `c_k = (beta/2)^(2k) / (k!)^2`.
///
/// With `beta` at most [`MAX_KAISER_BETA`] the largest coefficient is below 1e129, so the
/// series is summed directly in `f64`.
pub(crate) struct KaiserSeries {
    coef: Vec<f64>,
}

impl KaiserSeries {
    pub(crate) fn new(beta: f64) -> Self {
        let q = beta * beta / 4.0;
        let tiny = f64::EPSILON * 1.0e-4;
        let mut coef = Vec::new();
        let (mut c, mut cmax) = (1.0f64, 1.0f64);
        for k in 1..100_000 {
            coef.push(c);
            cmax = cmax.max(c);
            c *= q / (k as f64 * k as f64);
            if k > 8 && c < cmax * tiny {
                break;
            }
        }
        Self { coef }
    }

    /// Integral of `I0(beta * sqrt(1 - v^2))` over `[0, u]` for `u` in `[0, 1]`, term by term
    /// with `J_k` = integral of `(1 - v^2)^k`, from `(2k + 1) J_k = 2k J_(k-1) + u (1 - u^2)^k`.
    pub(crate) fn half_integral(&self, u: f64) -> f64 {
        let w = 1.0 - u * u;
        let (mut jk, mut wk) = (u, 1.0);
        let mut sum = self.coef[0] * jk;
        for (k, c) in self.coef.iter().enumerate().skip(1) {
            wk *= w;
            jk = (2.0 * k as f64 * jk + u * wk) / (2 * k + 1) as f64;
            sum += c * jk;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaiser_series_integrates_the_window() {
        // The integral of I0(beta * sqrt(1 - v^2)) over [0, 1] is sinh(beta) / beta.
        for beta in [1.0e-3, 1.0, 2.0, 10.0, 50.0, 100.0, f64::from(MAX_KAISER_BETA)] {
            let exact = beta.sinh() / beta;
            let error = (KaiserSeries::new(beta).half_integral(1.0) - exact).abs() / exact;
            assert!(error < 1e-13, "beta={beta}: relative error {error}");
        }
    }
}
