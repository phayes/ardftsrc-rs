//! This code was orignially copied from the `linear-predictive-coding` crate and adapted for our use here.
//! Our thanks to the authors of the linear-predictive-coding crate!

use num_traits::Float;

/// Maximum LPC order used during extrapolation.
pub(crate) const EXTRAPOLATION_MAX_ORDER: usize = 64;

/// Per-coefficient damping multiplier for LPC stabilization.
pub(crate) const LPC_DAMPING_FACTOR: f64 = 0.999;

/// Fallback behavior used when LPC fitting fails.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtrapolateFallback {
    /// Zero-order hold (repeat last sample)
    Hold,
    /// Silence.
    #[allow(dead_code)]
    Silence,
}

impl ExtrapolateFallback {
    fn value<T>(&self) -> T
    where
        T: Float,
    {
        match self {
            ExtrapolateFallback::Hold => -T::one(),
            ExtrapolateFallback::Silence => T::zero(),
        }
    }
}

/// Compute autocorrelation-like lag sums for `a`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn correlate<T>(a: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut out = Vec::with_capacity(a.len());
    for lag in 0..a.len() {
        let mut sum = T::zero();
        for idx in lag..a.len() {
            sum = sum + a[idx] * a[idx - lag];
        }
        out.push(sum);
    }
    out
}

/// Levinson-Durbin recursion for LPC coefficients.
///
/// Returns `(coefficients, prediction_error)` where `coefficients` has length `depth`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn calc_lpc_by_levinson_durbin<T>(a: &[T], depth: usize) -> Option<(Vec<T>, T)>
where
    T: Float,
{
    if depth == 0 {
        return Some((Vec::new(), T::zero()));
    }
    if a.len() < depth {
        return None;
    }

    let r = correlate(a);
    if r.is_empty() || !r[0].is_finite() || r[0] == T::zero() {
        return None;
    }

    let mut coeff = vec![T::one()];
    let mut error = r[0];

    for m in 1..=depth {
        if !error.is_finite() || error == T::zero() {
            return None;
        }

        let mut dot = T::zero();
        for i in 0..m {
            dot = dot + coeff[i] * r[m - i];
        }
        let k = -dot / error;

        let mut next = vec![T::zero(); m + 1];
        // u = [coeff..., 0], v = reverse(u), next = u + k*v
        for i in 0..=m {
            let u = if i < m { coeff[i] } else { T::zero() };
            let v = if i == 0 { T::zero() } else { coeff[m - i] };
            next[i] = u + v * k;
        }

        coeff = next;
        error = error * (T::one() - k * k);
    }

    Some((coeff[1..].to_vec(), error))
}

/// Burg method for LPC coefficients.
///
/// Returns coefficient vector of length `depth`.
pub(crate) fn calc_lpc_by_burg<T>(x: &[T], depth: usize) -> Option<Vec<T>>
where
    T: Float,
{
    if depth == 0 {
        return Some(Vec::new());
    }
    if x.len() < depth {
        return None;
    }

    let n = x.len();
    let mut a = vec![T::zero(); depth + 1];
    a[0] = T::one();

    let two = T::one() + T::one();
    let mut f = x.to_vec();
    let mut b = x.to_vec();

    for p in 0..depth {
        let forward = &f[p + 1..];
        let backward = &b[..n - p - 1];

        let mut d = T::zero();
        let mut num = T::zero();
        for (ff, bb) in forward.iter().zip(backward.iter()) {
            d = d + *ff * *ff + *bb * *bb;
            num = num + *ff * *bb;
        }
        if d == T::zero() || !d.is_finite() {
            return None;
        }

        let k = -two * num / d;
        if !k.is_finite() {
            return None;
        }

        let mut u = a[..=p + 1].to_vec();
        let mut v = u.clone();
        v.reverse();
        for i in 0..u.len() {
            u[i] = u[i] + v[i] * k;
        }
        a[..=p + 1].copy_from_slice(&u);

        let mut f_updates = vec![T::zero(); n - p - 1];
        let mut b_updates = vec![T::zero(); n - p - 1];
        for i in 0..(n - p - 1) {
            f_updates[i] = b[i] * k;
            b_updates[i] = f[p + 1 + i] * k;
        }
        for i in 0..(n - p - 1) {
            f[p + 1 + i] = f[p + 1 + i] + f_updates[i];
            b[i] = b[i] + b_updates[i];
        }
    }

    Some(a[1..].to_vec())
}

/// Predict `extra` forward samples via LPC with finite-value guards.
pub(crate) fn extrapolate_forward<T>(input: &[T], extra: usize, fallback: ExtrapolateFallback) -> Vec<T>
where
    T: Float,
{
    if extra == 0 {
        return Vec::new();
    }
    if input.is_empty() {
        return vec![T::zero(); extra];
    }

    let order = EXTRAPOLATION_MAX_ORDER.min((input.len() + 1) / 2).max(1);
    let lpc = lpc_coefficients(input, order, fallback);
    let mut work = input.to_vec();
    let mut output = Vec::with_capacity(extra);

    for _ in 0..extra {
        let base = work.len().saturating_sub(lpc.len());
        let mut next = T::zero();
        for (idx, coeff) in lpc.iter().rev().enumerate() {
            next = next - work[base + idx] * *coeff;
        }
        let next = if next.is_finite() { next } else { T::zero() };
        work.push(next);
        output.push(next);
    }

    output
}

/// Predict preceding samples by reversing, forward-extrapolating, then reversing back.
pub(crate) fn extrapolate_backward<T>(input: &[T], extra: usize, fallback: ExtrapolateFallback) -> Vec<T>
where
    T: Float,
{
    let mut reversed = input.to_vec();
    reversed.reverse();
    let mut predicted = extrapolate_forward(&reversed, extra, fallback);
    predicted.reverse();
    predicted
}

/// Estimate damped LPC coefficients via Burg with finite guards.
pub(crate) fn lpc_coefficients<T>(input: &[T], order: usize, fallback: ExtrapolateFallback) -> Vec<T>
where
    T: Float,
{
    if order == 0 {
        return Vec::new();
    }

    let depth = order.min(input.len()).max(1);
    let sanitized_input: Vec<T> = input
        .iter()
        .map(|sample| if sample.is_finite() { *sample } else { T::zero() })
        .collect();

    let coefficients = calc_lpc_by_burg(&sanitized_input, depth).unwrap_or_default();

    if coefficients.is_empty() {
        return vec![fallback.value::<T>()];
    }

    let mut lpc = coefficients;
    let mut damp = T::from(LPC_DAMPING_FACTOR).unwrap_or(T::one());
    for coeff in &mut lpc {
        *coeff = *coeff * damp;
        if !coeff.is_finite() {
            *coeff = T::zero();
        }
        damp = damp * T::from(LPC_DAMPING_FACTOR).unwrap_or(T::one());
    }

    lpc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= tol, "a={a}, e={e}, tol={tol}");
        }
    }

    #[test]
    fn test_correlate() {
        let a = vec![2.0, 3.0, -1.0, -2.0, 1.0, 4.0, 1.0];
        let expected = vec![36.0, 11.0, -16.0, -7.0, 13.0, 11.0, 2.0];
        assert_eq!(correlate(&a), expected);
    }

    #[test]
    fn test_calc_lpc_by_levinson_durbin() {
        let a = vec![2.0, 3.0, -1.0, -2.0, 1.0, 4.0, 1.0];
        let depth = 3;
        let expected = vec![-0.6919053749597684, 0.7615062761506278, -0.3457515288059223];
        let (coeff, _err) = calc_lpc_by_levinson_durbin(&a, depth).unwrap();
        assert_close(&coeff, &expected, 1e-12);
    }

    #[test]
    fn test_calc_lpc_by_burg() {
        let a = vec![2.0, 3.0, -1.0, -2.0, 1.0, 4.0, 1.0];
        let depth = 3;
        let expected = vec![-1.0650404360323664, 1.157238171254371, -0.5771692748969812];
        let coeff = calc_lpc_by_burg(&a, depth).unwrap();
        assert_close(&coeff, &expected, 1e-12);
    }

    #[test]
    fn test_extrapolate_fallback_values() {
        assert_eq!(ExtrapolateFallback::Hold.value::<f64>(), -1.0);
        assert_eq!(ExtrapolateFallback::Silence.value::<f64>(), 0.0);
    }

    #[test]
    fn test_lpc_coefficients_fallback_modes() {
        let input = vec![0.0f64; 16];
        let hold_coeff = lpc_coefficients(&input, 8, ExtrapolateFallback::Hold);
        let silence_coeff = lpc_coefficients(&input, 8, ExtrapolateFallback::Silence);
        assert_eq!(hold_coeff, vec![-1.0]);
        assert_eq!(silence_coeff, vec![0.0]);
    }

    #[test]
    fn test_extrapolate_forward_empty_and_zero_extra() {
        assert!(extrapolate_forward::<f64>(&[1.0, 2.0], 0, ExtrapolateFallback::Hold).is_empty());
        assert_eq!(
            extrapolate_forward::<f64>(&[], 3, ExtrapolateFallback::Hold),
            vec![0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_extrapolate_backward_length_and_finite() {
        let input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let predicted = extrapolate_backward(&input, 24, ExtrapolateFallback::Hold);
        assert_eq!(predicted.len(), 24);
        assert!(predicted.iter().all(|v| v.is_finite()));
    }
}
