//! Full-precision sine/cosine built only from [`HighPrecisionFloat`] arithmetic and rounding, for
//! backends whose crates don't provide trigonometry at their full precision.
//!
//! Range reduction follows the Cody-Waite shape used by extended-precision libraries: reduce to
//! `[-pi/4, pi/4]` by subtracting the nearest multiple of `pi/2`, then evaluate a polynomial in
//! that restricted range and use the quadrant to combine sin/cos.
//!
//! The polynomial is a plain Taylor series rather than a minimax fit: on `[-pi/4, pi/4]` the
//! Taylor remainder is already below `1e-40` by the 16th term (the factorial in the denominator
//! dominates), past the ulp of every backend that uses this module, so no externally-sourced
//! coefficient table is needed.

use super::HighPrecisionFloat;

const TERMS: usize = 16;

/// Taylor coefficients at a backend's precision. Build once per backend (e.g. in a `LazyLock`).
pub(super) struct Coefficients<T> {
    /// Coefficient of `x^(2k+1)` in the Taylor series of `sin(x)`, for `k = 1..=TERMS`.
    sin: [T; TERMS],
    /// Coefficient of `x^(2k)` in the Taylor series of `cos(x)`, for `k = 1..=TERMS`.
    cos: [T; TERMS],
}

impl<T: HighPrecisionFloat> Coefficients<T> {
    /// Computes `(-1)^k / n!` by dividing by each small (exactly representable) integer `n` in
    /// turn, so every coefficient is within a few ulps of exact.
    pub(super) fn new() -> Self {
        let mut sin = [T::zero(); TERMS];
        let mut cos = [T::zero(); TERMS];
        let mut inverse_factorial = T::one();
        for n in 1..=2 * TERMS + 1 {
            inverse_factorial = inverse_factorial / T::from_f64_exact(n as f64);
            let k = n / 2;
            if k == 0 {
                continue;
            }
            let signed = if k % 2 == 1 { -inverse_factorial } else { inverse_factorial };
            if n % 2 == 0 {
                cos[k - 1] = signed;
            } else {
                sin[k - 1] = signed;
            }
        }
        Self { sin, cos }
    }
}

/// Evaluates `1 + x2 * (c[0] + x2 * (c[1] + ...))` by Horner's rule.
fn one_plus_series<T: HighPrecisionFloat>(x2: T, coeffs: &[T; TERMS]) -> T {
    let mut acc = coeffs[TERMS - 1];
    for &coeff in coeffs[..TERMS - 1].iter().rev() {
        acc = acc * x2 + coeff;
    }
    T::one() + x2 * acc
}

pub(super) fn sin_cos<T: HighPrecisionFloat>(x: T, coefficients: &Coefficients<T>) -> (T, T) {
    // Dividing by powers of two is exact.
    let frac_pi_2 = T::tau() / T::from_f64_exact(4.0);
    let frac_pi_4 = T::tau() / T::from_f64_exact(8.0);

    let (r, quadrant) = if x.hp_abs() < frac_pi_4 {
        (x, 0)
    } else {
        let quotient = (x / frac_pi_2).hp_round();
        (x - quotient * frac_pi_2, (quotient.to_f64() as i64).rem_euclid(4))
    };

    let r2 = r * r;
    let s = r * one_plus_series(r2, &coefficients.sin);
    let c = one_plus_series(r2, &coefficients.cos);
    match quadrant {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}
