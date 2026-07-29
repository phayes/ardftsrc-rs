//! GPL-derived librempeg-compatible LPC edge extrapolation.
//!
//! This module is a Rust port of the LPC helper routines used by librempeg's
//! `libavfilter/ardftsrc_template.c`: the `autocorr`, `do_lpc`, and `extrapolate`
//! functions that synthesize samples at stream edges before the FFT-domain resampler runs.
//!
//! librempeg is distributed under the GPL
//! (<https://github.com/librempeg/librempeg/blob/master/COPYING.GPLv3>), and these routines
//! were ported from that GPL-licensed implementation rather than independently designed from
//! first principles. That makes this module meaningfully different from the rest of this
//! crate, which is licensed as MIT OR Apache-2.0. GPL code can impose copyleft obligations
//! on downstream users and distributors that do not apply to MIT/Apache code.
//!
//! For that reason, this module is compiled only when the `gpl_lpc` feature is enabled.
//! The default build excludes it and continues to use the crate's Apache / MIT
//! licensed LPC implementation. Enabling `gpl_lpc` is an explicit opt-in to include
//! GPL-derived code in the resulting build.

use num_traits::Float;

/// Maximum LPC order used by librempeg's ardftsrc edge extrapolation.
pub(crate) const LIBREMPEG_LPC_MAX_ORDER: usize = 64;

const LPC_DAMPING_FACTOR: f64 = 0.999;

/// Estimate LPC coefficients with the same autocorrelation and recursion used by librempeg.
///
/// The source implementation accumulates autocorrelation in double precision for every sample
/// format, applies an epsilon-based early stop, then damps the effective coefficients by 0.999^n.
pub(crate) fn librempeg_lpc_coefficients<T>(input: &[T], order: usize) -> Vec<T>
where
    T: Float,
{
    let order = order.min(LIBREMPEG_LPC_MAX_ORDER);
    if order == 0 {
        return Vec::new();
    }

    let autocorr = librempeg_autocorr(input, order);
    let coefficients = librempeg_do_lpc(&autocorr, order);

    coefficients
        .into_iter()
        .map(|coefficient| T::from(coefficient).unwrap_or_else(T::zero))
        .collect()
}

/// Predict `extra` samples after `input` using librempeg-compatible LPC extrapolation.
pub(crate) fn extrapolate_forward<T>(input: &[T], extra: usize) -> Vec<T>
where
    T: Float,
{
    if extra == 0 {
        return Vec::new();
    }
    if input.is_empty() {
        return vec![T::zero(); extra];
    }

    let order = LIBREMPEG_LPC_MAX_ORDER.min(input.len().div_ceil(2));
    let lpc = librempeg_lpc_coefficients(input, order);
    extrapolate_forward_with_coefficients(input, extra, &lpc)
}

/// Predict `extra` samples before `input` using librempeg-compatible LPC extrapolation.
pub(crate) fn extrapolate_backward<T>(input: &[T], extra: usize) -> Vec<T>
where
    T: Float,
{
    if extra == 0 {
        return Vec::new();
    }
    if input.is_empty() {
        return vec![T::zero(); extra];
    }

    let order = LIBREMPEG_LPC_MAX_ORDER.min(input.len().div_ceil(2));
    let lpc = librempeg_lpc_coefficients(input, order);
    extrapolate_backward_with_coefficients(input, extra, &lpc)
}

fn librempeg_autocorr<T>(input: &[T], order: usize) -> Vec<f64>
where
    T: Float,
{
    let mut autocorr = vec![0.0; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0;
        for idx in lag..input.len() {
            let current = input[idx].to_f64().unwrap_or(0.0);
            let delayed = input[idx - lag].to_f64().unwrap_or(0.0);
            sum += current * delayed;
        }
        autocorr[lag] = if sum.is_normal() { sum } else { 0.0 };
    }
    autocorr
}

fn librempeg_do_lpc(autocorr: &[f64], order: usize) -> Vec<f64> {
    let mut lpc = vec![0.0; order];
    let mut max_order = order;
    let mut error = autocorr[0] * (1.0 + 1e-10);
    let epsilon = 1e-9 * autocorr[0] + 1e-10;

    for idx in 0..order {
        if error < epsilon {
            lpc[idx..].fill(0.0);
            max_order = idx;
            break;
        }

        let mut reflection = -autocorr[idx + 1];
        for inner in 0..idx {
            reflection -= lpc[inner] * autocorr[idx - inner];
        }
        reflection /= error;

        lpc[idx] = reflection;
        for inner in 0..(idx / 2) {
            let tmp = lpc[inner];
            lpc[inner] += reflection * lpc[idx - 1 - inner];
            lpc[idx - 1 - inner] += reflection * tmp;
        }

        if idx & 1 == 1 {
            lpc[idx / 2] += lpc[idx / 2] * reflection;
        }

        error *= 1.0 - reflection * reflection;
    }

    let mut damp = LPC_DAMPING_FACTOR;
    for coefficient in lpc.iter_mut().take(max_order) {
        *coefficient *= damp;
        if !coefficient.is_normal() {
            *coefficient = 0.0;
        }
        damp *= LPC_DAMPING_FACTOR;
    }

    if max_order == 0 {
        vec![-1.0]
    } else {
        lpc.truncate(max_order);
        lpc
    }
}

fn extrapolate_forward_with_coefficients<T>(input: &[T], extra: usize, lpc: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut work = input.to_vec();
    let mut output = Vec::with_capacity(extra);

    for _ in 0..extra {
        let base = work.len().saturating_sub(lpc.len());
        let mut next = T::zero();
        for (idx, coefficient) in lpc.iter().rev().enumerate() {
            next = next - work[base + idx] * *coefficient;
        }
        let next = if next.is_finite() { next } else { T::zero() };
        work.push(next);
        output.push(next);
    }

    output
}

fn extrapolate_backward_with_coefficients<T>(input: &[T], extra: usize, lpc: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut work = Vec::with_capacity(extra + input.len());
    work.extend(std::iter::repeat_n(T::zero(), extra));
    work.extend_from_slice(input);

    let input_start = extra;
    for idx in 0..extra {
        let write_idx = input_start - 1 - idx;
        let mut next = T::zero();
        for coefficient_idx in 0..lpc.len() {
            let sample_idx = write_idx + 1 + coefficient_idx;
            next = next - work[sample_idx] * lpc[coefficient_idx];
        }
        work[write_idx] = if next.is_finite() { next } else { T::zero() };
    }

    work[..extra].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_no_nans;

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn coefficients_match_librempeg_recursion_shape() {
        let input = vec![2.0, 3.0, -1.0, -2.0, 1.0, 4.0, 1.0];
        let coefficients = librempeg_lpc_coefficients(&input, 3);
        let expected = vec![-0.6912134693692851, 0.7599840241742992, -0.34471531021563845];

        assert_no_nans(
            &coefficients,
            "gpl_lpc::coefficients_match_librempeg_recursion_shape coefficients",
        );
        assert_close(&coefficients, &expected, 1e-9);
    }

    #[test]
    fn silent_input_uses_librempeg_fallback_coefficient() {
        let coefficients = librempeg_lpc_coefficients(&[0.0f64; 16], 8);

        assert_eq!(coefficients, vec![-1.0]);
    }

    #[test]
    fn backward_extrapolation_uses_original_edge_coefficients() {
        let input = vec![0.2, 0.4, 0.1, -0.3, -0.2, 0.05];
        let predicted = extrapolate_backward(&input, 4);

        assert_no_nans(
            &predicted,
            "gpl_lpc::backward_extrapolation_uses_original_edge_coefficients predicted",
        );
        assert_eq!(predicted.len(), 4);
    }
}
