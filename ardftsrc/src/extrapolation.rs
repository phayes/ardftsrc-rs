//! Configurable strategies for synthesizing samples beyond the real edge of a stream, used by
//! [`crate::window`] whenever `pre`/`post` context doesn't cover everything a window needs.

use num_traits::Float;

use crate::lpc::{self, ExtrapolateFallback};

/// Strategy used to synthesize missing samples at a stream's start or end edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Extrapolation {
    /// Linear-predictive-coding based prediction (default). Adapts to the signal's own
    /// spectral content, but can be ill-conditioned on some windows; see [`crate::lpc`]'s
    /// divergence guard, which fades to silence rather than letting the recursion blow up.
    #[default]
    Lpc,
    /// Reflects existing samples back across the edge (whole-sample mirror, no repeated edge
    /// sample), repeating for spans longer than the available history.
    Mirror,
    /// Silence (zero-fill).
    Zero,
}

impl Extrapolation {
    /// Extrapolates `extra` samples forward, continuing on from the end of `input`.
    pub(crate) fn forward<T: Float>(self, input: &[T], extra: usize) -> Vec<T> {
        if extra == 0 {
            return Vec::new();
        }
        match self {
            Extrapolation::Lpc => lpc::extrapolate_forward(input, extra, ExtrapolateFallback::Hold),
            Extrapolation::Mirror => mirror_forward(input, extra),
            Extrapolation::Zero => vec![T::zero(); extra],
        }
    }

    /// Extrapolates `extra` samples backward, continuing before the start of `input`, by
    /// reversing `input`, running the forward extrapolation on that reversed sequence, then
    /// reversing the result back into chronological order.
    pub(crate) fn reverse<T: Float>(self, input: &[T], extra: usize) -> Vec<T> {
        let mut reversed_input = input.to_vec();
        reversed_input.reverse();
        let mut predicted = self.forward(&reversed_input, extra);
        predicted.reverse();
        predicted
    }
}

/// Whole-sample mirror continuation: `input[n-2], input[n-3], ..., input[0], input[1], ...`,
/// repeating with period `2 * (n - 1)` for spans longer than `input`.
fn mirror_forward<T: Float>(input: &[T], extra: usize) -> Vec<T> {
    let n = input.len();
    if n == 0 {
        return vec![T::zero(); extra];
    }
    if n == 1 {
        return vec![input[0]; extra];
    }

    let period = 2 * (n - 1);
    (0..extra)
        .map(|i| {
            let position = (n + i) % period;
            let idx = if position >= n { period - position } else { position };
            input[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_no_nans;

    #[test]
    fn zero_strategy_fills_silence() {
        let input = [1.0f64, 2.0, 3.0];
        assert_eq!(Extrapolation::Zero.forward(&input, 4), vec![0.0; 4]);
        assert_eq!(Extrapolation::Zero.reverse(&input, 4), vec![0.0; 4]);
    }

    #[test]
    fn forward_and_reverse_return_empty_for_zero_extra() {
        let input = [1.0f64, 2.0, 3.0];
        for strategy in [Extrapolation::Lpc, Extrapolation::Mirror, Extrapolation::Zero] {
            assert!(strategy.forward(&input, 0).is_empty());
            assert!(strategy.reverse(&input, 0).is_empty());
        }
    }

    #[test]
    fn mirror_reflects_without_repeating_the_edge_sample() {
        let input = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let predicted = mirror_forward(&input, 8);
        assert_eq!(predicted, vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mirror_handles_single_sample_input() {
        let input = [5.0f64];
        assert_eq!(mirror_forward(&input, 3), vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn mirror_reverse_is_symmetric_with_forward() {
        let input = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let mut reversed_input = input;
        reversed_input.reverse();
        let mut expected = mirror_forward(&reversed_input, 6);
        expected.reverse();
        assert_eq!(Extrapolation::Mirror.reverse(&input, 6), expected);
    }

    #[test]
    fn lpc_strategy_matches_crate_lpc_module() {
        let input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let expected_forward = lpc::extrapolate_forward(&input, 16, ExtrapolateFallback::Hold);
        assert_eq!(Extrapolation::Lpc.forward(&input, 16), expected_forward);

        let expected_backward = lpc::extrapolate_backward(&input, 16, ExtrapolateFallback::Hold);
        let actual_backward = Extrapolation::Lpc.reverse(&input, 16);
        assert_no_nans(&actual_backward, "extrapolation::lpc_strategy_matches_crate_lpc_module");
        assert_eq!(actual_backward, expected_backward);
    }
}
