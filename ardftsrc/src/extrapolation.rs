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
    /// Point reflection through the edge sample: each synthetic sample is `2 * edge - x`, where
    /// `x` is the sample the same distance inside the edge. Both the value and the slope stay
    /// continuous across the edge, so the join adds no kink. Spans longer than the available
    /// history continue as a [`Mirror`](Self::Mirror) of the reflected sequence.
    OddMirror,
    /// Reflects existing samples back across the edge (whole-sample mirror, no repeated edge
    /// sample), repeating for spans longer than the available history.
    ///
    /// The value is continuous across the edge, but the slope reverses there.
    Mirror,
    /// Silence (zero-fill).
    ///
    /// Can click if the signal near the extrapolation edge is not near zero.
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
            Extrapolation::OddMirror => odd_mirror_forward(input, extra),
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

/// Point-reflection continuation: `2 * e - input[n-2], 2 * e - input[n-3], ..., 2 * e - input[0]`
/// where `e = input[n-1]`. Spans longer than that continue as a whole-sample [`mirror_forward`]
/// of `input` followed by its reflection, which stays bounded instead of drifting by
/// `2 * (e - input[0])` per further reflection.
fn odd_mirror_forward<T: Float>(input: &[T], extra: usize) -> Vec<T> {
    let n = input.len();
    if n == 0 {
        return vec![T::zero(); extra];
    }

    let edge = input[n - 1];
    let reflected = extra.min(n - 1);
    let mut output: Vec<T> = (0..reflected).map(|i| edge + edge - input[n - 2 - i]).collect();
    if extra > reflected {
        let mut extended = Vec::with_capacity(n + reflected);
        extended.extend_from_slice(input);
        extended.extend_from_slice(&output);
        output.extend(mirror_forward(&extended, extra - reflected));
    }
    output
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
        for strategy in [Extrapolation::Lpc, Extrapolation::OddMirror, Extrapolation::Mirror, Extrapolation::Zero] {
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
    fn odd_mirror_reflects_through_the_edge_sample() {
        let input = [0.0f64, 1.0, 4.0, 5.0];
        // Edge 5: 2*5 - 4, 2*5 - 1, 2*5 - 0.
        assert_eq!(odd_mirror_forward(&input, 3), vec![6.0, 9.0, 10.0]);
    }

    #[test]
    fn odd_mirror_continues_a_linear_ramp_exactly() {
        let input: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let predicted = odd_mirror_forward(&input, 7);
        let expected: Vec<f64> = (8..15).map(|i| i as f64 * 0.5).collect();
        assert_eq!(predicted, expected);
    }

    #[test]
    fn odd_mirror_mirrors_the_reflected_sequence_past_the_history() {
        let input = [0.0f64, 1.0, 4.0, 5.0];
        // Reflection [6, 9, 10], then a whole-sample mirror of [0, 1, 4, 5, 6, 9, 10].
        let predicted = odd_mirror_forward(&input, 9);
        assert_eq!(predicted, vec![6.0, 9.0, 10.0, 9.0, 6.0, 5.0, 4.0, 1.0, 0.0]);
    }

    #[test]
    fn odd_mirror_handles_degenerate_inputs() {
        assert_eq!(odd_mirror_forward::<f64>(&[], 3), vec![0.0; 3]);
        assert_eq!(odd_mirror_forward(&[5.0f64], 3), vec![5.0; 3]);
    }

    #[test]
    fn odd_mirror_reverse_reflects_through_the_first_sample() {
        let input = [5.0f64, 4.0, 1.0, 0.0];
        assert_eq!(Extrapolation::OddMirror.reverse(&input, 3), vec![10.0, 9.0, 6.0]);
    }

    #[test]
    fn lpc_strategy_matches_crate_lpc_module() {
        let input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let expected_forward = lpc::extrapolate_forward(&input, 16, ExtrapolateFallback::Hold);
        assert_eq!(Extrapolation::Lpc.forward(&input, 16), expected_forward);

        let mut reversed = input.clone();
        reversed.reverse();
        let mut expected_backward = lpc::extrapolate_forward(&reversed, 16, ExtrapolateFallback::Hold);
        expected_backward.reverse();
        let actual_backward = Extrapolation::Lpc.reverse(&input, 16);
        assert_no_nans(&actual_backward, "extrapolation::lpc_strategy_matches_crate_lpc_module");
        assert_eq!(actual_backward, expected_backward);
    }
}
