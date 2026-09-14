use num_traits::Float;
use std::collections::VecDeque;

/// Minimum/maximum tap count for a decimation stage's FIR filter.
///
/// The lower bound keeps the filter from becoming a near-no-op at very lenient bandwidths; the
/// upper bound caps CPU/memory cost at very strict bandwidths (transition band close to zero
/// would otherwise imply an arbitrarily long filter).
const MIN_TAPS: usize = 15;
const MAX_TAPS: usize = 511;

/// Chooses how many 2:1 decimation stages to run ahead of the FFT resampler for a downsampling
/// conversion, so that the FFT stage still performs at least a genuine 2:1 reduction and remains
/// the dominant band-limiting element of the pipeline.
///
/// Each stage halves the working sample rate. We only add a stage while the resulting rate would
/// still be at least `2 * output_sample_rate`, i.e. the largest `n` such that
/// `input_sample_rate / 2^(n + 1) >= output_sample_rate`. This is derived purely from the rate
/// ratio.
pub(crate) fn decimation_stage_count(input_sample_rate: usize, output_sample_rate: usize) -> usize {
    if output_sample_rate == 0 || input_sample_rate <= output_sample_rate {
        return 0;
    }

    let ratio = input_sample_rate as f64 / output_sample_rate as f64;
    // n <= log2(ratio) - 1 == log2(ratio / 2)
    let max_stages = (ratio / 2.0).log2().floor();
    if max_stages < 1.0 {
        return 0;
    }

    max_stages as usize
}

/// Designs windowed-sinc lowpass FIR coefficients for one 2:1 decimation stage.
///
/// `bandwidth` is the same normalized `[0.0, 1.0]` value used by [`Config::bandwidth`](crate::Config::bandwidth)
/// for the final FFT resampler: the fraction of the stage's *post-decimation* Nyquist frequency
/// (`Fs/4`, where `Fs` is this stage's input rate) that is kept as a flat passband. Reusing the
/// caller's own bandwidth setting means the decimation prefilter makes exactly the same
/// passband/transition-band tradeoff the caller already chose for output quality.
///
/// The stage-selection rule in [`decimation_stage_count`] guarantees the FFT resampler stage
/// downstream still sees at least a 2:1 ratio of its own, so this prefilter never has to be the
/// sole band-limiting element -- it only needs to be good enough that nothing it lets alias
/// lands inside the passband the final stage still promises.
///
/// `max_group_delay` caps each stage's tap count so its group delay (`(taps - 1) / 2`) doesn't
/// grow unboundedly relative to the FFT stage's own (decimated-domain) chunk size. At
/// end-of-stream the cascade's trailing state has to be flushed and fed into that one remaining
/// chunk (see [`super::core`]'s finalize handling), so keeping the delay in the same ballpark as
/// the chunk size keeps that flush effectively lossless instead of needing truncation.
pub(crate) fn design_decimation_taps<T: Float>(bandwidth: f32, max_group_delay: usize) -> Vec<T> {
    let bandwidth = f64::from(bandwidth).clamp(0.0, 1.0);

    // Post-decimation Nyquist (Fs/4) expressed as a fraction of the full (pre-decimation) sample
    // rate (where 1.0 == Fs, Nyquist == 0.5).
    const NEW_NYQUIST_FRAC: f64 = 0.25;

    let cutoff_frac = bandwidth * NEW_NYQUIST_FRAC;
    let transition_frac = ((1.0 - bandwidth) * NEW_NYQUIST_FRAC).max(1e-4);

    // Blackman-window transition-width rule of thumb: transition_width ~= 5.5 / num_taps
    // (both expressed as a fraction of the sample rate).
    let raw_taps = (5.5 / transition_frac).ceil() as usize;
    let delay_cap_taps = max_group_delay.saturating_mul(2).saturating_add(1);
    let num_taps = raw_taps.clamp(MIN_TAPS, MAX_TAPS).min(delay_cap_taps.max(MIN_TAPS)) | 1;

    let mid = (num_taps - 1) as f64 / 2.0;
    let mut taps = vec![0.0_f64; num_taps];
    let mut sum = 0.0;
    for (n, tap) in taps.iter_mut().enumerate() {
        let x = n as f64 - mid;
        let sinc = if x.abs() < 1e-12 {
            2.0 * cutoff_frac
        } else {
            (2.0 * std::f64::consts::PI * cutoff_frac * x).sin() / (std::f64::consts::PI * x)
        };
        *tap = sinc * blackman_window(n, num_taps);
        sum += *tap;
    }

    // Normalize to unity DC gain.
    if sum != 0.0 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }

    taps.into_iter()
        .map(|value| T::from(value).unwrap_or_else(T::zero))
        .collect()
}

fn blackman_window(n: usize, len: usize) -> f64 {
    let denom = (len - 1) as f64;
    if denom == 0.0 {
        return 1.0;
    }
    let two_pi_n = 2.0 * std::f64::consts::PI * n as f64 / denom;
    0.42 - 0.5 * two_pi_n.cos() + 0.08 * (2.0 * two_pi_n).cos()
}

/// A single linear-phase FIR lowpass filter followed by a 2:1 downsample.
struct FirDecimator<T: Float> {
    taps: Vec<T>,
    delay_line: VecDeque<T>,
    /// `true` when the *next* sample pushed into the delay line should produce an output sample.
    emit_next: bool,
    /// Last real sample pushed in, used as hold-padding when flushing at end-of-stream.
    last_sample: T,
}

impl<T: Float> FirDecimator<T> {
    fn new(taps: Vec<T>) -> Self {
        let len = taps.len();
        Self {
            taps,
            delay_line: VecDeque::from(vec![T::zero(); len]),
            emit_next: true,
            last_sample: T::zero(),
        }
    }

    fn reset(&mut self) {
        for sample in &mut self.delay_line {
            *sample = T::zero();
        }
        self.emit_next = true;
        self.last_sample = T::zero();
    }

    /// Feeds `input` through the filter, appending decimated (2:1) output samples to `out`.
    fn process_into(&mut self, input: &[T], out: &mut Vec<T>) {
        for &sample in input {
            self.delay_line.pop_front();
            self.delay_line.push_back(sample);
            self.last_sample = sample;

            if self.emit_next {
                let mut acc = T::zero();
                for (tap, delayed) in self.taps.iter().zip(self.delay_line.iter()) {
                    acc = acc + *tap * *delayed;
                }
                out.push(acc);
            }
            self.emit_next = !self.emit_next;
        }
    }

    /// Flushes the filter's remaining delay-line state by feeding it hold-padding (repeats of
    /// the last real sample), so real samples still "in flight" inside the filter at
    /// end-of-stream get pushed out into `out` instead of being silently dropped.
    fn flush_into(&mut self, out: &mut Vec<T>) {
        let padding = vec![self.last_sample; self.taps.len()];
        self.process_into(&padding, out);
    }

    fn group_delay(&self) -> usize {
        (self.taps.len() - 1) / 2
    }
}

/// A cascade of 2:1 decimation stages, each halving the sample rate.
///
/// Empty (zero-stage) chains are a transparent passthrough, so this type is cheap to keep around
/// unconditionally on the hot path when decimation is disabled.
pub(crate) struct DecimationChain<T: Float> {
    stages: Vec<FirDecimator<T>>,
    // Ping-pong scratch buffers for cascading through more than one stage without extra
    // allocation on the steady-state streaming path.
    buf_a: Vec<T>,
    buf_b: Vec<T>,
}

impl<T: Float> DecimationChain<T> {
    pub(crate) fn new(num_stages: usize, taps: &[T]) -> Self {
        Self {
            stages: (0..num_stages).map(|_| FirDecimator::new(taps.to_vec())).collect(),
            buf_a: Vec::new(),
            buf_b: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn num_stages(&self) -> usize {
        self.stages.len()
    }

    pub(crate) fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }

    /// Cascades `input` through every decimation stage (halving the sample count at each stage)
    /// and writes the final decimated result into `out` (cleared first).
    pub(crate) fn process(&mut self, input: &[T], out: &mut Vec<T>) {
        out.clear();

        match self.stages.len() {
            0 => out.extend_from_slice(input),
            1 => self.stages[0].process_into(input, out),
            _ => {
                self.buf_a.clear();
                self.stages[0].process_into(input, &mut self.buf_a);

                let last = self.stages.len() - 1;
                for stage in &mut self.stages[1..last] {
                    self.buf_b.clear();
                    stage.process_into(&self.buf_a, &mut self.buf_b);
                    std::mem::swap(&mut self.buf_a, &mut self.buf_b);
                }

                self.stages[last].process_into(&self.buf_a, out);
            }
        }
    }

    /// Flushes every stage's remaining delay-line state, in cascade order, so real trailing
    /// samples still buffered inside the FIR filters at end-of-stream aren't silently dropped.
    ///
    /// Each stage's own flush output must itself pass through every downstream stage (and be
    /// followed by *that* stage's flush) before the cascade is fully drained, since downstream
    /// stages have their own delay lines holding real data fed to them earlier.
    pub(crate) fn flush(&mut self, out: &mut Vec<T>) {
        out.clear();
        if self.stages.is_empty() {
            return;
        }

        let mut current = Vec::new();
        self.stages[0].flush_into(&mut current);

        for stage in self.stages.iter_mut().skip(1) {
            let mut next = Vec::new();
            stage.process_into(&current, &mut next);
            stage.flush_into(&mut next);
            current = next;
        }

        *out = current;
    }

    /// Total algorithmic group delay of the cascade, expressed in samples at the pre-decimation
    /// (raw) rate.
    pub(crate) fn raw_group_delay(&self) -> usize {
        let mut delay = 0usize;
        let mut scale = 1usize;
        for stage in &self.stages {
            delay += stage.group_delay() * scale;
            scale *= 2;
        }
        delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_finite_and_bounded(samples: &[f64], bound: f64) -> bool {
        samples.iter().all(|sample| sample.is_finite() && sample.abs() <= bound)
    }

    #[test]
    fn stage_count_is_zero_below_4x_ratio() {
        assert_eq!(decimation_stage_count(44_100, 44_100), 0);
        assert_eq!(decimation_stage_count(48_000, 44_100), 0);
        assert_eq!(decimation_stage_count(88_200, 44_100), 0); // exactly 2x
        assert_eq!(decimation_stage_count(44_100, 48_000), 0); // upsampling
    }

    #[test]
    fn stage_count_keeps_final_stage_at_least_2x() {
        for (input_rate, output_rate) in [
            (176_400, 44_100),
            (192_000, 8_000),
            (384_000, 8_000),
            (2_822_400, 44_100), // DSD-ish extreme ratio
        ] {
            let stages = decimation_stage_count(input_rate, output_rate);
            let decimated_rate = input_rate >> stages;
            assert!(
                decimated_rate >= 2 * output_rate,
                "decimated_rate={decimated_rate} should stay >= 2x output_rate={output_rate}"
            );
            // Adding one more stage should violate the invariant (otherwise we under-decimated).
            let decimated_rate_one_more = input_rate >> (stages + 1);
            assert!(
                decimated_rate_one_more < 2 * output_rate,
                "stage_count={stages} was not maximal for {input_rate} -> {output_rate}"
            );
        }
    }

    #[test]
    fn taps_have_unity_dc_gain() {
        for bandwidth in [0.5f32, 0.8, 0.9114534, 0.9873534, 0.9952346] {
            let taps: Vec<f64> = design_decimation_taps(bandwidth, usize::MAX);
            let sum: f64 = taps.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "bandwidth={bandwidth}, sum={sum}");
            assert!(taps.len() >= MIN_TAPS && taps.len() <= MAX_TAPS);
            assert_eq!(taps.len() % 2, 1, "tap count should be odd for linear phase");
        }
    }

    #[test]
    fn chain_halves_sample_count_per_stage() {
        let taps: Vec<f64> = design_decimation_taps(0.9, usize::MAX);
        let mut chain = DecimationChain::new(3, &taps);
        let input = vec![0.0f64; 800];
        let mut out = Vec::new();
        chain.process(&input, &mut out);
        assert_eq!(out.len(), input.len() / 8);
    }

    #[test]
    fn chain_attenuates_high_frequency_tone() {
        // A tone right at the original Nyquist should be almost entirely removed after one
        // decimation stage with a reasonably conservative bandwidth.
        let taps: Vec<f64> = design_decimation_taps(0.9, usize::MAX);
        let mut chain = DecimationChain::new(1, &taps);

        let n = 4096;
        let input: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(); // Nyquist tone
        let mut out = Vec::new();
        chain.process(&input, &mut out);

        assert!(is_finite_and_bounded(&out, 1.0));
        let settled = &out[out.len() / 2..];
        let max_abs = settled.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_abs < 0.05,
            "expected strong attenuation at Nyquist, got max_abs={max_abs}"
        );
    }

    #[test]
    fn chain_passes_low_frequency_content() {
        let taps: Vec<f64> = design_decimation_taps(0.9, usize::MAX);
        let mut chain = DecimationChain::new(1, &taps);

        let n = 4096;
        // Low-frequency tone, well inside the passband regardless of decimation.
        let freq_frac = 0.01; // fraction of sample rate
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq_frac * i as f64).sin())
            .collect();
        let mut out = Vec::new();
        chain.process(&input, &mut out);

        assert!(is_finite_and_bounded(&out, 1.2));
        let settled = &out[out.len() / 2..];
        let max_abs = settled.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_abs > 0.8,
            "expected low frequency content to pass through mostly intact, got {max_abs}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let taps: Vec<f64> = design_decimation_taps(0.9, usize::MAX);
        let mut chain = DecimationChain::new(2, &taps);
        let input = vec![1.0f64; 64];
        let mut out = Vec::new();
        chain.process(&input, &mut out);
        assert!(out.iter().any(|value| *value != 0.0));

        chain.reset();
        let mut out_after_reset = Vec::new();
        chain.process(&vec![0.0f64; 64], &mut out_after_reset);
        assert!(out_after_reset.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn zero_stage_chain_is_passthrough() {
        let mut chain: DecimationChain<f64> = DecimationChain::new(0, &[]);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = Vec::new();
        chain.process(&input, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn design_decimation_taps_respects_group_delay_cap() {
        // A tight bandwidth would normally ask for hundreds of taps; capping max_group_delay
        // should force a much shorter (but still valid, odd, unity-gain) filter.
        let uncapped: Vec<f64> = design_decimation_taps(0.99, usize::MAX);
        assert!(uncapped.len() > 100);

        for max_group_delay in [5usize, 20, 50] {
            let taps: Vec<f64> = design_decimation_taps(0.99, max_group_delay);
            let group_delay = (taps.len() - 1) / 2;
            assert!(
                group_delay <= max_group_delay.max((MIN_TAPS - 1) / 2),
                "group_delay={group_delay} should respect max_group_delay={max_group_delay}"
            );
            assert_eq!(taps.len() % 2, 1);
            let sum: f64 = taps.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn flush_recovers_trailing_real_samples_instead_of_dropping_them() {
        // Without a flush, the last `group_delay`-ish real samples fed into a single-stage
        // decimator never appear in `process`'s output at all -- they're still sitting in the
        // delay line. `flush` should recover them (as a hold-padded, but non-zero, tail).
        let taps: Vec<f64> = design_decimation_taps(0.9, usize::MAX);
        let mut chain = DecimationChain::new(1, &taps);

        let input = vec![1.0f64; 4096];
        let mut out = Vec::new();
        chain.process(&input, &mut out);
        assert!(
            out.iter().rev().take(10).all(|value| (value - 1.0).abs() < 1e-6),
            "steady-state output before flush should already reflect the constant input"
        );

        let mut flushed = Vec::new();
        chain.flush(&mut flushed);
        assert!(!flushed.is_empty());
        assert!(is_finite_and_bounded(&flushed, 1.2));
        // Flushing a constant-input filter with hold-padding of that same constant should
        // continue producing that same constant (no spurious energy from the flush itself).
        assert!(
            flushed.iter().all(|value| (value - 1.0).abs() < 1e-6),
            "flush of a constant signal should hold at that constant, got {flushed:?}"
        );
    }

    #[test]
    fn flush_of_empty_chain_is_a_noop() {
        let mut chain: DecimationChain<f64> = DecimationChain::new(0, &[]);
        let mut out = Vec::new();
        chain.flush(&mut out);
        assert!(out.is_empty());
    }
}
