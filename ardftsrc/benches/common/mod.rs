//! Shared bench input: deterministic stereo white noise of track length.
//!
//! Every throughput bench uses the same channel count and duration so rows are comparable with
//! each other. The duration matches a typical track because `process_all` has a fixed per-stream
//! cost (edge extrapolation, final flush) that a short input would overweight.

#![allow(dead_code)]

/// Channels in every bench input.
pub const CHANNELS: usize = 2;
/// Duration of every throughput bench input.
pub const INPUT_SECONDS: usize = 240;
/// Input rates, in the order the benches run and report them.
pub const INPUT_SAMPLE_RATES: &[usize] = &[44_100, 22_050, 96_000];
pub const OUTPUT_SAMPLE_RATES: &[usize] = &[22_050, 48_000, 96_000];

/// Deterministic interleaved white noise in `[-0.5, 0.5)`, `frames * CHANNELS` samples long.
pub fn noise(frames: usize) -> Vec<f64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..frames * CHANNELS)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// [`INPUT_SECONDS`] of [`noise`] at `sample_rate_hz`.
pub fn track(sample_rate_hz: usize) -> Vec<f64> {
    noise(sample_rate_hz * INPUT_SECONDS)
}
