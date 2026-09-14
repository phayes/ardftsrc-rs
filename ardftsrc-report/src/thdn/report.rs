//! Sweeps ardftsrc's `f64` resamplers across frequency, amplitude, sample-rate
//! pair, and preset, computing THD+N via [`crate::sine_fit`] and
//! [`crate::residual_spectrum`] for each combination.

use ardftsrc::{Config, InterleavedResampler};
use dasp_signal::Signal;
use serde::{Deserialize, Serialize};
use tabled::builder::Builder;
use tabled::settings::object::Columns;
use tabled::settings::{Alignment, Style};

use crate::preset::Preset;
use crate::thdn::{NOISE_FLOOR_DB, residual_spectrum, sine_fit};

/// Band used for the "audio-band" THD+N figure.
pub const AUDIO_BAND_HZ: (f64, f64) = (20.0, 20_000.0);

pub const DEFAULT_FREQUENCIES_HZ: &[f64] = &[20.0, 100.0, 1_000.0, 5_000.0, 10_000.0, 15_000.0, 18_000.0, 20_000.0];
pub const DEFAULT_AMPLITUDES_DBFS: &[f64] = &[-1.0, -20.0, -60.0];
// 192_000 -> 48_000 is a 4x downsampling ratio, the threshold at which Config::decimate
// starts engaging (see its docs) — included so the sweep actually exercises that path,
// while still keeping the 20 kHz test tone comfortably under its 24 kHz output Nyquist.
pub const DEFAULT_RATE_PAIRS: &[(usize, usize)] = &[
    (44_100, 48_000),
    (48_000, 44_100),
    (44_100, 96_000),
    (96_000, 44_100),
    (192_000, 48_000),
];
pub const DEFAULT_PRESETS: &[Preset] = &Preset::ALL;

#[cfg(feature = "f128")]
pub const F128_VARIANTS: &[bool] = &[false, true];
#[cfg(not(feature = "f128"))]
pub const F128_VARIANTS: &[bool] = &[false];

/// `Config::decimate`'s pre-decimation stage only ever engages when downsampling by at
/// least 4x (see its docs); below that it's a documented no-op. So rate pairs under that
/// ratio only need the `decimate=false` case, and only eligible pairs get both.
pub fn decimation_eligible(input_rate: usize, output_rate: usize) -> bool {
    input_rate > output_rate && input_rate >= output_rate * 4
}

fn decimate_variants(input_rate: usize, output_rate: usize) -> &'static [bool] {
    if decimation_eligible(input_rate, output_rate) {
        &[false, true]
    } else {
        &[false]
    }
}

/// Total number of cases [`run_sweep`] will run for the given axes — accounts for
/// [`decimate_variants`] varying per rate pair, so it isn't a flat product of axis lengths.
pub fn planned_case_count(
    rate_pairs: &[(usize, usize)],
    presets: &[Preset],
    frequencies_hz: &[f64],
    amplitudes_dbfs: &[f64],
) -> usize {
    let decimate_cases: usize = rate_pairs.iter().map(|&(i, o)| decimate_variants(i, o).len()).sum();
    decimate_cases * presets.len() * F128_VARIANTS.len() * frequencies_hz.len() * amplitudes_dbfs.len()
}

#[cfg(feature = "f128")]
fn config_with_f128(config: Config, f128: bool) -> Config {
    config.with_f128(f128)
}

#[cfg(not(feature = "f128"))]
fn config_with_f128(config: Config, f128: bool) -> Config {
    assert!(!f128, "f128 requested but this build has no `f128` feature");
    config
}

/// THD+N and related measurements for one (rate pair, decimate, preset, f128,
/// frequency, amplitude) combination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseResult {
    pub input_rate: usize,
    pub output_rate: usize,
    /// Whether `Config::decimate`'s pre-decimation stage was enabled. Always `false` for
    /// rate pairs below the 4x downsampling ratio where it would engage (see
    /// [`decimation_eligible`]).
    pub decimate: bool,
    pub preset: Preset,
    pub f128: bool,
    pub freq_hz: f64,
    pub amplitude_dbfs: f64,
    /// THD+N over the full DC-Nyquist bandwidth (time-domain sine fit, no FFT).
    pub thdn_broadband_db: f64,
    /// THD+N limited to [`AUDIO_BAND_HZ`].
    pub thdn_audio_band_db: f64,
    /// `20*log10(measured_gain / expected_gain)`. Nonzero near/above the transition
    /// band is expected attenuation, not distortion.
    pub gain_error_db: f64,
    pub max_spur_freq_hz: f64,
    pub max_spur_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub noise_floor_db: f64,
    pub audio_band_hz: (f64, f64),
    pub cases: Vec<CaseResult>,
}

/// Runs one configuration end to end: builds the resampler, generates a steady single
/// tone, resamples it, and fits/analyzes a steady-state window of the output.
///
/// The input length is sized dynamically from
/// [`InterleavedResampler::output_delay_frames`] so the analysis window always sits
/// well clear of startup/tail transients, regardless of preset.
pub fn run_case(
    input_rate: usize,
    output_rate: usize,
    preset: Preset,
    f128: bool,
    decimate: bool,
    freq_hz: f64,
    amplitude_dbfs: f64,
) -> CaseResult {
    let config = config_with_f128(preset.base_config(), f128)
        .with_input_rate(input_rate)
        .with_output_rate(output_rate)
        .with_channels(1)
        .with_decimate(decimate);
    let mut resampler = InterleavedResampler::<f64>::new(config).expect("thdn sweep configs are always valid");

    let amplitude = 10f64.powf(amplitude_dbfs / 20.0);

    // Skip well past both the resampler's algorithmic delay and any LPC-extrapolation
    // settling at the head; skip past the (much smaller) tail similarly. These are
    // deliberately generous multiples of output_delay_frames() since it's cheap to
    // generate a bit more input.
    let delay_out = resampler.output_delay_frames();
    let head_skip = (delay_out * 3).max((0.25 * output_rate as f64).round() as usize);
    let tail_skip = delay_out.max((0.1 * output_rate as f64).round() as usize);
    let min_analysis_frames = output_rate; // >= 1 second, for reasonable FFT bin resolution.

    let needed_output_frames = head_skip + tail_skip + min_analysis_frames;
    // input_rate extra frames of margin absorb rate-conversion rounding.
    let total_input_frames = needed_output_frames * input_rate / output_rate + input_rate;

    let input: Vec<f64> = dasp_signal::rate(input_rate as f64)
        .const_hz(freq_hz)
        .sine()
        .take(total_input_frames)
        .map(|s| s * amplitude)
        .collect();

    let output = resampler
        .process_all(&input)
        .expect("resampling a clean sine should never fail")
        .interleave();
    assert!(
        output.len() > head_skip + tail_skip,
        "resampled output ({} frames) too short to skip {head_skip} head + {tail_skip} tail",
        output.len()
    );

    let steady_state = &output[head_skip..output.len() - tail_skip];

    let fit = sine_fit(steady_state, output_rate as f64, freq_hz);
    let spectrum = residual_spectrum(&fit.residual, fit.fundamental_rms, output_rate as f64, AUDIO_BAND_HZ);

    CaseResult {
        input_rate,
        output_rate,
        decimate,
        preset,
        f128,
        freq_hz,
        amplitude_dbfs,
        thdn_broadband_db: fit.thdn_db,
        thdn_audio_band_db: spectrum.band_thdn_db,
        gain_error_db: 20.0 * (fit.gain / amplitude).log10(),
        max_spur_freq_hz: spectrum.max_spur.freq_hz,
        max_spur_db: spectrum.max_spur.level_db,
    }
}

/// Runs the full sweep, calling `on_case` as each case completes (useful for progress
/// reporting on what is a bench-scale amount of work).
pub fn run_sweep<F: FnMut(&CaseResult)>(
    rate_pairs: &[(usize, usize)],
    presets: &[Preset],
    frequencies_hz: &[f64],
    amplitudes_dbfs: &[f64],
    mut on_case: F,
) -> Report {
    let mut cases = Vec::new();
    for &(input_rate, output_rate) in rate_pairs {
        for &decimate in decimate_variants(input_rate, output_rate) {
            for &preset in presets {
                for &f128 in F128_VARIANTS {
                    for &freq_hz in frequencies_hz {
                        for &amplitude_dbfs in amplitudes_dbfs {
                            let case = run_case(
                                input_rate,
                                output_rate,
                                preset,
                                f128,
                                decimate,
                                freq_hz,
                                amplitude_dbfs,
                            );
                            on_case(&case);
                            cases.push(case);
                        }
                    }
                }
            }
        }
    }
    Report {
        noise_floor_db: NOISE_FLOOR_DB,
        audio_band_hz: AUDIO_BAND_HZ,
        cases,
    }
}

fn max_abs_signed(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |worst, v| if v.abs() > worst.abs() { v } else { worst })
}

fn worst_db(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(f64::NEG_INFINITY, f64::max)
}

/// Gain error at or below this is treated as exactly zero for display purposes (e.g.
/// `-2.78e-13` from float round-off in a passthrough/no-op case).
const GAIN_ERROR_ZERO_EPSILON_DB: f64 = 1e-9;

/// Footnote marker used wherever a gain error is snapped to zero (see
/// [`GAIN_ERROR_ZERO_EPSILON_DB`]); the report body explains it once, up top.
const GAIN_ERROR_ZERO_FOOTNOTE: &str = "0 †";

/// Formats a gain error as plain fixed-point when that's precise enough to be legible,
/// switching to scientific notation only for small-but-real values that `.2` fixed-point
/// would otherwise flatten to a misleading "0.00" (below 0.005 in magnitude), and to
/// [`GAIN_ERROR_ZERO_FOOTNOTE`] when the value is within [`GAIN_ERROR_ZERO_EPSILON_DB`]
/// of zero (float round-off from a linear, distortion-free resampler).
fn format_gain_error_db(db: f64) -> String {
    if db.abs() < GAIN_ERROR_ZERO_EPSILON_DB {
        GAIN_ERROR_ZERO_FOOTNOTE.to_string()
    } else if db.abs() < 0.005 {
        format!("{db:.2e}")
    } else {
        format!("{db:.2}")
    }
}

/// Renders `rows` as a Markdown pipe table with the given column headings/alignments.
pub(crate) fn render_table(headings: &[(&str, Alignment)], rows: Vec<Vec<String>>) -> String {
    let mut builder = Builder::new();
    builder.push_record(headings.iter().map(|(label, _)| label.to_string()));
    for row in rows {
        builder.push_record(row);
    }
    let mut table = builder.build();
    table.with(Style::markdown());
    for (i, &(_, align)) in headings.iter().enumerate() {
        table.modify(Columns::one(i), align);
    }
    table.to_string()
}

impl Report {
    /// Returns a copy of this report containing only cases for `preset`.
    pub fn for_preset(&self, preset: Preset) -> Report {
        Report {
            noise_floor_db: self.noise_floor_db,
            audio_band_hz: self.audio_band_hz,
            cases: self.cases.iter().filter(|c| c.preset == preset).cloned().collect(),
        }
    }

    /// Renders a human-readable Markdown *section* (no document title, and no preset
    /// configuration table -- both are the embedding caller's responsibility, since this
    /// is only ever embedded under a preset-scoped report's own title): a worst-case
    /// summary table followed by one detailed table per (rate pair, preset, f128)
    /// configuration. Headings start at `###` on the assumption the caller has already
    /// printed an enclosing `##` heading for this section.
    pub fn to_markdown(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let _ = writeln!(
            out,
            "`f64` only. Broadband THD+N covers DC-Nyquist (time-domain sine fit, no FFT); \
             audio-band THD+N is limited to {:.0} Hz-{:.0} Hz. Values at or below the {:.0} dB \
             noise floor reflect the plain-`f64` analyzer's own precision ceiling.",
            self.audio_band_hz.0, self.audio_band_hz.1, self.noise_floor_db
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "† Gain error is below {GAIN_ERROR_ZERO_EPSILON_DB:.0e} dB. It is below f64 round-off error."
        );
        let _ = writeln!(out);

        // Group consecutive cases by (rate pair, decimate, preset, f128); run_sweep
        // always emits cases in exactly this grouping, so a simple key-change scan
        // reconstructs it.
        let mut groups: Vec<(usize, usize, bool, Preset, bool)> = Vec::new();
        for c in &self.cases {
            let key = (c.input_rate, c.output_rate, c.decimate, c.preset, c.f128);
            if groups.last() != Some(&key) {
                groups.push(key);
            }
        }

        let _ = writeln!(out, "### Summary (worst case per configuration)");
        let _ = writeln!(out);
        // Sorted (stably) by preset so every preset's rows sit together, grouped across
        // rate pairs/decimate/f128 rather than the other way around.
        let mut by_preset = groups.clone();
        by_preset.sort_by_key(|&(_, _, _, preset, _)| preset);
        let summary_rows: Vec<Vec<String>> = by_preset
            .iter()
            .map(|&(input_rate, output_rate, decimate, preset, f128)| {
                let group: Vec<&CaseResult> = self
                    .cases
                    .iter()
                    .filter(|c| {
                        c.input_rate == input_rate
                            && c.output_rate == output_rate
                            && c.decimate == decimate
                            && c.preset == preset
                            && c.f128 == f128
                    })
                    .collect();
                let worst_broadband = worst_db(group.iter().map(|c| c.thdn_broadband_db));
                let worst_audio_band = worst_db(group.iter().map(|c| c.thdn_audio_band_db));
                let worst_gain_error = max_abs_signed(group.iter().map(|c| c.gain_error_db));
                let worst_spur = worst_db(group.iter().map(|c| c.max_spur_db));
                vec![
                    format!("{input_rate} -> {output_rate}"),
                    decimate.to_string(),
                    f128.to_string(),
                    format!("{worst_broadband:.2}"),
                    format!("{worst_audio_band:.2}"),
                    format_gain_error_db(worst_gain_error),
                    format!("{worst_spur:.2}"),
                ]
            })
            .collect();
        let _ = writeln!(
            out,
            "{}",
            render_table(
                &[
                    ("Rate pair", Alignment::left()),
                    ("Decimate", Alignment::left()),
                    ("f128", Alignment::left()),
                    ("Worst broadband THD+N (dB)", Alignment::right()),
                    ("Worst audio-band THD+N (dB)", Alignment::right()),
                    ("Worst gain error (dB)", Alignment::right()),
                    ("Worst spur (dB)", Alignment::right()),
                ],
                summary_rows
            )
        );

        let detail_headings = [
            ("Freq (Hz)", Alignment::right()),
            ("Amplitude (dBFS)", Alignment::right()),
            ("Broadband THD+N (dB)", Alignment::right()),
            ("Audio-band THD+N (dB)", Alignment::right()),
            ("Gain error (dB)", Alignment::right()),
            ("Max spur (dB @ Hz)", Alignment::right()),
        ];
        for &(input_rate, output_rate, decimate, preset, f128) in &groups {
            let _ = writeln!(
                out,
                "### {input_rate} -> {output_rate}, decimate={decimate}, f128={f128}"
            );
            let _ = writeln!(out);
            let detail_rows: Vec<Vec<String>> = self
                .cases
                .iter()
                .filter(|c| {
                    c.input_rate == input_rate
                        && c.output_rate == output_rate
                        && c.decimate == decimate
                        && c.preset == preset
                        && c.f128 == f128
                })
                .map(|c| {
                    vec![
                        format!("{:.0}", c.freq_hz),
                        format!("{:.0}", c.amplitude_dbfs),
                        format!("{:.2}", c.thdn_broadband_db),
                        format!("{:.2}", c.thdn_audio_band_db),
                        format_gain_error_db(c.gain_error_db),
                        format!("{:.2} @ {:.0}", c.max_spur_db, c.max_spur_freq_hz),
                    ]
                })
                .collect();
            let _ = writeln!(out, "{}", render_table(&detail_headings, detail_rows));
        }

        out
    }
}
