//! Pre-ringing analysis for ardftsrc's `f64` resamplers.
//!
//! Two measurements per (rate pair, decimate, preset) configuration:
//!
//! - **Impulse pre-ringing** ([`run_impulse`]): a unit impulse is resampled and the response is
//!   scanned backward from the impulse's ideal output time to find how long before it the
//!   response first exceeds each of [`THRESHOLDS_DB`], plus the dominant frequency of that
//!   pre-ringing tail. A full-band impulse excites the whole transition band, so this measures
//!   how long and at what frequency the filter rings.
//! - **Below-rolloff leakage** ([`run_transient`]): a Gaussian click or Gaussian-windowed tone
//!   burst whose spectrum is below [`TRANSIENT_SPECTRAL_FLOOR_DB`] from the passband edge
//!   upward is resampled and compared sample-by-sample against its analytically known ideal
//!   output. The passband gain is exactly unity, so an ideal linear-phase resampler returns
//!   such a transient unchanged: whatever residual remains before the transient is pre-ringing
//!   that reached frequencies below the rolloff. A best-fit constant delay is removed first
//!   (and reported) so a fractional-sample timing offset isn't mistaken for ringing.
//!
//! Each measurement is repeated with the stimulus at several positions within the
//! resampler's processing chunk ([`CHUNK_POSITIONS`]), since block processing could make the
//! response position-dependent; the report shows the worst case.

use ardftsrc::InterleavedResampler;
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use tabled::settings::Alignment;

use crate::preset::Preset;
use crate::thdn::NOISE_FLOOR_DB;
use crate::thdn::report::{DEFAULT_RATE_PAIRS, decimate_variants, render_table};

pub use crate::thdn::report::DEFAULT_PRESETS;

/// Pre-ringing durations are reported at each of these levels, in dB relative to the
/// impulse response's peak.
pub const THRESHOLDS_DB: &[f64] = &[-60.0, -100.0, -140.0];

/// Stimulus positions within the resampler's input chunk, as a fraction of the chunk length.
/// `0.0` (the chunk's first sample) leaves the least room for pre-ringing inside the FFT
/// window, so it is the position most likely to expose block-processing effects.
pub const CHUNK_POSITIONS: &[f64] = &[0.0, 0.25, 0.5, 0.75];

/// Carrier frequencies of the below-rolloff transients, as fractions of the passband edge.
/// `0.0` is a plain Gaussian click: the part of an impulse that lies below the rolloff.
pub const TRANSIENT_CARRIERS: &[f64] = &[0.0, 0.5, 0.85];

/// Each transient's spectrum is at most this far below its peak at the passband edge, so
/// none of its energy reaches the transition band at a level that could matter.
pub const TRANSIENT_SPECTRAL_FLOOR_DB: f64 = -300.0;

/// Fraction of the nominal passband edge (`bandwidth` x lower Nyquist) treated as the edge
/// when sizing transients, absorbing the up-to-one-bin rounding of the real FFT-bin edge.
const PASSBAND_EDGE_MARGIN: f64 = 0.98;

/// Sub-sample offset of each transient's center from the input sample grid, so the ideal
/// output is not trivially sampled at the transient's peak.
const TRANSIENT_CENTER_OFFSET: f64 = 0.37;

/// Search range, in output samples, for the best-fit delay between a transient's output and
/// its ideal.
const DELAY_SEARCH_SAMPLES: f64 = 2.0;

/// The configuration's filter geometry in Hz.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Band {
    /// Nominal passband edge: `bandwidth` x the lower of the two Nyquist frequencies. The
    /// filter's gain is exactly unity below this.
    pub passband_edge_hz: f64,
    /// The lower of the two Nyquist frequencies, where the transition band ends.
    pub stopband_edge_hz: f64,
}

impl Band {
    pub fn new(preset: Preset, input_rate: usize, output_rate: usize) -> Self {
        let stopband_edge_hz = input_rate.min(output_rate) as f64 / 2.0;
        Self {
            passband_edge_hz: f64::from(preset.base_config().bandwidth) * stopband_edge_hz,
            stopband_edge_hz,
        }
    }
}

/// Impulse pre-ringing measured at one position within the input chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImpulseResult {
    /// Position within the resampler's input chunk, as a fraction of its length.
    pub chunk_position: f64,
    /// Largest absolute output sample, the reference for [`THRESHOLDS_DB`].
    pub peak: f64,
    /// For each of [`THRESHOLDS_DB`], how long before the impulse's ideal output time the
    /// response first reaches that level, in ms. `0.0` when no earlier sample does.
    pub pre_ring_ms: Vec<f64>,
    /// Frequency of the largest spectral peak of the pre-ringing tail (main lobe excluded).
    pub ringing_freq_hz: f64,
}

/// One below-rolloff transient measured at one position within the input chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientResult {
    /// Position within the resampler's input chunk, as a fraction of its length.
    pub chunk_position: f64,
    /// Carrier frequency; `0.0` for a plain Gaussian click.
    pub carrier_hz: f64,
    /// Standard deviation of the Gaussian envelope, in ms.
    pub sigma_ms: f64,
    /// Best-fit constant delay of the output relative to the ideal, in output samples.
    /// Positive means the output is late.
    pub delay_offset_samples: f64,
    /// Largest residual before the transient's center, in dB relative to its peak.
    /// Clamped at [`NOISE_FLOOR_DB`].
    pub pre_echo_db: f64,
    /// How far before the transient's center [`pre_echo_db`](Self::pre_echo_db) occurs, in ms.
    pub pre_echo_lead_ms: f64,
    /// Largest residual after the transient's center, in dB relative to its peak.
    /// Clamped at [`NOISE_FLOOR_DB`].
    pub post_echo_db: f64,
    /// How far after the transient's center [`post_echo_db`](Self::post_echo_db) occurs, in ms.
    pub post_echo_lag_ms: f64,
}

/// All pre-ringing measurements for one (rate pair, decimate, preset) configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigResult {
    pub input_rate: usize,
    pub output_rate: usize,
    /// Whether `Config::decimate`'s pre-decimation stage was enabled.
    pub decimate: bool,
    pub preset: Preset,
    pub band: Band,
    pub impulses: Vec<ImpulseResult>,
    pub transients: Vec<TransientResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub noise_floor_db: f64,
    pub thresholds_db: Vec<f64>,
    pub results: Vec<ConfigResult>,
}

fn db(ratio: f64) -> f64 {
    if ratio <= 0.0 {
        return NOISE_FLOOR_DB;
    }
    (20.0 * ratio.log10()).max(NOISE_FLOOR_DB)
}

/// A mono resampler for one configuration, plus its raw input chunk length.
fn build_resampler(
    preset: Preset,
    input_rate: usize,
    output_rate: usize,
    decimate: bool,
) -> (InterleavedResampler<f64>, usize) {
    let config = preset
        .base_config()
        .with_input_rate(input_rate)
        .with_output_rate(output_rate)
        .with_channels(1)
        .with_decimate(decimate);
    let resampler = InterleavedResampler::<f64>::new(config).expect("pre-ringing configs are always valid");
    let chunk = resampler.input_buffer_size();
    (resampler, chunk)
}

/// Input layout shared by both measurements: the stimulus sits at input sample
/// `stimulus_index`, at `chunk_position` within its chunk, with at least two chunks and half
/// a second of silence on either side so the full ringing tail is captured.
struct Layout {
    stimulus_index: usize,
    total_frames: usize,
}

impl Layout {
    fn new(chunk: usize, input_rate: usize, chunk_position: f64) -> Self {
        let margin_chunks = 2.max((input_rate / 2).div_ceil(chunk));
        let offset = (chunk_position * chunk as f64).floor() as usize;
        let stimulus_index = margin_chunks * chunk + offset;
        Self {
            stimulus_index,
            total_frames: stimulus_index + (margin_chunks + 1) * chunk,
        }
    }
}

/// Resamples a unit impulse at `chunk_position` and measures its pre-ringing.
pub fn run_impulse(
    preset: Preset,
    input_rate: usize,
    output_rate: usize,
    decimate: bool,
    chunk_position: f64,
) -> ImpulseResult {
    let band = Band::new(preset, input_rate, output_rate);
    let (mut resampler, chunk) = build_resampler(preset, input_rate, output_rate, decimate);
    let layout = Layout::new(chunk, input_rate, chunk_position);

    let mut input = vec![0.0; layout.total_frames];
    input[layout.stimulus_index] = 1.0;
    let output = resampler
        .process_all(&input)
        .expect("resampling an impulse should never fail")
        .interleave();

    let center = layout.stimulus_index as f64 * output_rate as f64 / input_rate as f64;
    let peak = output.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let ms_per_sample = 1000.0 / output_rate as f64;

    // Earliest sample at or above each threshold. Scanning forward from the start, the first
    // hit is the furthest-ahead one.
    let lead_at = |threshold_db: f64| -> f64 {
        let level = peak * 10f64.powf(threshold_db / 20.0);
        output
            .iter()
            .enumerate()
            .take_while(|&(j, _)| (j as f64) < center)
            .find(|(_, v)| v.abs() >= level)
            .map_or(0.0, |(j, _)| center - j as f64)
    };
    let pre_ring_ms = THRESHOLDS_DB.iter().map(|&t| lead_at(t) * ms_per_sample).collect();

    // Excludes roughly the first two cycles at the passband edge (the band-limited impulse's
    // own main lobe), so the spectrum reflects the ringing tail rather than the pulse shape.
    let guard = (2.0 * output_rate as f64 / band.passband_edge_hz).ceil() + 1.0;
    let deepest = THRESHOLDS_DB.iter().copied().fold(f64::INFINITY, f64::min);
    let tail = lead_at(deepest).max(guard * 8.0);
    let start = (center - tail).floor().max(0.0) as usize;
    let end = (center - guard).floor().max(0.0) as usize;
    let ringing_freq_hz = dominant_frequency(&output[start..end.max(start)], output_rate as f64);

    ImpulseResult {
        chunk_position,
        peak,
        pre_ring_ms,
        ringing_freq_hz,
    }
}

/// Frequency of the largest peak in the Hann-windowed spectrum of `segment`, refined by
/// parabolic interpolation on log magnitude. `0.0` for segments too short to analyze.
fn dominant_frequency(segment: &[f64], sample_rate: f64) -> f64 {
    let n = segment.len();
    if n < 4 {
        return 0.0;
    }
    let nfft = (n * 8).next_power_of_two().max(4096);
    let mut buf = vec![0.0; nfft];
    for (i, (dst, &v)) in buf.iter_mut().zip(segment).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
        *dst = v * w;
    }
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut buf, &mut spectrum)
        .expect("fixed-size real FFT process should not fail");

    let power: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();
    let k = power
        .iter()
        .enumerate()
        .skip(1)
        .fold((0, 0.0), |best, (i, &p)| if p > best.1 { (i, p) } else { best })
        .0;
    let offset = if k > 0 && k + 1 < power.len() && power[k - 1] > 0.0 && power[k + 1] > 0.0 {
        let (a, b, c) = (power[k - 1].ln(), power[k].ln(), power[k + 1].ln());
        let denom = a - 2.0 * b + c;
        if denom != 0.0 { 0.5 * (a - c) / denom } else { 0.0 }
    } else {
        0.0
    };
    (k as f64 + offset) * sample_rate / nfft as f64
}

/// A Gaussian click (`carrier_hz == 0`) or Gaussian-windowed cosine burst with unit peak,
/// centered at `t = 0`.
#[derive(Debug, Clone, Copy)]
struct Transient {
    carrier_hz: f64,
    sigma_s: f64,
}

impl Transient {
    /// Sizes the envelope so the spectrum is [`TRANSIENT_SPECTRAL_FLOOR_DB`] down at
    /// `edge_hz`: a Gaussian envelope with time deviation `sigma` has spectral deviation
    /// `1 / (2 pi sigma)`.
    fn below(edge_hz: f64, carrier_hz: f64) -> Self {
        let spread = (-2.0 * (TRANSIENT_SPECTRAL_FLOOR_DB / 20.0) * std::f64::consts::LN_10).sqrt();
        let distance_hz = edge_hz - carrier_hz;
        assert!(distance_hz > 0.0, "transient carrier must be below the passband edge");
        Self {
            carrier_hz,
            sigma_s: spread / (2.0 * std::f64::consts::PI * distance_hz),
        }
    }

    fn at(&self, t: f64) -> f64 {
        let envelope = (-0.5 * (t / self.sigma_s).powi(2)).exp();
        if self.carrier_hz == 0.0 {
            envelope
        } else {
            envelope * (2.0 * std::f64::consts::PI * self.carrier_hz * t).cos()
        }
    }
}

/// Exact time, in seconds, of `sample` at `rate` relative to the instant
/// `center_index + TRANSIENT_CENTER_OFFSET` at `center_rate`. The integer part is computed
/// exactly so that precision does not degrade with absolute stream position.
fn relative_time(sample: usize, rate: usize, center_index: usize, center_rate: usize) -> f64 {
    let numerator = sample as i128 * center_rate as i128 - center_index as i128 * rate as i128;
    numerator as f64 / (rate as f64 * center_rate as f64) - TRANSIENT_CENTER_OFFSET / center_rate as f64
}

/// Sum of squared residuals against `transient` delayed by `delay` output samples, over
/// `window` (output indices).
fn delayed_error(
    output: &[f64],
    window: std::ops::Range<usize>,
    times: &[f64],
    transient: &Transient,
    delay_s: f64,
) -> f64 {
    window
        .zip(times)
        .map(|(j, &t)| {
            let e = output[j] - transient.at(t - delay_s);
            e * e
        })
        .sum()
}

/// Least-squares constant delay (in output samples) between `output` and `transient`, over
/// `±DELAY_SEARCH_SAMPLES`: a 0.01-sample grid scan, then golden-section refinement.
fn fit_delay(
    output: &[f64],
    output_rate: usize,
    center: f64,
    transient: &Transient,
    times: &dyn Fn(usize) -> f64,
) -> f64 {
    let half_width = (12.0 * transient.sigma_s * output_rate as f64).ceil() + DELAY_SEARCH_SAMPLES + 2.0;
    let lo = (center - half_width).floor().max(0.0) as usize;
    let hi = ((center + half_width).ceil() as usize).min(output.len());
    let window_times: Vec<f64> = (lo..hi).map(times).collect();
    let error = |delay_samples: f64| {
        delayed_error(
            output,
            lo..hi,
            &window_times,
            transient,
            delay_samples / output_rate as f64,
        )
    };

    let steps = (DELAY_SEARCH_SAMPLES * 100.0) as i32;
    let mut best = (0.0, f64::INFINITY);
    for step in -steps..=steps {
        let d = step as f64 / 100.0;
        let e = error(d);
        if e < best.1 {
            best = (d, e);
        }
    }

    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (best.0 - 0.01, best.0 + 0.01);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut fc, mut fd) = (error(c), error(d));
    while b - a > 1e-12 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = error(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = error(d);
        }
    }
    (a + b) / 2.0
}

/// Resamples one below-rolloff transient per entry of `carriers_hz` at `chunk_position`, and
/// measures each one's residual against its ideal. The delay is fitted on the first entry
/// and reused for the rest, since it is a property of the configuration rather than the
/// stimulus; pass the click (`0.0`) first, as it has no carrier-period ambiguity.
pub fn run_transients(
    preset: Preset,
    input_rate: usize,
    output_rate: usize,
    decimate: bool,
    chunk_position: f64,
    carriers_hz: &[f64],
) -> Vec<TransientResult> {
    let band = Band::new(preset, input_rate, output_rate);
    let edge_hz = band.passband_edge_hz * PASSBAND_EDGE_MARGIN;
    let mut delay = None;

    carriers_hz
        .iter()
        .map(|&carrier_hz| {
            let transient = Transient::below(edge_hz, carrier_hz);
            let (mut resampler, chunk) = build_resampler(preset, input_rate, output_rate, decimate);
            let layout = Layout::new(chunk, input_rate, chunk_position);
            let center_index = layout.stimulus_index;

            let input: Vec<f64> = (0..layout.total_frames)
                .map(|n| transient.at(relative_time(n, input_rate, center_index, input_rate)))
                .collect();
            let output = resampler
                .process_all(&input)
                .expect("resampling a transient should never fail")
                .interleave();

            let times = |j: usize| relative_time(j, output_rate, center_index, input_rate);
            let center = (center_index as f64 + TRANSIENT_CENTER_OFFSET) * output_rate as f64 / input_rate as f64;
            let delay_samples =
                *delay.get_or_insert_with(|| fit_delay(&output, output_rate, center, &transient, &times));
            let delay_s = delay_samples / output_rate as f64;

            let mut pre = (0.0_f64, 0.0);
            let mut post = (0.0_f64, 0.0);
            for (j, &y) in output.iter().enumerate() {
                let t = times(j) - delay_s;
                let residual = (y - transient.at(t)).abs();
                let slot = if t < 0.0 { &mut pre } else { &mut post };
                if residual > slot.0 {
                    *slot = (residual, t.abs());
                }
            }

            TransientResult {
                chunk_position,
                carrier_hz,
                sigma_ms: transient.sigma_s * 1000.0,
                delay_offset_samples: delay_samples,
                pre_echo_db: db(pre.0),
                pre_echo_lead_ms: pre.1 * 1000.0,
                post_echo_db: db(post.0),
                post_echo_lag_ms: post.1 * 1000.0,
            }
        })
        .collect()
}

/// Runs every pre-ringing measurement for one configuration.
pub fn run_config(preset: Preset, input_rate: usize, output_rate: usize, decimate: bool) -> ConfigResult {
    let band = Band::new(preset, input_rate, output_rate);
    let carriers_hz: Vec<f64> = TRANSIENT_CARRIERS
        .iter()
        .map(|&fraction| fraction * band.passband_edge_hz * PASSBAND_EDGE_MARGIN)
        .collect();

    let impulses = CHUNK_POSITIONS
        .iter()
        .map(|&position| run_impulse(preset, input_rate, output_rate, decimate, position))
        .collect();
    let transients = CHUNK_POSITIONS
        .iter()
        .flat_map(|&position| run_transients(preset, input_rate, output_rate, decimate, position, &carriers_hz))
        .collect();

    ConfigResult {
        input_rate,
        output_rate,
        decimate,
        preset,
        band,
        impulses,
        transients,
    }
}

/// Number of configurations [`run_sweep`] will run for the given axes.
pub fn planned_config_count(rate_pairs: &[(usize, usize)], presets: &[Preset]) -> usize {
    let decimate_cases: usize = rate_pairs.iter().map(|&(i, o)| decimate_variants(i, o).len()).sum();
    decimate_cases * presets.len()
}

/// Runs [`run_config`] over every rate pair (and decimate variant) and preset, calling
/// `on_config` as each completes.
pub fn run_sweep<F: FnMut(&ConfigResult)>(
    rate_pairs: &[(usize, usize)],
    presets: &[Preset],
    mut on_config: F,
) -> Report {
    let mut results = Vec::new();
    for &(input_rate, output_rate) in rate_pairs {
        for &decimate in decimate_variants(input_rate, output_rate) {
            for &preset in presets {
                let result = run_config(preset, input_rate, output_rate, decimate);
                on_config(&result);
                results.push(result);
            }
        }
    }
    Report {
        noise_floor_db: NOISE_FLOOR_DB,
        thresholds_db: THRESHOLDS_DB.to_vec(),
        results,
    }
}

/// [`run_sweep`] over [`DEFAULT_RATE_PAIRS`] and [`DEFAULT_PRESETS`].
pub fn run_default_sweep<F: FnMut(&ConfigResult)>(on_config: F) -> Report {
    run_sweep(DEFAULT_RATE_PAIRS, DEFAULT_PRESETS, on_config)
}

impl ConfigResult {
    /// The impulse position with the longest pre-ringing at the deepest threshold.
    pub fn worst_impulse(&self) -> Option<&ImpulseResult> {
        self.impulses.iter().max_by(|a, b| {
            let key = |r: &ImpulseResult| r.pre_ring_ms.last().copied().unwrap_or(0.0);
            key(a).total_cmp(&key(b))
        })
    }

    /// For one carrier, the chunk position with the largest pre-echo.
    pub fn worst_transient(&self, carrier_hz: f64) -> Option<&TransientResult> {
        self.transients
            .iter()
            .filter(|t| t.carrier_hz == carrier_hz)
            .max_by(|a, b| a.pre_echo_db.total_cmp(&b.pre_echo_db))
    }

    /// Distinct carrier frequencies, in measurement order.
    pub fn carriers_hz(&self) -> Vec<f64> {
        let mut carriers: Vec<f64> = Vec::new();
        for t in &self.transients {
            if !carriers.contains(&t.carrier_hz) {
                carriers.push(t.carrier_hz);
            }
        }
        carriers
    }

    fn rate_pair_label(&self) -> String {
        format!("{} -> {}", self.input_rate, self.output_rate)
    }
}

fn transient_label(carrier_hz: f64) -> String {
    if carrier_hz == 0.0 {
        "Click".to_string()
    } else {
        format!("{carrier_hz:.0} Hz burst")
    }
}

fn format_echo(db: f64, ms: f64) -> String {
    format!("{db:.1} @ {ms:.3}")
}

impl Report {
    /// Returns a copy of this report containing only results for `preset`.
    pub fn for_preset(&self, preset: Preset) -> Report {
        Report {
            noise_floor_db: self.noise_floor_db,
            thresholds_db: self.thresholds_db.clone(),
            results: self.results.iter().filter(|r| r.preset == preset).cloned().collect(),
        }
    }

    /// Renders a Markdown *section* body (no `##` heading of its own; subsections start at
    /// `###`), for embedding under a preset-scoped report's own title.
    pub fn to_markdown(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let _ = writeln!(
            out,
            "`f64` only, no high-precision backend. Every measurement is repeated with the \
             stimulus at {} positions within the resampler's processing chunk; tables show the \
             worst position.",
            CHUNK_POSITIONS.len()
        );
        let _ = writeln!(out);

        let _ = writeln!(out, "### Impulse pre-ringing");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "A unit impulse is resampled, and the output is scanned backward from the impulse's \
             ideal output time. Each \"Pre-ring\" column is how long before that time the response \
             first reaches the given level, relative to the response's peak (worst position per \
             column). \"Ringing freq\" is \
             the strongest frequency in the pre-ringing tail, excluding the main lobe. The \
             transition band runs from the passband edge (unity gain below it) to the lower \
             Nyquist frequency."
        );
        let _ = writeln!(out);

        let mut headings = vec![
            ("Rate pair".to_string(), Alignment::left()),
            ("Decimate".to_string(), Alignment::left()),
            ("Transition band (Hz)".to_string(), Alignment::right()),
        ];
        for t in &self.thresholds_db {
            headings.push((format!("Pre-ring ≥ {t:.0} dB (ms)"), Alignment::right()));
        }
        headings.push(("Ringing freq (Hz)".to_string(), Alignment::right()));
        let impulse_rows: Vec<Vec<String>> = self
            .results
            .iter()
            .filter_map(|r| {
                let worst = r.worst_impulse()?;
                let mut row = vec![
                    r.rate_pair_label(),
                    r.decimate.to_string(),
                    format!("{:.0} - {:.0}", r.band.passband_edge_hz, r.band.stopband_edge_hz),
                ];
                // Worst case per threshold, which can come from different positions.
                for i in 0..self.thresholds_db.len() {
                    let ms = r.impulses.iter().map(|imp| imp.pre_ring_ms[i]).fold(0.0, f64::max);
                    row.push(format!("{ms:.2}"));
                }
                row.push(format!("{:.0}", worst.ringing_freq_hz));
                Some(row)
            })
            .collect();
        let heading_refs: Vec<(&str, Alignment)> = headings.iter().map(|(s, a)| (s.as_str(), *a)).collect();
        let _ = writeln!(out, "{}", render_table(&heading_refs, impulse_rows));
        let _ = writeln!(out);

        let _ = writeln!(out, "### Pre-ringing below the rolloff");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "Gaussian transients whose spectra are at least {:.0} dB down at the passband edge \
             (a click, the part of an impulse below the rolloff, plus tone bursts) are resampled and \
             compared sample-by-sample against their exact ideal output. The passband gain is \
             unity, so an ideal linear-phase resampler would return them unchanged: any residual \
             before the transient is pre-ringing that reached frequencies below the rolloff. A \
             best-fit constant delay is removed first, so a timing offset isn't counted as \
             ringing. Echo columns are the \
             largest residual in dB relative to the transient's peak, @ ms before (pre) or after \
             (post) its center; a lead well beyond the envelope σ is a true pre-echo, ahead of the \
             transient's own rise. Values near the THD+N section's noise floor are `f64` rounding, \
             not leakage.",
            TRANSIENT_SPECTRAL_FLOOR_DB
        );
        let _ = writeln!(out);

        if let Some((worst_result, worst)) = self
            .results
            .iter()
            .flat_map(|r| r.transients.iter().map(move |t| (r, t)))
            .max_by(|a, b| a.1.pre_echo_db.total_cmp(&b.1.pre_echo_db))
        {
            let _ = writeln!(
                out,
                "Worst below-rolloff pre-echo: **{:.1} dB**, {:.3} ms ahead of the {} ({}, decimate={}).",
                worst.pre_echo_db,
                worst.pre_echo_lead_ms,
                if worst.carrier_hz == 0.0 {
                    "click".to_string()
                } else {
                    transient_label(worst.carrier_hz)
                },
                worst_result.rate_pair_label(),
                worst_result.decimate
            );
            let _ = writeln!(out);
        }

        let transient_rows: Vec<Vec<String>> = self
            .results
            .iter()
            .flat_map(|r| {
                r.carriers_hz().into_iter().filter_map(move |carrier_hz| {
                    let worst = r.worst_transient(carrier_hz)?;
                    let worst_post = r
                        .transients
                        .iter()
                        .filter(|t| t.carrier_hz == carrier_hz)
                        .max_by(|a, b| a.post_echo_db.total_cmp(&b.post_echo_db))?;
                    Some(vec![
                        r.rate_pair_label(),
                        r.decimate.to_string(),
                        transient_label(carrier_hz),
                        format!("{:.3}", worst.sigma_ms),
                        format_echo(worst.pre_echo_db, worst.pre_echo_lead_ms),
                        format_echo(worst_post.post_echo_db, worst_post.post_echo_lag_ms),
                    ])
                })
            })
            .collect();
        let _ = writeln!(
            out,
            "{}",
            render_table(
                &[
                    ("Rate pair", Alignment::left()),
                    ("Decimate", Alignment::left()),
                    ("Transient", Alignment::left()),
                    ("Envelope σ (ms)", Alignment::right()),
                    ("Worst pre-echo (dB @ ms)", Alignment::right()),
                    ("Worst post-echo (dB @ ms)", Alignment::right()),
                ],
                transient_rows
            )
        );

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn transient_spectrum_is_at_the_floor_at_the_edge() {
        let edge_hz = 20_000.0;
        for carrier_hz in [0.0, 10_000.0, 17_000.0] {
            let transient = Transient::below(edge_hz, carrier_hz);
            // Gaussian envelope spectrum relative to its peak, evaluated at the edge.
            let sigma_f = 1.0 / (2.0 * PI * transient.sigma_s);
            let level_db = 20.0 * (-0.5 * ((edge_hz - carrier_hz) / sigma_f).powi(2)).exp().log10();
            assert!(
                (level_db - TRANSIENT_SPECTRAL_FLOOR_DB).abs() < 1e-6,
                "carrier={carrier_hz} level_db={level_db}"
            );
        }
    }

    #[test]
    fn relative_time_is_exact_far_into_a_stream() {
        // 44.1k -> 48k: output sample 1_280_000 lands exactly on input sample 1_176_000.
        let t = relative_time(1_280_000, 48_000, 1_176_000, 44_100);
        let expected = -TRANSIENT_CENTER_OFFSET / 44_100.0;
        assert!((t - expected).abs() < 1e-20, "t={t} expected={expected}");
    }

    #[test]
    fn dominant_frequency_recovers_a_decaying_tone() {
        let sample_rate = 48_000.0;
        let freq = 21_234.0;
        let segment: Vec<f64> = (0..600)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin() * (i as f64 / 200.0).exp())
            .collect();
        let found = dominant_frequency(&segment, sample_rate);
        assert!((found - freq).abs() < 10.0, "found={found}");
    }

    #[test]
    fn fit_delay_recovers_a_fractional_shift() {
        let output_rate = 48_000;
        let transient = Transient::below(20_000.0, 0.0);
        let center = 500.0;
        let shift = -0.3125;
        let times = |j: usize| (j as f64 - center) / output_rate as f64;
        let output: Vec<f64> = (0..1000)
            .map(|j| transient.at(times(j) - shift / output_rate as f64))
            .collect();
        let fitted = fit_delay(&output, output_rate, center, &transient, &times);
        assert!((fitted - shift).abs() < 1e-6, "fitted={fitted}");
    }
}
