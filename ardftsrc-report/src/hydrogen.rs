//! Runs ardftsrc through the HydrogenAudio Test Suite (the external
//! `hydrogen_src`/`octave` toolchain at <https://github.com/phayes/hydrogen_src>) once
//! per preset, `f64` only with no high-precision backend.
//!
//! Requires GNU Octave on `PATH` (with the `signal` and `image` packages).

use std::path::{Path, PathBuf};

use ardftsrc::{Config, PlanarResampler};
use hydrogen_src::{HydrogenError, LocalHarness, LocalTestResults, ResampleRequestF64};
use serde::{Deserialize, Serialize};

use crate::preset::Preset;

/// One preset's HydrogenAudio Test Suite scores. Mirrors [`LocalTestResults`] but
/// replaces `figures` (absolute, workdir/machine-dependent paths) with file names of
/// copies made alongside the report (see [`copy_figures`]), so the Markdown report can
/// reference them with plain relative links and the JSON report stays diffable across
/// machines/runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetResult {
    pub preset: Preset,
    pub balanced_score: f64,
    pub spectrogram_score: f64,
    pub bandwidth_score: f64,
    pub impulse_freq_score: f64,
    pub average_impulse_freq: f64,
    pub alias_score: f64,
    pub preringing_score: f64,
    pub gapless_score: f64,
    pub intermoddiff_score: f64,
    pub delay_score: f64,
    /// File names (relative to the report `--out-dir`) of the whitelisted
    /// [`FIGURES`] copied from [`LocalTestResults::figures`], each prefixed with
    /// `<preset>_` so all four presets' figures can share one output directory
    /// without colliding.
    pub figures: Vec<String>,
}

/// One HydrogenAudio analysis figure copied into `--out-dir` and embedded in
/// `report_<preset>.md`.
#[derive(Debug, Clone, Copy)]
pub struct Figure {
    /// Original file name in `hydrogen_src`'s `analysis_output` directory.
    pub file_name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
}

impl Figure {
    pub fn dest_name(&self, preset: Preset) -> String {
        format!("{}_{}", preset.label(), self.file_name)
    }
}

/// Figures included in the report, in display order. Octave also writes a
/// high-res scoring-only spectrogram (`sweep-1-to-44KHz-1to11secHighRES.png`);
/// that file is intentionally omitted.
pub const FIGURES: &[Figure] = &[
    Figure {
        file_name: "sweep-1-to-44KHz-1to11sec.png",
        title: "Spectrogram of a sweep 1 to 22 kHz",
        description: "A sine sweep from 1 to 22 kHz. The ideal plot is a single strong red line. \
Issues with aliasing effects or filter cutoff would show as extra lines. Noise would appear as \
dots across the plot (instead of a black background). [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "sweep-1-to-44KHz.png",
        title: "Spectrogram of a sweep 1 to 22 kHz (extended)",
        description: "The source signal included a sine sweep all the way up to 44 kHz, however \
when downsampled to 44 kHz the highest frequency which can be represented is 22 kHz. This plot \
would show if the sine above 22 kHz is filtering down into the plot, there should be nothing \
plotted after 10 seconds. [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "aliasing150db.png",
        title: "Aliasing",
        description: "A 23 kHz sine at -4 dBFS with a white noise floor of -150 dBFS over 30 \
seconds. The ideal plot is a continuous line touching the -150dB line, many SRC routines engage \
a gradual filter at 20 kHz which would be visible on this plot. [96 kHz source resampled to 44 \
kHz]",
    },
    Figure {
        file_name: "nyquist-filter.png",
        title: "Nyquist Filter",
        description: "Zoomed on nyquist frequency (22.050 kHz), the bandwidth of the SRC is \
displayed. A SRC with 100% bandwidth, that is full frequency preservation would be a straight \
line of noise. [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "intermodulation-harmonic-distortion.png",
        title: "Intermodulation Harmonic Distortion",
        description: "Two sine waves one at 64.59 Hz, -6 dBFS and the second at 6998 Hz, \
-18.0412 dBFS, which equals quarter the amplitude of the first sine. This test will highlight \
aliasing and dynamic range of processing. It will also show if dither has been applied, look \
for high frequency signals on the plot. [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "intermodulation-harmonic-distortion-difference.png",
        title: "Intermodulation Harmonic Distortion (difference)",
        description: "The difference between the ideal and measured signals from the \
Intermodulation Harmonic Distortion test. [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "impulse-frequency.png",
        title: "Impulse Frequency",
        description: "A frequency response displaying leakage beyond the ideal frequency \
response. Impulses are at sample positions n so that {n mod 320} is a permutation of {0, 1, 2, \
. . . , 319}. All fractional differences in sample positions between input and output signal \
are addressed exactly once. The output signal is upsampled to 14.112 MHz and the impulse \
responses are added to obtain a high resolution impulse response. [96 kHz source resampled to \
44 kHz]",
    },
    Figure {
        file_name: "impulse-response.png",
        title: "Impulse Response",
        description: "Displays the phase response, most SRC balance the response with pre and \
post ringing, personal preference might prefer a minimum phase response where there is no \
pre-ringing. The ideal response is not shown because a post-ringing representation, would not \
tally with the other ideal plots (phase, etc). [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "impulse-phase.png",
        title: "Impulse Phase",
        description: "The actual phase across the frequency range. [96 kHz source resampled to \
44 kHz]",
    },
    Figure {
        file_name: "impulse-passband.png",
        title: "Impulse Passband",
        description: "Displays the SRC filter used close to the nyquist frequency. [96 kHz \
source resampled to 44 kHz]",
    },
    Figure {
        file_name: "impulse-transition.png",
        title: "Impulse Transition",
        description: "A zoomed plot of the nyquist frequency showing the filter response. [96 \
kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "gaplesstestsine.png",
        title: "Gapless Sine",
        description: "A sine wave is split into two, both are resampled independently. This \
plot is the two signals joined back together. The blue line shows the transition after \
resampling. Deviation from sine wave shape likely means an audible glitch. Any value over +1 \
would clip if later not corrected. [96 kHz source resampled to 44 kHz]",
    },
    Figure {
        file_name: "gaplesstest-frequency.png",
        title: "Gapless Sine (frequency plot)",
        description: "A frequency plot of the gapless sine test. [96 kHz source resampled to \
44 kHz]",
    },
];

impl PresetResult {
    fn from_local_results(preset: Preset, results: LocalTestResults, out_dir: &Path) -> Self {
        PresetResult {
            preset,
            balanced_score: results.balanced_score,
            spectrogram_score: results.spectrogram_score,
            bandwidth_score: results.bandwidth_score,
            impulse_freq_score: results.impulse_freq_score,
            average_impulse_freq: results.average_impulse_freq,
            alias_score: results.alias_score,
            preringing_score: results.preringing_score,
            gapless_score: results.gapless_score,
            intermoddiff_score: results.intermoddiff_score,
            delay_score: results.delay_score,
            figures: copy_figures(preset, &results.figures, out_dir),
        }
    }
}

/// Copies each of [`FIGURES`] from `figures` (absolute paths into the
/// `hydrogen_src` workdir's `analysis_output`) into `out_dir`, renamed
/// `<preset>_<original file name>` so the four presets' figures coexist in one
/// report directory and `report_<preset>.md` can reference them with a plain
/// relative Markdown image link. Source files that are missing or fail to copy
/// are skipped with a warning rather than failing the whole run.
fn copy_figures(preset: Preset, figures: &[PathBuf], out_dir: &Path) -> Vec<String> {
    FIGURES
        .iter()
        .filter_map(|spec| {
            let Some(src) = figures
                .iter()
                .find(|path| path.file_name().and_then(|name| name.to_str()) == Some(spec.file_name))
            else {
                eprintln!(
                    "hydrogen_src: warning: missing figure {} for {}",
                    spec.file_name,
                    preset.label()
                );
                return None;
            };
            let dest_name = spec.dest_name(preset);
            let dest_path = out_dir.join(&dest_name);
            match std::fs::copy(src, &dest_path) {
                Ok(_) => Some(dest_name),
                Err(e) => {
                    eprintln!(
                        "hydrogen_src: warning: failed to copy figure {} to {}: {e}",
                        src.display(),
                        dest_path.display()
                    );
                    None
                }
            }
        })
        .collect()
}

fn resample_f64(request: ResampleRequestF64, base: Config) -> Vec<f64> {
    let config = Config {
        input_sample_rate: request.sample_rate,
        output_sample_rate: request.target_sample_rate,
        channels: request.channels,
        ..base
    };
    let mut resampler = PlanarResampler::<f64>::new(config).expect("ardftsrc-report preset configs are always valid");
    let mut output = resampler
        .process_all(&[request.samples.as_slice()])
        .expect("resampling the HydrogenAudio test pack should never fail");
    output
        .pop_channel()
        .expect("PlanarResampler output always has exactly one channel for mono input")
}

/// Runs the HydrogenAudio Test Suite once per preset (`f64`, no high-precision backend) and
/// writes `hydrogen_src_<preset>_report.json` into `out_dir` as each preset finishes.
/// `workdir` is passed straight to [`LocalHarness::new`]; `None` uses `hydrogen_src`'s own
/// default (platform cache directory).
///
/// Deliberately reuses a single [`LocalHarness`] (and therefore a single workspace)
/// across every preset. `LocalHarness::run`'s one-time setup -- checking for `octave`,
/// generating the fixed Octave test-signal WAVs -- is gated behind a process-global
/// flag, so it only actually runs for the *first* `LocalHarness::run()` call in this
/// process no matter how many harness instances exist; a fresh harness/workdir per
/// preset would silently skip that setup (and the workspace-relative output/analysis
/// directory preparation that comes with it) for every preset after the first. `run()`
/// restores its callback slot after each call, so calling it repeatedly on one instance
/// -- with a new `set_callback_f64` before each call -- is the supported way to sweep
/// presets.
pub fn run_all_presets(
    workdir: Option<&Path>,
    out_dir: &Path,
    quiet: bool,
) -> Result<Vec<PresetResult>, HydrogenError> {
    let mut harness = LocalHarness::new(workdir.map(PathBuf::from))?;
    if !quiet {
        eprintln!("hydrogen_src: workspace at {}", harness.workspace().display());
    }

    let mut all = Vec::with_capacity(Preset::ALL.len());
    for &preset in Preset::ALL.iter() {
        if !quiet {
            eprintln!("hydrogen_src: running {}", preset.label());
        }
        let base = preset.base_config();
        harness.set_callback_f64(move |request: ResampleRequestF64| resample_f64(request, base.clone()));
        let results = harness.run()?;
        let case = PresetResult::from_local_results(preset, results, out_dir);

        let json_path = out_dir.join(format!("hydrogen_src_{}_report.json", preset.label()));
        std::fs::write(
            &json_path,
            serde_json::to_string_pretty(&case).expect("PresetResult serializes to JSON"),
        )
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", json_path.display()));
        if !quiet {
            eprintln!(
                "hydrogen_src: [{}] balanced={:.2} -> wrote {}",
                preset.label(),
                case.balanced_score,
                json_path.display()
            );
        }
        all.push(case);
    }
    Ok(all)
}
