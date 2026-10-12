//! Runs ardftsrc through the HydrogenAudio Test Suite (the external
//! `hydrogen_src`/`octave` toolchain at <https://github.com/phayes/hydrogen_src>) once
//! once per [`Variant`]: every preset in `f64`, plus [`Preset::Fast`] in `f32`. No
//! high-precision backend.
//!
//! Requires GNU Octave on `PATH` (with the `signal` and `image` packages).

use std::path::{Path, PathBuf};

use ardftsrc::{Config, PlanarResampler};
use hydrogen_src::{HydrogenError, LocalHarness, LocalTestResults, ResampleRequestF32, ResampleRequestF64};
use serde::{Deserialize, Serialize};

use crate::preset::Preset;

/// Sample type the resampler runs in for a HydrogenAudio Test Suite run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    #[default]
    F64,
}

impl Precision {
    pub fn label(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }
}

/// One HydrogenAudio Test Suite run: a preset in a given [`Precision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variant {
    pub preset: Preset,
    pub precision: Precision,
}

impl Variant {
    /// File-name stem shared by this variant's JSON, figures, and Markdown report:
    /// `<preset>` for `f64`, `<preset>_f32` for `f32`.
    pub fn label(self) -> String {
        match self.precision {
            Precision::F64 => self.preset.label().to_string(),
            Precision::F32 => format!("{}_{}", self.preset.label(), self.precision.label()),
        }
    }
}

/// Every variant run by [`run_all`], in report display order.
pub const VARIANTS: &[Variant] = &[
    Variant {
        preset: Preset::Fast,
        precision: Precision::F64,
    },
    Variant {
        preset: Preset::Fast,
        precision: Precision::F32,
    },
    Variant {
        preset: Preset::Good,
        precision: Precision::F64,
    },
    Variant {
        preset: Preset::High,
        precision: Precision::F64,
    },
];

/// One preset's HydrogenAudio Test Suite scores. Mirrors [`LocalTestResults`] but
/// replaces `figures` (absolute, workdir/machine-dependent paths) with file names of
/// copies made alongside the report (see [`copy_figures`]), so the Markdown report can
/// reference them with plain relative links and the JSON report stays diffable across
/// machines/runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetResult {
    pub preset: Preset,
    #[serde(default)]
    pub precision: Precision,
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
    /// [`Variant::label`] so every variant's figures can share one output directory
    /// without colliding.
    pub figures: Vec<String>,
}

/// One HydrogenAudio analysis figure copied into `--out-dir` and embedded in
/// `report_<variant>.md`.
#[derive(Debug, Clone, Copy)]
pub struct Figure {
    /// Original file name in `hydrogen_src`'s `analysis_output` directory.
    pub file_name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
}

impl Figure {
    pub fn dest_name(&self, variant: Variant) -> String {
        format!("{}_{}", variant.label(), self.file_name)
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
    pub fn variant(&self) -> Variant {
        Variant {
            preset: self.preset,
            precision: self.precision,
        }
    }

    fn from_local_results(variant: Variant, results: LocalTestResults, out_dir: &Path) -> Self {
        PresetResult {
            preset: variant.preset,
            precision: variant.precision,
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
            figures: copy_figures(variant, &results.figures, out_dir),
        }
    }
}

/// Copies each of [`FIGURES`] from `figures` (absolute paths into the
/// `hydrogen_src` workdir's `analysis_output`) into `out_dir`, renamed
/// `<variant>_<original file name>` so every variant's figures coexist in one
/// report directory and `report_<variant>.md` can reference them with a plain
/// relative Markdown image link. Source files that are missing or fail to copy
/// are skipped with a warning rather than failing the whole run.
fn copy_figures(variant: Variant, figures: &[PathBuf], out_dir: &Path) -> Vec<String> {
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
                    variant.label()
                );
                return None;
            };
            let dest_name = spec.dest_name(variant);
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

fn resample_f32(request: ResampleRequestF32, base: Config) -> Vec<f32> {
    let config = Config {
        input_sample_rate: request.sample_rate,
        output_sample_rate: request.target_sample_rate,
        channels: request.channels,
        ..base
    };
    let mut resampler = PlanarResampler::<f32>::new(config).expect("ardftsrc-report preset configs are always valid");
    let mut output = resampler
        .process_all(&[request.samples.as_slice()])
        .expect("resampling the HydrogenAudio test pack should never fail");
    output
        .pop_channel()
        .expect("PlanarResampler output always has exactly one channel for mono input")
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

/// Runs the HydrogenAudio Test Suite once per [`VARIANTS`] entry (no high-precision
/// backend) and writes `hydrogen_src_<variant>_report.json` into `out_dir` as each variant
/// finishes. `workdir` is passed straight to [`LocalHarness::new`]; `None` uses
/// `hydrogen_src`'s own default (platform cache directory).
///
/// Every run shares one workspace. `LocalHarness::run`'s one-time setup -- checking for
/// `octave`, clearing the workspace's output/analysis directories, generating the fixed
/// Octave test-signal WAVs -- is gated behind a process-global flag, so it only actually
/// runs for the *first* `LocalHarness::run()` call in this process no matter how many
/// harness instances exist; a fresh workdir per variant would silently skip that setup for
/// every variant after the first. `run()` restores its callback slot after each call, so
/// calling it repeatedly on one instance with a new callback before each call is the
/// supported way to sweep presets. A harness rejects having both an `f32` and an `f64`
/// callback set, so each precision gets its own harness over the same workspace.
///
/// `f32` variants run before any `f64` variant. Resampled outputs accumulate in the shared
/// workspace across runs (named `<signal>-32bitfloat.wav` / `<signal>-64bitfloat.wav`), and
/// before each analysis the harness copies every one of them, in sorted order, to the same
/// suffix-free `<signal>.wav` that Octave scores. A `-64bitfloat` output left by an earlier
/// `f64` run would therefore overwrite the `-32bitfloat` one and an `f32` run would be scored
/// on `f64` output; with `f32` first, each later `f64` run's output correctly wins instead.
pub fn run_all(workdir: Option<&Path>, out_dir: &Path, quiet: bool) -> Result<Vec<PresetResult>, HydrogenError> {
    let mut harness_f32 = LocalHarness::new(workdir.map(PathBuf::from))?;
    let mut harness_f64 = LocalHarness::new(workdir.map(PathBuf::from))?;
    if !quiet {
        eprintln!("hydrogen_src: workspace at {}", harness_f64.workspace().display());
    }

    let run_order = VARIANTS
        .iter()
        .filter(|v| v.precision == Precision::F32)
        .chain(VARIANTS.iter().filter(|v| v.precision == Precision::F64));

    let mut all = Vec::with_capacity(VARIANTS.len());
    for &variant in run_order {
        let label = variant.label();
        if !quiet {
            eprintln!("hydrogen_src: running {label}");
        }
        let base = variant.preset.base_config();
        let results = match variant.precision {
            Precision::F32 => {
                harness_f32.set_callback_f32(move |request: ResampleRequestF32| resample_f32(request, base.clone()));
                harness_f32.run()?
            }
            Precision::F64 => {
                harness_f64.set_callback_f64(move |request: ResampleRequestF64| resample_f64(request, base.clone()));
                harness_f64.run()?
            }
        };
        let case = PresetResult::from_local_results(variant, results, out_dir);

        let json_path = out_dir.join(format!("hydrogen_src_{label}_report.json"));
        std::fs::write(
            &json_path,
            serde_json::to_string_pretty(&case).expect("PresetResult serializes to JSON"),
        )
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", json_path.display()));
        if !quiet {
            eprintln!(
                "hydrogen_src: [{label}] balanced={:.2} -> wrote {}",
                case.balanced_score,
                json_path.display()
            );
        }
        all.push(case);
    }
    Ok(all)
}
