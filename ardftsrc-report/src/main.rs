use std::path::PathBuf;

use ardftsrc_report::{hydrogen, preset::Preset, report, thdn};
use clap::{Args, Parser, Subcommand};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Combined THD+N and HydrogenAudio quality reporting for ardftsrc's `f64` resamplers.
#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run one report stage in isolation and write its JSON artifact(s).
    Run {
        #[command(subcommand)]
        target: RunTarget,
    },
    /// Render report_<preset>.md from thdn_<preset>_report.json and
    /// hydrogen_src_<preset>_report.json already present in --out-dir.
    Report(ReportArgs),
    /// Run both analyses and render the combined reports: equivalent to
    /// `run thdn`, then `run hydrogen-src`, then `report`.
    All(AllArgs),
}

#[derive(Debug, Subcommand)]
enum RunTarget {
    /// Run the THD+N sweep and write thdn_<preset>_report.json.
    Thdn(ThdnArgs),
    /// Run the HydrogenAudio Test Suite once per preset (f64, dd_fft off) and
    /// write hydrogen_src_<preset>_report.json. Requires GNU Octave on PATH.
    HydrogenSrc(HydrogenSrcArgs),
}

#[derive(Debug, Args)]
struct ThdnArgs {
    /// Directory to write thdn_<preset>_report.json to. Relative paths resolve against
    /// the current working directory. Created if missing.
    #[arg(long, default_value = ".")]
    out_dir: PathBuf,

    /// Suppress per-case progress output on stderr.
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Args, Clone)]
struct HydrogenSrcArgs {
    /// Directory to write hydrogen_src_<preset>_report.json to. Relative paths resolve
    /// against the current working directory. Created if missing.
    #[arg(long, default_value = ".")]
    out_dir: PathBuf,

    /// Workspace directory for hydrogen_src's generated samples/output/analysis
    /// artifacts (can grow large; safe to point at a scratch/gitignored directory).
    /// Defaults to hydrogen_src's own platform cache directory.
    #[arg(long)]
    workdir: Option<PathBuf>,

    /// Suppress per-preset progress output on stderr.
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// Directory to read thdn_*.json / hydrogen_src_*.json from and write report_*.md
    /// to. Relative paths resolve against the current working directory.
    #[arg(long, default_value = ".")]
    out_dir: PathBuf,

    /// Suppress per-preset progress output on stderr.
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, Args)]
struct AllArgs {
    #[arg(long, default_value = ".")]
    out_dir: PathBuf,

    /// See `run hydrogen-src --workdir`.
    #[arg(long)]
    workdir: Option<PathBuf>,

    #[arg(long)]
    quiet: bool,
}

fn run_thdn(args: &ThdnArgs) {
    std::fs::create_dir_all(&args.out_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", args.out_dir.display()));

    let total = thdn::report::planned_case_count(
        thdn::report::DEFAULT_RATE_PAIRS,
        thdn::report::DEFAULT_PRESETS,
        thdn::report::DEFAULT_FREQUENCIES_HZ,
        thdn::report::DEFAULT_AMPLITUDES_DBFS,
    );
    let mut done = 0usize;

    if !args.quiet {
        eprintln!(
            "thdn: running {total} cases ({} rate pairs (some also run decimate=true) x {} presets x {} dd_fft variant(s) x {} frequencies x {} amplitudes)",
            thdn::report::DEFAULT_RATE_PAIRS.len(),
            thdn::report::DEFAULT_PRESETS.len(),
            thdn::report::DD_FFT_VARIANTS.len(),
            thdn::report::DEFAULT_FREQUENCIES_HZ.len(),
            thdn::report::DEFAULT_AMPLITUDES_DBFS.len(),
        );
    }

    let result = thdn::report::run_sweep(
        thdn::report::DEFAULT_RATE_PAIRS,
        thdn::report::DEFAULT_PRESETS,
        thdn::report::DEFAULT_FREQUENCIES_HZ,
        thdn::report::DEFAULT_AMPLITUDES_DBFS,
        |case| {
            done += 1;
            if !args.quiet {
                eprintln!(
                    "[{done}/{total}] {} -> {} decimate={} {} dd_fft={} {:.0}Hz {:.0}dBFS: broadband={:.2}dB audio_band={:.2}dB gain_err={:.3}dB spur={:.2}dB@{:.0}Hz",
                    case.input_rate,
                    case.output_rate,
                    case.decimate,
                    case.preset.label(),
                    case.dd_fft,
                    case.freq_hz,
                    case.amplitude_dbfs,
                    case.thdn_broadband_db,
                    case.thdn_audio_band_db,
                    case.gain_error_db,
                    case.max_spur_db,
                    case.max_spur_freq_hz,
                );
            }
        },
    );

    for &preset in Preset::ALL.iter() {
        let preset_result = result.for_preset(preset);
        if preset_result.cases.is_empty() {
            continue;
        }

        let json_path = args.out_dir.join(format!("thdn_{}_report.json", preset.label()));
        let json = serde_json::to_string_pretty(&preset_result).expect("Report serializes to JSON");
        std::fs::write(&json_path, json).unwrap_or_else(|e| panic!("failed to write {}: {e}", json_path.display()));

        if !args.quiet {
            eprintln!("thdn: wrote {}", json_path.display());
        }
    }
}

fn run_hydrogen_src(args: &HydrogenSrcArgs) {
    std::fs::create_dir_all(&args.out_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", args.out_dir.display()));

    hydrogen::run_all_presets(args.workdir.as_deref(), &args.out_dir, args.quiet)
        .unwrap_or_else(|e| panic!("hydrogen_src run failed: {e}"));
}

fn run_report(args: &ReportArgs) {
    std::fs::create_dir_all(&args.out_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", args.out_dir.display()));
    report::write_all(&args.out_dir, args.quiet);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            target: RunTarget::Thdn(args),
        } => run_thdn(&args),
        Command::Run {
            target: RunTarget::HydrogenSrc(args),
        } => run_hydrogen_src(&args),
        Command::Report(args) => run_report(&args),
        Command::All(args) => {
            run_thdn(&ThdnArgs {
                out_dir: args.out_dir.clone(),
                quiet: args.quiet,
            });
            run_hydrogen_src(&HydrogenSrcArgs {
                out_dir: args.out_dir.clone(),
                workdir: args.workdir.clone(),
                quiet: args.quiet,
            });
            run_report(&ReportArgs {
                out_dir: args.out_dir,
                quiet: args.quiet,
            });
        }
    }
}
