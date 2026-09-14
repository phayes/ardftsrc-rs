//! Reads back `thdn_<preset>_report.json` (written by `run thdn`) and
//! `hydrogen_src_<preset>_report.json` (written by `run hydrogen-src`) from `--out-dir`
//! and renders one combined `report_<preset>.md` per preset present, plus a
//! `README.md` index linking those reports. A preset missing
//! both files is skipped with a warning; a preset missing just one still gets a report,
//! with a note in place of the missing section.
//!
//! Layout: an overall title, a transposed preset-configuration table, the HydrogenAudio
//! Test Suite section (with titled whitelist figures copied alongside the report by
//! `run hydrogen-src` embedded directly), then THD+N -- which omits its own
//! preset-configuration table since it would just repeat the one already shown at the
//! top.

use std::fmt::Write as _;
use std::path::Path;

use tabled::settings::Alignment;

use crate::git::git_revision;
use crate::hydrogen::{FIGURES, PresetResult};
use crate::preset::Preset;
use crate::thdn::report::{Report as ThdnReport, render_table};

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Renders the preset's `ardftsrc::Config` as a `Field | Value` table (one row per
/// field, transposed relative to thdn's old one-row-per-preset layout, since a combined
/// report only ever covers a single preset). Omits "Phase intensity" when phase is
/// exactly zero, since it has no effect in that case.
fn preset_config_table(preset: Preset) -> String {
    let c = preset.base_config();
    let mut rows = vec![
        vec!["Quality".to_string(), c.quality.to_string()],
        vec!["Bandwidth".to_string(), format!("{:.4}", c.bandwidth)],
        vec!["Taper".to_string(), format!("{:?}", c.taper_type)],
        vec!["Phase".to_string(), c.phase.to_string()],
    ];
    if c.phase != 0.0 {
        rows.push(vec!["Phase intensity".to_string(), c.phase_intensity.to_string()]);
    }
    render_table(&[("Field", Alignment::left()), ("Value", Alignment::right())], rows)
}

fn hydrogen_src_table(case: &PresetResult) -> String {
    let rows = vec![
        vec!["Balanced score".to_string(), format!("{:.2}%", case.balanced_score)],
        vec!["Spectrogram".to_string(), format!("{:.2}%", case.spectrogram_score)],
        vec!["Bandwidth".to_string(), format!("{:.2}%", case.bandwidth_score)],
        vec![
            "Impulse frequency".to_string(),
            format!("{:.2}%", case.impulse_freq_score),
        ],
        vec![
            "Average impulse freq".to_string(),
            format!("{:.2} dB", case.average_impulse_freq),
        ],
        vec!["Aliasing".to_string(), format!("{:.2}%", case.alias_score)],
        vec!["Pre-ringing".to_string(), format!("{:.2}%", case.preringing_score)],
        vec!["Gapless".to_string(), format!("{:.2}%", case.gapless_score)],
        vec![
            "Intermodulation".to_string(),
            format!("{:.2}%", case.intermoddiff_score),
        ],
        vec!["Delay".to_string(), format!("{:.2}%", case.delay_score)],
    ];
    render_table(&[("Metric", Alignment::left()), ("Score", Alignment::right())], rows)
}

/// Reads `thdn_<preset>_report.json` and `hydrogen_src_<preset>_report.json` from
/// `out_dir` for each preset in [`Preset::ALL`] and writes `report_<preset>.md`
/// combining whichever of the two is present, plus a `README.md` index linking
/// each generated report.
pub fn write_all(out_dir: &Path, quiet: bool) {
    let revision = git_revision();
    let mut written = Vec::new();

    for &preset in Preset::ALL.iter() {
        let thdn_path = out_dir.join(format!("thdn_{}_report.json", preset.label()));
        let hydrogen_src_path = out_dir.join(format!("hydrogen_src_{}_report.json", preset.label()));

        let thdn: Option<ThdnReport> = read_json(&thdn_path);
        let hydrogen: Option<PresetResult> = read_json(&hydrogen_src_path);

        if thdn.is_none() && hydrogen.is_none() {
            if !quiet {
                eprintln!(
                    "report: skipping {} (neither {} nor {} found)",
                    preset.label(),
                    thdn_path.display(),
                    hydrogen_src_path.display()
                );
            }
            continue;
        }

        let mut out = String::new();

        let _ = writeln!(out, "# ardftsrc Quality Report: {}", capitalize(preset.label()));
        let _ = writeln!(out);
        if let Some(revision) = revision.as_deref() {
            let _ = writeln!(out, "Revision: {revision}");
            let _ = writeln!(out);
        }

        let _ = writeln!(out, "## Preset Configuration");
        let _ = writeln!(out);
        let _ = writeln!(out, "{}", preset_config_table(preset));
        let _ = writeln!(out);

        let _ = writeln!(out, "## HydrogenAudio Test Suite");
        let _ = writeln!(out);
        match &hydrogen {
            Some(case) => {
                let _ = writeln!(
                    out,
                    "Local HydrogenAudio Test Suite run, `f64`, `f128` off. See \
                     <https://src.hydrogenaudio.org/> for the test suite's own scoring \
                     methodology."
                );
                let _ = writeln!(out);
                let _ = writeln!(out, "{}", hydrogen_src_table(case));
                for spec in FIGURES {
                    let dest = spec.dest_name(case.preset);
                    if !case.figures.iter().any(|name| name == &dest) {
                        continue;
                    }
                    let _ = writeln!(out);
                    let _ = writeln!(out, "### {}", spec.title);
                    let _ = writeln!(out);
                    let _ = writeln!(out, "{}", spec.description);
                    let _ = writeln!(out);
                    let _ = writeln!(
                        out,
                        r#"<img src="{dest}" alt="{}" width="50%" />"#,
                        spec.title
                    );
                }
            }
            None => {
                let _ = writeln!(
                    out,
                    "No `{}` found. Run `ardftsrc-report run hydrogen-src` first \
                     (requires GNU Octave on `PATH`).",
                    hydrogen_src_path.display()
                );
            }
        }
        let _ = writeln!(out);

        let _ = writeln!(out, "## THD+N");
        let _ = writeln!(out);
        match &thdn {
            Some(report) => {
                let _ = write!(out, "{}", report.to_markdown());
            }
            None => {
                let _ = writeln!(
                    out,
                    "No `{}` found. Run `ardftsrc-report run thdn` first.",
                    thdn_path.display()
                );
            }
        }

        let md_path = out_dir.join(format!("report_{}.md", preset.label()));
        std::fs::write(&md_path, out).unwrap_or_else(|e| panic!("failed to write {}: {e}", md_path.display()));
        written.push(preset);
        if !quiet {
            eprintln!("report: wrote {}", md_path.display());
        }
    }

    write_index(out_dir, &written, revision.as_deref(), quiet);
}

/// Writes `README.md` in `out_dir` linking each `report_<preset>.md` that
/// [`write_all`] produced. Skipped if no preset reports were written.
fn write_index(out_dir: &Path, presets: &[Preset], revision: Option<&str>, quiet: bool) {
    if presets.is_empty() {
        return;
    }

    let mut out = String::new();
    let _ = writeln!(out, "# ardftsrc Quality Reports");
    let _ = writeln!(out);
    if let Some(revision) = revision {
        let _ = writeln!(out, "Revision: {revision}");
        let _ = writeln!(out);
    }
    for &preset in presets {
        let _ = writeln!(
            out,
            "- [{}](report_{}.md)",
            capitalize(preset.label()),
            preset.label()
        );
    }

    let path = out_dir.join("README.md");
    std::fs::write(&path, out).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    if !quiet {
        eprintln!("report: wrote {}", path.display());
    }
}
