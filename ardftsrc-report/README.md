# ardftsrc-report

Combined quality reporting for [ardftsrc](https://crates.io/crates/ardftsrc)'s `f64`
resamplers: per-preset THD+N (total harmonic distortion + noise), pre-ringing, and
HydrogenAudio SRC scores, joined into one Markdown report per preset. `PRESET_FAST` also
gets a separate `f32` report covering the HydrogenAudio SRC scores.

## Flow

1. `run thdn` -- pure-Rust THD+N sweep across frequency, amplitude, sample-rate pair,
   preset, and (opt-in, via `--features high_precision`) FFT backend. Writes
   `thdn_<preset>_report.json`.
2. `run preringing` -- pure-Rust impulse pre-ringing and below-rolloff pre-echo
   measurements across sample-rate pair and preset (`f64`, no high-precision backend).
   Writes `preringing_<preset>_report.json`.
3. `run hydrogen-src` -- runs ardftsrc through the HydrogenAudio Test Suite's local
   Octave analysis once per preset in `f64`, plus `fast` in `f32` (no high-precision
   backend). Writes `hydrogen_src_<variant>_report.json`, plus copies of that variant's
   figure PNGs (`<variant>_<figure>.png`) into `--out-dir` for the report to embed.
   `<variant>` is the preset name for `f64` and `<preset>_f32` for `f32`.
4. `report` -- reads back whichever of the three JSON files are present per variant and
   writes a combined `report_<variant>.md`. THD+N and pre-ringing are `f64`-only, so
   `report_fast_f32.md` carries the HydrogenAudio section and links to `report_fast.md`
   for the rest.

`all` runs all four stages in sequence.

## CLI

```bash
cargo run -p ardftsrc-report --release -- all --out-dir reports
```

`--out-dir` (default `.`) is created if missing; relative paths resolve against the
current working directory. Pass `--quiet` to any subcommand to suppress per-case/preset
progress output on stderr.

### `run thdn`

```bash
cargo run -p ardftsrc-report --release -- run thdn --out-dir reports
# slower run that also covers the high-precision backends (double-double, f128, f256):
cargo run -p ardftsrc-report --release --features high_precision -- run thdn --out-dir reports
```

`high_precision` is opt-in rather than on by default. Passing `--features high_precision`
covers the default and all three high-precision FFT backends in one run -- expect it to
take much longer, since the high-precision backends are far slower. Writes one machine-readable `thdn_<preset>_report.json` per preset (for
regression testing -- diff it against a checked-in baseline).

### `run preringing`

```bash
cargo run -p ardftsrc-report --release -- run preringing --out-dir reports
```

Two measurements per rate pair (plus `decimate=true` where it engages), each repeated with
the stimulus at several positions within the resampler's processing chunk:

- **Impulse pre-ringing**: a unit impulse is resampled, and the report gives how long
  before the impulse's ideal output time the response first reaches -60/-100/-140 dB
  relative to its peak, plus the dominant frequency of that pre-ringing tail.
- **Pre-ringing below the rolloff**: Gaussian clicks and tone bursts whose spectra are
  at least 300 dB down at the passband edge are resampled and compared against their exact
  analytic ideal output, after removing a best-fit constant delay (recorded in the JSON as
  `delay_offset_samples`). The passband gain is unity, so any residual ahead of the
  transient is pre-ringing that reached frequencies below the rolloff.

Writes one `preringing_<preset>_report.json` per preset.

### `run hydrogen-src`

```bash
cargo run -p ardftsrc-report --release -- run hydrogen-src --out-dir reports
```

Runs the [HydrogenAudio Test Suite](https://src.hydrogenaudio.org/)'s own local
Octave analysis toolbox (via the external
[`hydrogen_src`](https://github.com/phayes/hydrogen_src) crate) once per preset in `f64`,
plus `fast` in `f32`, with no high-precision backend. The `f32` run goes first: the
harness's shared workspace keeps every run's resampled outputs, and a leftover `f64`
output would otherwise be scored in place of the `f32` one.

Requires GNU Octave on `PATH`, with the `signal` and `image` packages installed.
`hydrogen_src` isn't published to crates.io, so it's a `git` dependency in
`Cargo.toml` pointing at its GitHub repo -- no local checkout needed.

The first run does network I/O (fetches the `hydrogen_src` source at build time, plus
downloads/extracts a reference file at runtime if missing) and
generates the fixed Octave test-signal WAVs; expect it to be noticeably slower than
later runs. Pass `--workdir <DIR>` to control where `hydrogen_src` keeps its generated
WAVs/PNGs/analysis output (can grow large; defaults to `hydrogen_src`'s own platform
cache directory if omitted). A whitelist of figure PNGs is copied into `--out-dir`,
renamed `<variant>_<figure>.png`, and `report` embeds each with its title and
description.

### `report`

```bash
cargo run -p ardftsrc-report --release -- report --out-dir reports
```

Reads `thdn_<preset>_report.json`, `preringing_<preset>_report.json`, and
`hydrogen_src_<variant>_report.json` back from `--out-dir` for each variant and writes
`report_<variant>.md` combining them (for sharing), plus a `README.md` that links each
generated report. A preset missing some of the JSON files still gets a report, with a
note in place of each missing section, so `report` can be re-run at any point in the flow.
Each report notes the workspace version (shared by `ardftsrc`) and the repo's current git
revision (preferring a `v*` tag pointing at `HEAD`, falling back to the short commit hash;
omitted if `git` isn't installed, the working directory isn't inside a git repo, or the
tag is just `v<version>`).

## Tests

```bash
cargo test -p ardftsrc-report --release
# also exercises the high-precision backends:
cargo test -p ardftsrc-report --release --features high_precision
```

