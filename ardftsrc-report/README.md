# ardftsrc-report

Combined quality reporting for [ardftsrc](https://crates.io/crates/ardftsrc)'s `f64`
resamplers: per-preset THD+N (total harmonic distortion + noise) plus HydrogenAudio SRC
scores, joined into one Markdown report per preset.

## Flow

1. `run thdn` -- pure-Rust THD+N sweep across frequency, amplitude, sample-rate pair,
   preset, and (opt-in, via `--features f128`) FFT backend. Writes
   `thdn_<preset>_report.json`.
2. `run hydrogen-src` -- runs ardftsrc through the HydrogenAudio Test Suite's local
   Octave analysis once per preset (`f64`, `f128` off). Writes
   `hydrogen_src_<preset>_report.json`, plus copies of that preset's figure PNGs
   (`<preset>_<figure>.png`) into `--out-dir` for the report to embed.
3. `report` -- reads back whichever of the two JSON files are present per preset and
   writes a combined `report_<preset>.md`.

`all` runs all three stages in sequence.

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
# slower run that also covers the f128 backend (requires a nightly rustc):
cargo run -p ardftsrc-report --release --features f128 -- run thdn --out-dir reports
```

`f128` is opt-in (it requires a nightly `rustc`, since the `f128` primitive type is still
unstable) rather than on by default. Passing `--features f128` covers both FFT backends
in one run -- expect it to take several minutes longer, since the `f128` backend is much
slower. Writes one machine-readable `thdn_<preset>_report.json` per preset (for
regression testing -- diff it against a checked-in baseline).

### `run hydrogen-src`

```bash
cargo run -p ardftsrc-report --release -- run hydrogen-src --out-dir reports
```

Runs the [HydrogenAudio Test Suite](https://src.hydrogenaudio.org/)'s own local
Octave analysis toolbox (via the external
[`hydrogen_src`](https://github.com/phayes/hydrogen_src) crate) once per preset, `f64`
only, `f128` off -- matching the single score already linked per preset in
[`PERFORMANCE.md`](../PERFORMANCE.md).

Requires GNU Octave on `PATH`, with the `signal` and `image` packages installed.
`hydrogen_src` isn't published to crates.io, so it's a `git` dependency in
`Cargo.toml` pointing at its GitHub repo -- no local checkout needed.

The first run does network I/O (fetches the `hydrogen_src` source at build time, plus
downloads/extracts a reference file at runtime if missing) and
generates the fixed Octave test-signal WAVs; expect it to be noticeably slower than
later runs. Pass `--workdir <DIR>` to control where `hydrogen_src` keeps its generated
WAVs/PNGs/analysis output (can grow large; defaults to `hydrogen_src`'s own platform
cache directory if omitted). A whitelist of figure PNGs is copied into `--out-dir`,
renamed `<preset>_<figure>.png`, and `report` embeds each with its title and
description.

### `report`

```bash
cargo run -p ardftsrc-report --release -- report --out-dir reports
```

Reads `thdn_<preset>_report.json` and `hydrogen_src_<preset>_report.json` back from
`--out-dir` for each preset and writes `report_<preset>.md` combining both (for
sharing), plus a `README.md` that links each generated report. A preset missing one of
the two JSON files still gets a report, with a note
in place of the missing section, so `report` can be re-run at any point in the flow.
Each report notes the repo's current git revision (preferring a `v*` tag pointing at
`HEAD`, falling back to the short commit hash; omitted if `git` isn't installed or the
working directory isn't inside a git repo).

## Tests

```bash
cargo test -p ardftsrc-report --release
# also exercises the f128 backend (requires a nightly rustc):
cargo +nightly test -p ardftsrc-report --release --features f128
```

