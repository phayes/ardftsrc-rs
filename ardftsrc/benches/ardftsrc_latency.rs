//! Realtime suitability report: end-to-end latency and per-chunk compute time.
//!
//! FFT resampling is bursty: nothing is emitted until a whole input chunk has arrived, and then
//! the whole chunk is transformed at once. Mean throughput therefore says little about realtime
//! fitness. This bench reports, per preset and rate pair:
//!
//! - **chunk**: input chunk length in frames and in milliseconds of audio.
//! - **latency**: measured delay of an impulse through [`RealtimeResampler`] when samples are
//!   written at the input cadence and read at the output cadence, as a live pipeline would.
//! - **min latency**: the smallest fixed output delay a custom pipeline built directly on
//!   `process_chunk()` could use without underrunning (a lower bound, excluding compute time).
//! - **chunk compute**: wall-clock time of each `process_chunk()` call (median, p99, max). This
//!   burst lands in a single audio callback when resampling on the audio thread.
//! - **load**: median chunk compute time as a fraction of the chunk's audio duration (one core).
//!
//! A second table reports the latency floor at a very low `quality`, showing how small the chunk
//! can get for each rate pair regardless of preset.
//!
//! Run with `cargo bench --bench ardftsrc_latency` (add `--features high_precision` to include
//! the `high+dd` rows, which run `PRESET_HIGH` in `DoubleDouble`). Output is a Markdown table.

use std::hint::black_box;
use std::time::{Duration, Instant};

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use ardftsrc::{Config, InterleavedResampler, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, RealtimeResampler, Sample};

const CHANNELS: usize = 2;
const RATE_PAIRS: &[(usize, usize)] = &[
    (44_100, 48_000),
    (48_000, 44_100),
    (44_100, 96_000),
    (96_000, 48_000),
    (192_000, 48_000),
];

/// Chunks processed before timing starts.
const WARMUP_CHUNKS: usize = 50;
/// Timing stops after this many chunks or [`MAX_TIMING`], whichever comes first.
const MAX_TIMED_CHUNKS: usize = 5_000;
const MIN_TIMED_CHUNKS: usize = 100;
const MAX_TIMING: Duration = Duration::from_secs(3);

/// Chunks of steady-state stream written before the latency impulse.
const IMPULSE_AFTER_CHUNKS: usize = 8;

struct Preset {
    name: &'static str,
    config: Config,
}

fn presets() -> Vec<Preset> {
    #[allow(unused_mut)]
    let mut presets = vec![
        Preset {
            name: "fast",
            config: PRESET_FAST,
        },
        Preset {
            name: "good",
            config: PRESET_GOOD,
        },
        Preset {
            name: "high",
            config: PRESET_HIGH,
        },
    ];
    #[cfg(feature = "high_precision")]
    presets.push(Preset {
        name: "high+dd",
        config: PRESET_HIGH,
    });
    presets
}

struct Row {
    preset: &'static str,
    sample_type: &'static str,
    input_rate: usize,
    output_rate: usize,
    chunk_frames: usize,
    chunk_ms: f64,
    latency_ms: f64,
    min_latency_ms: f64,
    underruns: usize,
    median_us: f64,
    p99_us: f64,
    max_us: f64,
}

impl Row {
    fn load_pct(&self) -> f64 {
        self.median_us / (self.chunk_ms * 1_000.0) * 100.0
    }
}

/// Sample types the bench runs, with conversions to and from `f64`.
trait BenchSample: Sample {
    fn widen(value: f64) -> Self;
    fn narrow(self) -> f64;
}

impl BenchSample for f32 {
    fn widen(value: f64) -> Self {
        value as f32
    }

    fn narrow(self) -> f64 {
        f64::from(self)
    }
}

impl BenchSample for f64 {
    fn widen(value: f64) -> Self {
        value
    }

    fn narrow(self) -> f64 {
        self
    }
}

#[cfg(feature = "high_precision")]
impl BenchSample for ardftsrc::DoubleDouble {
    fn widen(value: f64) -> Self {
        Self::from(value)
    }

    fn narrow(self) -> f64 {
        self.to_f64()
    }
}

/// Deterministic white noise in `[-0.5, 0.5)`.
fn noise<T: BenchSample>(len: usize) -> Vec<T> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            T::widen((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
        })
        .collect()
}

fn config_for(preset: &Preset, input_rate: usize, output_rate: usize) -> Config {
    preset
        .config
        .clone()
        .with_input_rate(input_rate)
        .with_output_rate(output_rate)
        .with_channels(CHANNELS)
}

/// Times each `process_chunk()` call and returns `(chunk_frames, sorted durations in µs)`.
fn time_chunks<T>(config: Config) -> (usize, Vec<f64>)
where
    T: BenchSample,
{
    let mut resampler = InterleavedResampler::<T>::new(config).expect("valid config");
    let input_len = resampler.input_buffer_size();
    let input = noise::<T>(input_len);
    let mut output = vec![T::widen(0.0); resampler.output_buffer_size()];

    for _ in 0..WARMUP_CHUNKS {
        black_box(resampler.process_chunk(black_box(&input), &mut output).unwrap());
    }

    let mut durations = Vec::with_capacity(MAX_TIMED_CHUNKS);
    let started = Instant::now();
    while durations.len() < MAX_TIMED_CHUNKS && (durations.len() < MIN_TIMED_CHUNKS || started.elapsed() < MAX_TIMING) {
        let t0 = Instant::now();
        black_box(resampler.process_chunk(black_box(&input), &mut output).unwrap());
        durations.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    durations.sort_by(f64::total_cmp);
    (input_len / CHANNELS, durations)
}

/// Returns the smallest fixed delay (ms) at which a chunk-driven pipeline never runs dry: the
/// worst case, over chunks, of the time a sample waits between arriving and its output existing.
fn min_pipeline_latency<T>(config: Config) -> f64
where
    T: BenchSample,
{
    let input_rate = config.input_sample_rate as f64;
    let output_rate = config.output_sample_rate as f64;
    let mut resampler = InterleavedResampler::<T>::new(config).expect("valid config");
    let input = vec![T::widen(0.0); resampler.input_buffer_size()];
    let chunk_frames = (input.len() / CHANNELS) as f64;
    let mut output = vec![T::widen(0.0); resampler.output_buffer_size()];

    let mut output_frames = 0usize;
    let mut worst: f64 = 0.0;
    for chunk in 1..=IMPULSE_AFTER_CHUNKS * 4 {
        // The first output frame not yet produced becomes available once this chunk has arrived.
        let wait = chunk as f64 * chunk_frames / input_rate - output_frames as f64 / output_rate;
        if chunk > IMPULSE_AFTER_CHUNKS {
            worst = worst.max(wait);
        }
        output_frames += resampler.process_chunk(&input, &mut output).unwrap() / CHANNELS;
    }
    worst * 1_000.0
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}

/// Measures impulse latency through [`RealtimeResampler`] in stream time, writing input frames at
/// the input cadence and reading one output frame per output period. Returns
/// `(latency in ms, underruns after priming)`.
fn measure_latency<T>(config: Config, chunk_frames: usize) -> (f64, usize)
where
    T: BenchSample,
{
    let input_rate = config.input_sample_rate;
    let output_rate = config.output_sample_rate;
    let mut resampler = RealtimeResampler::<T>::new(config).expect("valid config");

    let impulse_frame = chunk_frames * IMPULSE_AFTER_CHUNKS + chunk_frames / 3;
    let total_input_frames = impulse_frame + chunk_frames * 6;
    let mut input = vec![T::widen(0.0); total_input_frames * CHANNELS];
    input[impulse_frame * CHANNELS] = T::widen(1.0);

    let total_output_frames = total_input_frames * output_rate / input_rate;
    let mut frame = [T::widen(0.0); CHANNELS];
    let mut written_frames = 0;
    let mut primed = false;
    let mut underruns = 0;
    let mut peak = (0usize, 0.0f64);

    for out_frame in 0..total_output_frames {
        // Every input frame whose timestamp is at or before this output frame has arrived.
        let due = (out_frame * input_rate / output_rate + 1).min(total_input_frames);
        if due > written_frames {
            resampler
                .write_samples(&input[written_frames * CHANNELS..due * CHANNELS])
                .unwrap();
            written_frames = due;
        }

        let read = resampler.read_samples(&mut frame).unwrap();
        if read < CHANNELS {
            if primed {
                underruns += 1;
            }
            continue;
        }
        primed = true;
        let magnitude = frame[0].narrow().abs();
        if magnitude > peak.1 {
            peak = (out_frame, magnitude);
        }
    }

    let latency_s = peak.0 as f64 / output_rate as f64 - impulse_frame as f64 / input_rate as f64;
    (latency_s * 1_000.0, underruns)
}

fn measure<T>(preset: &Preset, sample_type: &'static str, input_rate: usize, output_rate: usize) -> Option<Row>
where
    T: BenchSample,
{
    let config = config_for(preset, input_rate, output_rate);
    if InterleavedResampler::<T>::new(config.clone()).is_err() {
        return None;
    }

    let (chunk_frames, durations) = time_chunks::<T>(config.clone());
    let min_latency_ms = min_pipeline_latency::<T>(config.clone());
    let (latency_ms, underruns) = measure_latency::<T>(config, chunk_frames);

    Some(Row {
        preset: preset.name,
        sample_type,
        input_rate,
        output_rate,
        chunk_frames,
        chunk_ms: chunk_frames as f64 / input_rate as f64 * 1_000.0,
        latency_ms,
        min_latency_ms,
        underruns,
        median_us: percentile(&durations, 0.5),
        p99_us: percentile(&durations, 0.99),
        max_us: *durations.last().unwrap(),
    })
}

fn khz(rate: usize) -> String {
    let khz = rate as f64 / 1_000.0;
    if khz.fract() == 0.0 {
        format!("{khz:.0}k")
    } else {
        format!("{khz}k")
    }
}

fn print_header() {
    println!(
        "| Preset | Type | Scenario | Chunk (frames) | Chunk (ms) | Latency (ms) | Min latency (ms) | Compute median (µs) | Compute p99 (µs) | Compute max (µs) | Load (%) |"
    );
    println!("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
}

fn print_row(row: &Row) {
    let underruns = if row.underruns > 0 {
        format!(" ({} underruns)", row.underruns)
    } else {
        String::new()
    };
    println!(
        "| {} | `{}` | {} → {} | {} | {:.2} | {:.2}{} | {:.2} | {:.1} | {:.1} | {:.1} | {:.2} |",
        row.preset,
        row.sample_type,
        khz(row.input_rate),
        khz(row.output_rate),
        row.chunk_frames,
        row.chunk_ms,
        row.latency_ms,
        underruns,
        row.min_latency_ms,
        row.median_us,
        row.p99_us,
        row.max_us,
        row.load_pct(),
    );
}

/// Prints the chunk size and latency floor at `quality = FLOOR_QUALITY` for each rate pair.
fn print_latency_floor() {
    const FLOOR_QUALITY: usize = 64;
    println!();
    println!("Latency floor at quality={FLOOR_QUALITY} (`f64`):");
    println!();
    println!("| Scenario | Chunk (frames) | Chunk (ms) | Latency (ms) | Min latency (ms) |");
    println!("| --- | ---: | ---: | ---: | ---: |");
    for &(input_rate, output_rate) in RATE_PAIRS {
        let config = PRESET_FAST
            .with_quality(FLOOR_QUALITY)
            .with_input_rate(input_rate)
            .with_output_rate(output_rate)
            .with_channels(CHANNELS);
        let Ok(resampler) = InterleavedResampler::<f64>::new(config.clone()) else {
            continue;
        };
        let chunk_frames = resampler.input_buffer_size() / CHANNELS;
        let (latency_ms, _) = measure_latency::<f64>(config.clone(), chunk_frames);
        println!(
            "| {} → {} | {} | {:.2} | {:.2} | {:.2} |",
            khz(input_rate),
            khz(output_rate),
            chunk_frames,
            chunk_frames as f64 / input_rate as f64 * 1_000.0,
            latency_ms,
            min_pipeline_latency::<f64>(config),
        );
    }
}

fn main() {
    // `cargo bench` passes `--bench`; ignore all arguments.
    if cfg!(feature = "rayon") {
        panic!("disable the rayon feature: it parallelizes channels and skews per-chunk timings");
    }

    print_header();
    for preset in presets() {
        for &(input_rate, output_rate) in RATE_PAIRS {
            if preset.name == "high+dd" {
                #[cfg(feature = "high_precision")]
                if let Some(row) = measure::<ardftsrc::DoubleDouble>(&preset, "DoubleDouble", input_rate, output_rate) {
                    print_row(&row);
                }
                continue;
            }
            // f32 is only recommended for the fast preset.
            if preset.name == "fast"
                && let Some(row) = measure::<f32>(&preset, "f32", input_rate, output_rate)
            {
                print_row(&row);
            }
            if let Some(row) = measure::<f64>(&preset, "f64", input_rate, output_rate) {
                print_row(&row);
            }
        }
    }

    print_latency_floor();
}
