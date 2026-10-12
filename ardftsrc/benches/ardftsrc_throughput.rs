mod common;

use std::thread::sleep;
use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use ardftsrc::{Config, InterleavedResampler, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, Sample};
use criterion::measurement::WallTime;
use common::{CHANNELS, INPUT_SAMPLE_RATES, INPUT_SECONDS, OUTPUT_SAMPLE_RATES};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
const INTER_TEST_SLEEP: Duration = Duration::from_millis(100);

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
    // Run with `--features high_precision` to include this preset, which runs `PRESET_HIGH` in
    // `DoubleDouble`.
    #[cfg(feature = "high_precision")]
    presets.push(Preset {
        name: "high+dd",
        config: PRESET_HIGH,
    });
    presets
}

/// One input stream, in every sample type the bench runs.
struct Input {
    sample_rate_hz: usize,
    samples_f32: Vec<f32>,
    samples_f64: Vec<f64>,
    #[cfg(feature = "high_precision")]
    samples_dd: Vec<ardftsrc::DoubleDouble>,
}

fn make_input(sample_rate_hz: usize) -> Input {
    let samples_f64 = common::track(sample_rate_hz);
    Input {
        sample_rate_hz,
        samples_f32: samples_f64.iter().map(|&sample| sample as f32).collect(),
        #[cfg(feature = "high_precision")]
        samples_dd: samples_f64
            .iter()
            .map(|&sample| ardftsrc::DoubleDouble::from(sample))
            .collect(),
        samples_f64,
    }
}

fn assert_supported_feature_combo() {
    let rayon_enabled = cfg!(feature = "rayon");
    let avx_enabled = cfg!(feature = "avx");
    let neon_enabled = cfg!(feature = "neon");
    let sse_enabled = cfg!(feature = "sse");
    let wasm_simd_enabled = cfg!(feature = "wasm_simd");
    let simd_present = avx_enabled || neon_enabled || sse_enabled || wasm_simd_enabled;

    if rayon_enabled || !simd_present {
        panic!(
            "unsupported bench feature combo: rayon is enabled or SIMD is not present \
             (rayon={rayon_enabled}, avx={avx_enabled}, neon={neon_enabled}, sse={sse_enabled}, wasm_simd={wasm_simd_enabled})"
        );
    }
}

fn benchmark_sample_type<T>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sample_type_flag: &str,
    input: &Input,
    preset: &Preset,
    output_sample_rate_hz: usize,
    samples: &[T],
) where
    T: Sample,
{
    let config = preset
        .config
        .clone()
        .with_input_rate(input.sample_rate_hz)
        .with_output_rate(output_sample_rate_hz)
        .with_channels(CHANNELS);
    let resampler = InterleavedResampler::<T>::new(config);
    if resampler.is_err() {
        return;
    }
    let mut resampler = resampler.expect("resampler construction checked above");
    // Elements are input frames, so elements per second divided by the input rate is the
    // realtime factor.
    group.throughput(Throughput::Elements((samples.len() / CHANNELS) as u64));
    group.bench_with_input(
        benchmark_id(sample_type_flag, input, preset.name, output_sample_rate_hz),
        samples,
        |b, samples| {
            b.iter(|| {
                resampler.reset();
                resampler.process_all(samples).unwrap();
            });
        },
    );

    sleep(INTER_TEST_SLEEP);
}

fn benchmark_id(
    sample_type_flag: &str,
    input: &Input,
    preset_name: &str,
    output_sample_rate_hz: usize,
) -> BenchmarkId {
    BenchmarkId::from_parameter(format!(
        "{sample_type_flag} --preset={preset_name} --input={} --output={output_sample_rate_hz} --channels={CHANNELS} --seconds={INPUT_SECONDS}",
        input.sample_rate_hz
    ))
}

fn benchmark_process_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("ardftsrc/process_all");
    // Each iteration resamples a whole track, so take few, flat samples.
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    let presets = presets();

    for &input_sample_rate_hz in INPUT_SAMPLE_RATES {
        // Generated per rate so only one rate's samples are held at a time.
        let input = make_input(input_sample_rate_hz);
        for preset in &presets {
            for &output_sample_rate_hz in OUTPUT_SAMPLE_RATES {
                if preset.name == "high+dd" {
                    #[cfg(feature = "high_precision")]
                    benchmark_sample_type(
                        &mut group,
                        "--dd",
                        &input,
                        preset,
                        output_sample_rate_hz,
                        &input.samples_dd,
                    );
                    continue;
                }
                benchmark_sample_type(
                    &mut group,
                    "--f32",
                    &input,
                    preset,
                    output_sample_rate_hz,
                    &input.samples_f32,
                );
                benchmark_sample_type(
                    &mut group,
                    "--f64",
                    &input,
                    preset,
                    output_sample_rate_hz,
                    &input.samples_f64,
                );
            }
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    assert_supported_feature_combo();
    benchmark_process_all(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
