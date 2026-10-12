mod common;

use std::hint::black_box;
use std::thread::sleep;
use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use ardftsrc::{Config, InterleavedResampler, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use common::{CHANNELS, INPUT_SAMPLE_RATES, OUTPUT_SAMPLE_RATES};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use num_traits::Float;
use realfft::FftNum;

const INTER_TEST_SLEEP: Duration = Duration::from_millis(100);

struct Preset {
    name: &'static str,
    config: Config,
}

const PRESETS: &[Preset] = &[
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
    input_sample_rate_hz: usize,
    preset: &Preset,
    output_sample_rate_hz: usize,
    widen: fn(f64) -> T,
) where
    T: Float + FftNum,
{
    let config = preset
        .config
        .clone()
        .with_input_rate(input_sample_rate_hz)
        .with_output_rate(output_sample_rate_hz)
        .with_channels(CHANNELS);
    let resampler = InterleavedResampler::<T>::new(config);
    if resampler.is_err() {
        return;
    }
    let mut resampler = resampler.expect("resampler construction checked above");
    let input_chunk_size = resampler.input_buffer_size();
    let input_chunk: Vec<T> = common::noise(input_chunk_size / CHANNELS)
        .into_iter()
        .map(widen)
        .collect();
    let mut output_chunk = vec![T::zero(); resampler.output_buffer_size()];

    group.bench_with_input(
        benchmark_id(
            sample_type_flag,
            input_sample_rate_hz,
            preset.name,
            output_sample_rate_hz,
            input_chunk_size / CHANNELS,
        ),
        &input_chunk,
        |b, input| {
            b.iter(|| {
                let written = resampler.process_chunk(input, &mut output_chunk).unwrap();
                black_box(written);
            });
        },
    );

    sleep(INTER_TEST_SLEEP);
}

fn benchmark_id(
    sample_type_flag: &str,
    input_sample_rate_hz: usize,
    preset_name: &str,
    output_sample_rate_hz: usize,
    input_chunk_frames: usize,
) -> BenchmarkId {
    BenchmarkId::from_parameter(format!(
        "{sample_type_flag} --preset={preset_name} --input={input_sample_rate_hz} --output={output_sample_rate_hz} --chunk-frames={input_chunk_frames} --channels={CHANNELS}"
    ))
}

fn benchmark_chunk_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("ardftsrc/chunk_time");

    for &input_sample_rate_hz in INPUT_SAMPLE_RATES {
        for preset in PRESETS {
            for &output_sample_rate_hz in OUTPUT_SAMPLE_RATES {
                benchmark_sample_type(
                    &mut group,
                    "--f32",
                    input_sample_rate_hz,
                    preset,
                    output_sample_rate_hz,
                    |sample| sample as f32,
                );
                benchmark_sample_type(
                    &mut group,
                    "--f64",
                    input_sample_rate_hz,
                    preset,
                    output_sample_rate_hz,
                    |sample| sample,
                );
            }
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    assert_supported_feature_combo();
    benchmark_chunk_time(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
