use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use ardftsrc::{Config, InterleavedResampler, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Float;
use realfft::FftNum;
use wavers::{Wav, read};

const TARGET_SAMPLE_RATES: &[usize] = &[22_050, 48_000, 96_000];
const FIXTURE_PATHS: &[&str] = &[
    "../test_wavs/example-pcm16-44k1-stereo.wav",
    "../test_wavs/sweep-pcm16-22k05.wav",
    "../test_wavs/sweep-f32-96k.wav",
];
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
    Preset {
        name: "extreme",
        config: PRESET_EXTREME,
    },
];

#[derive(Clone)]
struct WavData {
    file: String,
    sample_rate_hz: usize,
    channels: usize,
    samples_f32: Vec<f32>,
    samples_f64: Vec<f64>,
}

fn read_wav(path: &Path) -> WavData {
    let wav =
        Wav::<f32>::from_path(path).unwrap_or_else(|err| panic!("failed to open fixture {}: {err}", path.display()));
    let (samples, _) =
        read::<f32, _>(path).unwrap_or_else(|err| panic!("failed to read samples {}: {err}", path.display()));
    let samples_f32 = samples.to_vec();
    let samples_f64 = samples_f32.iter().map(|&sample| sample as f64).collect();

    WavData {
        file: path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .unwrap_or("fixture.wav")
            .to_string(),
        sample_rate_hz: wav.sample_rate() as usize,
        channels: wav.n_channels() as usize,
        samples_f32,
        samples_f64,
    }
}

fn load_fixtures() -> Vec<WavData> {
    FIXTURE_PATHS
        .iter()
        .map(|rel_path| read_wav(Path::new(rel_path)))
        .collect()
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
    fixture: &WavData,
    preset: &Preset,
    output_sample_rate_hz: usize,
    samples: &[T],
) where
    T: Float + FftNum + Send + Sync,
{
    let config = preset
        .config
        .clone()
        .with_input_rate(fixture.sample_rate_hz)
        .with_output_rate(output_sample_rate_hz)
        .with_channels(fixture.channels);
    let resampler = InterleavedResampler::<T>::new(config);
    if resampler.is_err() {
        return;
    }
    let mut resampler = resampler.expect("resampler construction checked above");
    let input_frames = samples.len() / fixture.channels;
    let output_frames = resampler.expected_output_size(input_frames);
    let output_samples = output_frames * fixture.channels;

    group.throughput(Throughput::Elements(output_samples as u64));
    group.bench_with_input(
        benchmark_id(sample_type_flag, fixture, preset.name, output_sample_rate_hz),
        samples,
        |b, input| {
            b.iter(|| {
                resampler.reset();
                resampler.process_all(input).unwrap();
            });
        },
    );

    sleep(INTER_TEST_SLEEP);
}

fn benchmark_id(
    sample_type_flag: &str,
    fixture: &WavData,
    preset_name: &str,
    output_sample_rate_hz: usize,
) -> BenchmarkId {
    BenchmarkId::from_parameter(format!(
        "{sample_type_flag} --preset={preset_name} --input={} --output={output_sample_rate_hz} --file={}",
        fixture.sample_rate_hz, fixture.file
    ))
}

fn benchmark_process_all(c: &mut Criterion, fixtures: &[WavData]) {
    let mut group = c.benchmark_group("ardftsrc/process_all");

    for fixture in fixtures {
        for preset in PRESETS {
            for &output_sample_rate_hz in TARGET_SAMPLE_RATES {
                benchmark_sample_type(
                    &mut group,
                    "--f32",
                    fixture,
                    preset,
                    output_sample_rate_hz,
                    &fixture.samples_f32,
                );
                benchmark_sample_type(
                    &mut group,
                    "--f64",
                    fixture,
                    preset,
                    output_sample_rate_hz,
                    &fixture.samples_f64,
                );
            }
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    assert_supported_feature_combo();
    // Preload and decode all WAV fixtures before any timed benchmark loops.
    let fixtures = load_fixtures();
    benchmark_process_all(c, &fixtures);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
