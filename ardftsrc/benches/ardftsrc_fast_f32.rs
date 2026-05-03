use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use ardftsrc::{Ardftsrc, PRESET_FAST};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use wavers::{Wav, read};

const TARGET_SAMPLE_RATES: &[(usize, &str)] = &[(22_050, "22k05"), (48_000, "48k"), (96_000, "96k")];
const FIXTURE_PATHS: &[&str] = &[
    "../test_wavs/example-pcm16-44k1-stereo.wav",
    "../test_wavs/sweep-pcm16-22k05.wav",
    "../test_wavs/sweep-f32-96k.wav",
];
const INTER_TEST_SLEEP: Duration = Duration::from_millis(100);

#[derive(Clone)]
struct WavData {
    name: String,
    sample_rate_hz: usize,
    channels: usize,
    samples: Vec<f32>,
}

fn read_wav_f32(path: &Path) -> WavData {
    let wav =
        Wav::<f32>::from_path(path).unwrap_or_else(|err| panic!("failed to open fixture {}: {err}", path.display()));
    let (samples, _) =
        read::<f32, _>(path).unwrap_or_else(|err| panic!("failed to read samples {}: {err}", path.display()));

    WavData {
        name: path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("fixture")
            .to_string(),
        sample_rate_hz: wav.sample_rate() as usize,
        channels: wav.n_channels() as usize,
        samples: samples.to_vec(),
    }
}

fn load_fixtures() -> Vec<WavData> {
    FIXTURE_PATHS
        .iter()
        .map(|rel_path| read_wav_f32(Path::new(rel_path)))
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

fn benchmark_process_all(c: &mut Criterion, fixtures: &[WavData]) {
    let mut group = c.benchmark_group("fast");
    for fixture in fixtures {
        for (target_sample_rate_hz, target_label) in TARGET_SAMPLE_RATES {
            let config = PRESET_FAST
                .with_input_rate(fixture.sample_rate_hz)
                .with_output_rate(*target_sample_rate_hz)
                .with_channels(fixture.channels);
            let mut resampler: Ardftsrc<f32> = Ardftsrc::new(config).unwrap();
            let input_frames = fixture.samples.len() / fixture.channels;
            let output_frames = resampler.expected_output_size(input_frames);
            let output_samples = output_frames * fixture.channels;
            group.throughput(Throughput::Elements(output_samples as u64));
            let bench_id = BenchmarkId::new(&fixture.name, format!("to_{target_label}"));
            group.bench_with_input(bench_id, fixture, |b, wav| {
                b.iter(|| {
                    resampler.reset();
                    resampler.process_all(&wav.samples).unwrap();
                });
            });
            sleep(INTER_TEST_SLEEP);
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
