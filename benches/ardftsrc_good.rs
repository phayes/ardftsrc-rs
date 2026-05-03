// "Good" quality: f64, quality=2048, bandwidth=0.95

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use ardftsrc::{Ardftsrc, Config, TaperType};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use wavers::{Wav, read};

const BENCH_QUALITY: usize = 2048;
const BENCH_BANDWIDTH: f32 = 0.95;
const TARGET_SAMPLE_RATES: &[(usize, &str)] = &[(22_050, "22k05"), (48_000, "48k"), (96_000, "96k")];
const FIXTURE_PATHS: &[&str] = &[
    "test_wavs/example-pcm16-44k1-stereo.wav",
    "test_wavs/sweep-pcm16-22k05.wav",
    "test_wavs/sweep-f32-96k.wav",
];
const INTER_TEST_SLEEP: Duration = Duration::from_millis(100);

#[derive(Clone)]
struct WavData {
    name: String,
    sample_rate_hz: usize,
    channels: usize,
    samples: Vec<f64>,
}

fn read_wav_f64(path: &Path) -> WavData {
    let wav =
        Wav::<f64>::from_path(path).unwrap_or_else(|err| panic!("failed to open fixture {}: {err}", path.display()));
    let (samples, _) =
        read::<f64, _>(path).unwrap_or_else(|err| panic!("failed to read samples {}: {err}", path.display()));

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
        .map(|rel_path| read_wav_f64(Path::new(rel_path)))
        .collect()
}

fn benchmark_process_all(c: &mut Criterion, fixtures: &[WavData]) {
    let mut group = c.benchmark_group("process_all");
    for fixture in fixtures {
        for (target_sample_rate_hz, target_label) in TARGET_SAMPLE_RATES {
            let config = Config {
                input_sample_rate: fixture.sample_rate_hz,
                output_sample_rate: *target_sample_rate_hz,
                channels: fixture.channels,
                quality: BENCH_QUALITY,
                bandwidth: BENCH_BANDWIDTH,
                taper_type: TaperType::Cosine(3.45),
            };
            let mut resampler: Ardftsrc<f64> = Ardftsrc::new(config).unwrap();
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
    // Preload and decode all WAV fixtures before any timed benchmark loops.
    let fixtures = load_fixtures();
    benchmark_process_all(c, &fixtures);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
