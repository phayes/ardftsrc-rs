// "Fast" quality: f32, quality=512, bandwidth=0.95

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use ardftsrc::{Ardftsrc, Config, TaperType};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hound::{SampleFormat, WavReader};

const BENCH_QUALITY: usize = 512;
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
    samples: Vec<f32>,
}

fn read_wav_f32(path: &Path) -> WavData {
    let mut reader =
        WavReader::open(path).unwrap_or_else(|err| panic!("failed to open fixture {}: {err}", path.display()));
    let spec = reader.spec();
    let samples = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|err| panic!("failed to read float samples {}: {err}", path.display())),
        SampleFormat::Int => read_int_samples(&mut reader, spec.bits_per_sample, path),
    };

    WavData {
        name: path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("fixture")
            .to_string(),
        sample_rate_hz: spec.sample_rate as usize,
        channels: spec.channels as usize,
        samples,
    }
}

fn read_int_samples<R: std::io::Read>(reader: &mut WavReader<R>, bits_per_sample: u16, path: &Path) -> Vec<f32> {
    let scale = (1_i64 << (bits_per_sample.saturating_sub(1) as u32)) as f32;
    if bits_per_sample <= 16 {
        reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| sample as f32 / scale))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|err| panic!("failed to read int16 samples {}: {err}", path.display()))
    } else {
        reader
            .samples::<i32>()
            .map(|sample| sample.map(|sample| sample as f32 / scale))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|err| panic!("failed to read int32 samples {}: {err}", path.display()))
    }
}

fn load_fixtures() -> Vec<WavData> {
    FIXTURE_PATHS
        .iter()
        .map(|rel_path| read_wav_f32(Path::new(rel_path)))
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
    // Preload and decode all WAV fixtures before any timed benchmark loops.
    let fixtures = load_fixtures();
    benchmark_process_all(c, &fixtures);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
