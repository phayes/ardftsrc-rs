// rubato: sync, FixedSync::Input, chunk-size=512, sub-chunks=1

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use audioadapter_buffers::direct::InterleavedSlice;
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rubato::{Fft, FixedSync, Resampler, Sample};
use wavers::{Wav, read};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const BENCH_CHUNK_SIZE: usize = 512;
const BENCH_SUB_CHUNKS: usize = 1;
const BENCH_FIXED_SYNC: FixedSync = FixedSync::Input;
const TARGET_SAMPLE_RATES: &[usize] = &[22_050, 48_000, 96_000];
const FIXTURE_PATHS: &[&str] = &[
    "../test_wavs/example-pcm16-44k1-stereo.wav",
    "../test_wavs/sweep-pcm16-22k05.wav",
    "../test_wavs/sweep-f32-96k.wav",
];
const INTER_TEST_SLEEP: Duration = Duration::from_millis(100);

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

fn benchmark_sample_type<T>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sample_type_flag: &str,
    fixture: &WavData,
    output_sample_rate_hz: usize,
    samples: &[T],
) where
    T: Sample,
{
    let mut resampler = Fft::<T>::new(
        fixture.sample_rate_hz,
        output_sample_rate_hz,
        BENCH_CHUNK_SIZE,
        BENCH_SUB_CHUNKS,
        fixture.channels,
        BENCH_FIXED_SYNC,
    )
    .unwrap();
    let input_frames = samples.len() / fixture.channels;
    let input = InterleavedSlice::new(samples, fixture.channels, input_frames).unwrap();
    let output_frames = resampler.process_all_needed_output_len(input_frames);
    let output_samples = output_frames * fixture.channels;
    let mut output_data = vec![T::coerce(0.0); output_samples];

    group.throughput(Throughput::Elements(output_samples as u64));
    group.bench_function(benchmark_id(sample_type_flag, fixture, output_sample_rate_hz), |b| {
        b.iter(|| {
            resampler.reset();
            let mut output = InterleavedSlice::new_mut(&mut output_data, fixture.channels, output_frames).unwrap();
            resampler
                .process_all_into_buffer(&input, &mut output, input_frames, None)
                .unwrap();
        });
    });

    sleep(INTER_TEST_SLEEP);
}

fn benchmark_id(sample_type_flag: &str, fixture: &WavData, output_sample_rate_hz: usize) -> BenchmarkId {
    BenchmarkId::from_parameter(format!(
        "{sample_type_flag} --input={} --output={output_sample_rate_hz} --file={}",
        fixture.sample_rate_hz, fixture.file
    ))
}

fn benchmark_process_all(c: &mut Criterion, fixtures: &[WavData]) {
    let mut group = c.benchmark_group("rubato/process_all");

    for fixture in fixtures {
        for &output_sample_rate_hz in TARGET_SAMPLE_RATES {
            benchmark_sample_type(
                &mut group,
                "--f32",
                fixture,
                output_sample_rate_hz,
                &fixture.samples_f32,
            );
            benchmark_sample_type(
                &mut group,
                "--f64",
                fixture,
                output_sample_rate_hz,
                &fixture.samples_f64,
            );
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
