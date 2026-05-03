// rubato: sync, f32, FixedSync, chunk-size=512, sub-chunks=1, FixedSync::Both

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};
use wavers::{Wav, read};

const BENCH_CHUNK_SIZE: usize = 512;
const BENCH_SUB_CHUNKS: usize = 1;
const BENCH_FIXED_SYNC: FixedSync = FixedSync::Input;
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
    frame_count: usize,
    planar_samples: Vec<Vec<f32>>,
}

fn read_wav_f32(path: &Path) -> WavData {
    let wav =
        Wav::<f32>::from_path(path).unwrap_or_else(|err| panic!("failed to open fixture {}: {err}", path.display()));
    let (samples, _) =
        read::<f32, _>(path).unwrap_or_else(|err| panic!("failed to read samples {}: {err}", path.display()));
    let channels = wav.n_channels() as usize;
    let sample_vec = samples.to_vec();

    WavData {
        name: path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("fixture")
            .to_string(),
        sample_rate_hz: wav.sample_rate() as usize,
        channels,
        frame_count: sample_vec.len() / channels,
        planar_samples: interleaved_to_planar(&sample_vec, channels),
    }
}

fn load_fixtures() -> Vec<WavData> {
    FIXTURE_PATHS
        .iter()
        .map(|rel_path| read_wav_f32(Path::new(rel_path)))
        .collect()
}

fn interleaved_to_planar(samples: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let frame_count = samples.len() / channels;
    let mut planar = (0..channels)
        .map(|_| Vec::with_capacity(frame_count))
        .collect::<Vec<_>>();
    for frame in samples.chunks_exact(channels) {
        for (channel, sample) in frame.iter().enumerate() {
            planar[channel].push(*sample);
        }
    }
    planar
}

fn benchmark_process_all(c: &mut Criterion, fixtures: &[WavData]) {
    let mut group = c.benchmark_group("process_all");
    for fixture in fixtures {
        for (target_sample_rate_hz, target_label) in TARGET_SAMPLE_RATES {
            let mut resampler = Fft::<f32>::new(
                fixture.sample_rate_hz,
                *target_sample_rate_hz,
                BENCH_CHUNK_SIZE,
                BENCH_SUB_CHUNKS,
                fixture.channels,
                BENCH_FIXED_SYNC,
            )
            .unwrap();
            let output_frames = resampler.process_all_needed_output_len(fixture.frame_count);
            let output_samples = output_frames * fixture.channels;
            let input =
                SequentialSliceOfVecs::new(&fixture.planar_samples, fixture.channels, fixture.frame_count).unwrap();
            let mut output_data = vec![vec![0.0_f32; output_frames]; fixture.channels];
            group.throughput(Throughput::Elements(output_samples as u64));
            let bench_id = BenchmarkId::new(&fixture.name, format!("to_{target_label}"));
            group.bench_with_input(bench_id, fixture, |b, wav| {
                b.iter(|| {
                    resampler.reset();
                    let mut output =
                        SequentialSliceOfVecs::new_mut(&mut output_data, wav.channels, output_frames).unwrap();
                    resampler
                        .process_all_into_buffer(&input, &mut output, wav.frame_count, None)
                        .unwrap();
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
