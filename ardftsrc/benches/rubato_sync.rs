// rubato: sync, FixedSync::Input, chunk-size=512, sub-chunks=1

mod common;

use std::thread::sleep;
use std::time::Duration;

use audioadapter_buffers::direct::InterleavedSlice;
use criterion::measurement::WallTime;
use common::{CHANNELS, INPUT_SAMPLE_RATES, INPUT_SECONDS, OUTPUT_SAMPLE_RATES};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use mimalloc::MiMalloc;
use rubato::{Fft, FixedSync, Resampler, Sample};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const BENCH_CHUNK_SIZE: usize = 512;
const BENCH_SUB_CHUNKS: usize = 1;
const BENCH_FIXED_SYNC: FixedSync = FixedSync::Input;
const INTER_TEST_SLEEP: Duration = Duration::from_millis(100);

/// One input stream, in every sample type the bench runs.
struct Input {
    sample_rate_hz: usize,
    samples_f32: Vec<f32>,
    samples_f64: Vec<f64>,
}

fn make_input(sample_rate_hz: usize) -> Input {
    let samples_f64 = common::track(sample_rate_hz);
    Input {
        sample_rate_hz,
        samples_f32: samples_f64.iter().map(|&sample| sample as f32).collect(),
        samples_f64,
    }
}

fn benchmark_sample_type<T>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sample_type_flag: &str,
    input: &Input,
    output_sample_rate_hz: usize,
    samples: &[T],
) where
    T: Sample,
{
    let mut resampler = Fft::<T>::new(
        input.sample_rate_hz,
        output_sample_rate_hz,
        BENCH_CHUNK_SIZE,
        BENCH_SUB_CHUNKS,
        CHANNELS,
        BENCH_FIXED_SYNC,
    )
    .unwrap();
    let input_frames = samples.len() / CHANNELS;
    let input_slice = InterleavedSlice::new(samples, CHANNELS, input_frames).unwrap();
    let output_frames = resampler.process_all_needed_output_len(input_frames);
    let mut output_data = vec![T::coerce(0.0); output_frames * CHANNELS];

    // Elements are input frames, so elements per second divided by the input rate is the
    // realtime factor.
    group.throughput(Throughput::Elements(input_frames as u64));
    group.bench_function(benchmark_id(sample_type_flag, input, output_sample_rate_hz), |b| {
        b.iter(|| {
            resampler.reset();
            let mut output = InterleavedSlice::new_mut(&mut output_data, CHANNELS, output_frames).unwrap();
            resampler
                .process_all_into_buffer(&input_slice, &mut output, input_frames, None)
                .unwrap();
        });
    });

    sleep(INTER_TEST_SLEEP);
}

fn benchmark_id(sample_type_flag: &str, input: &Input, output_sample_rate_hz: usize) -> BenchmarkId {
    BenchmarkId::from_parameter(format!(
        "{sample_type_flag} --input={} --output={output_sample_rate_hz} --channels={CHANNELS} --seconds={INPUT_SECONDS}",
        input.sample_rate_hz
    ))
}

fn benchmark_process_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("rubato/process_all");
    // Each iteration resamples a whole track, so take few, flat samples.
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for &input_sample_rate_hz in INPUT_SAMPLE_RATES {
        // Generated per rate so only one rate's samples are held at a time.
        let input = make_input(input_sample_rate_hz);
        for &output_sample_rate_hz in OUTPUT_SAMPLE_RATES {
            benchmark_sample_type(
                &mut group,
                "--f32",
                &input,
                output_sample_rate_hz,
                &input.samples_f32,
            );
            benchmark_sample_type(
                &mut group,
                "--f64",
                &input,
                output_sample_rate_hz,
                &input.samples_f64,
            );
        }
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    benchmark_process_all(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
