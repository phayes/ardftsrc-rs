# ARDFTSRC

A rust implementation of the Arbitrary Rate Discrete Fourier Transform Sample Rate Converter (ARDFTSRC) algorithm.

`ardftsrc` is a streaming audio sample-rate converter for interleaved audio streams, and is appropriate for both realtime and offline resampling. 

Generally `ardftsrc` is preferred over other resamplers when quality is paramount.  Although it is generic over both `f32` and `f64`, it is highly recommended to use it with `f64`, even when processing an `f32` audio stream. 

It is more compute and memory intensive than other resamplers, so consider [rubato](https://crates.io/crates/rubato) if you want more efficiency. 

## Input and output data format

This crate makes use of the [audioadapter](https://crates.io/crates/audioadapter) family of crates to help abstract the layout of input and output data. 

`ChunkResampler` and `BatchResampler` use an [`Adapter`](https://docs.rs/audioadapter/2.0.0/audioadapter/trait.Adapter.html) to accept input data. 

`ChunkResampler` uses [`AdapterMut`](https://docs.rs/audioadapter/2.0.0/audioadapter/trait.AdapterMut.html) for output, while `BatchResampler` returns `PlanarVecs` (which implements [`Adapter`](https://docs.rs/audioadapter/2.0.0/audioadapter/trait.Adapter.html) ).

A user of this crate should familiarize themselves with the conceptes in the [audioadapter](https://crates.io/crates/audioadapter) and [audioadapter-buffers](https://crates.io/crates/audioadapter-buffers) crates.

## Quick Start

Use `ChunkResampler::process_all` to resample a complete interleaved audio stream for a single track.

```rust
use ardftsrc::{ChunkResampler, PRESET_HIGH};
use audioadapter_buffers::direct::InterleavedSlice;

fn resample_all(input: &[f32], in_rate: usize, out_rate: usize, channels: usize) -> Vec<f32> {
    // When using a preset other than "FAST", f64 processing is preferred.
    let input_f64: Vec<f64> = input.iter().map(|v| *v as f64).collect();

    let config = PRESET_HIGH
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = ChunkResampler::<f64>::new(config).unwrap();

    let input_adapter = InterleavedSlice::new(&input_f64, channels, input_f64.len() / channels).unwrap();
    let output = resampler.process_all(&input_adapter).unwrap();

    // Convert back to the original interleaved f32
    output.interleave().into_iter().map(|v| v as f32).collect()
}
```

## Chunk Resampling

Use chunk resampling when you can control both read and write buffer sizes. Query `input_chunk_size()` and `output_chunk_size()` and size your input and output slices to the sizes required. The chunk API is more efficient than the streaming API and is preferred when you are not doing live resampling.

1. `process_chunk(...)` for each chunk, input and output slice sizes should match `input_chunk_size()` and `output_chunk_size()`
2. Call `process_chunk_final(...)` for the final chunk, it can be undersized. 
3. `finalize(...)` must be called once per stream to emit delayed tail samples and reset stream state.

To end the stream early, you can always just stop and call `reset()` on the stream.

```rust
use ardftsrc::{ChunkResampler, PRESET_GOOD};
use audioadapter_buffers::direct::InterleavedSlice;

fn resample_chunked(input: Vec<f32>, in_rate: usize, out_rate: usize, channels: usize) -> Vec<f32> {
    // When using a preset other than "FAST", f64 processing is preferred.
    let input_f64: Vec<f64> = input.into_iter().map(|v| v as f64).collect();

    let config = PRESET_GOOD
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = ChunkResampler::<f64>::new(config).unwrap();

    // Get the input and output chunk sizes
    // You must read and write in these buffer sizes
    let input_chunk_size = resampler.input_chunk_size();
    let output_chunk_size = resampler.output_chunk_size();
    let input_chunk_frames = input_chunk_size / channels;
    let output_chunk_frames = output_chunk_size / channels;
    let mut out_buf = vec![0.0_f64; output_chunk_size];
    let mut out_f64 = Vec::<f64>::new();
    let mut offset = 0;

    // Process whole chunks in the size of input_chunk_size
    while offset + input_chunk_size <= input_f64.len() {
        let chunk = &input_f64[offset..offset + input_chunk_size];

        // Prepare the input and output adapters.
        let input_adapter = InterleavedSlice::new(chunk, channels, input_chunk_frames).unwrap();
        let mut output_adapter = InterleavedSlice::new_mut(&mut out_buf, channels, output_chunk_frames).unwrap();

        // Process the chunk
        let written = resampler.process_chunk(&input_adapter, &mut output_adapter).unwrap();

        // Process otuput
        out_f64.extend_from_slice(&out_buf[..written]);
        offset += input_chunk_size;
    }

    // The final chunk can be undersized (or even zero sized)
    let final_chunk = &input_f64[offset..];

    // Prepare input and output adapters
    let final_input_adapter = InterleavedSlice::new(final_chunk, channels, final_chunk.len() / channels).unwrap();
    let mut final_output_adapter = InterleavedSlice::new_mut(&mut out_buf, channels, output_chunk_frames).unwrap();

    // Process Output
    let written = resampler.process_chunk_final(&final_input_adapter, &mut final_output_adapter).unwrap();
    out_f64.extend_from_slice(&out_buf[..written]);

    // After processing the final chunk, you must call "finalize()" to get tail content.
    // finalize() also resets the resampler instance so it can be used again.
    let mut finalize_output_adapter = InterleavedSlice::new_mut(&mut out_buf, channels, output_chunk_frames).unwrap();
    let written = resampler.finalize(&mut finalize_output_adapter).unwrap();
    out_f64.extend_from_slice(&out_buf[..written]);

    // Convert back into f32
    out_f64.into_iter().map(|v| v as f32).collect()
}
```

### Gapless Context

For adjacent tracks, you can set edge context before processing:

- `pre(&dyn Adapter)`: tail frames from the previous track
- `post(&dyn Adapter)`: head frames from the next track

`post(...)` may be called any time while the current stream is still active, but it must be
set before `process_chunk_final(...)`.

This enables live gapless handoff: while track A is streaming, once track B is known you can
call `post(...)` on A with B's head samples so A's stop-edge uses real next-track context.

## Streaming Resampler

Use the streaming resampler for live resampling. It accepts interleaved sample slices of any size and buffers internally until enough input is available for the underlying chunk resampler.

1. Call `write_samples(...)` with any incoming input size and call `read_samples(...)` to drain available output.
2. For multichannel streams, samples must be written interleaved.
3. Call `new_span(input_sample_rate, channels)` when the input sample rate or channel count changes.
4. Before calling `new_span(...)` or `finalize_samples(...)`, the current span must be frame aligned.
5. Call `finalize_samples(...)` once at end-of-stream, then keep calling `read_samples(...)` until it returns `0`.

Expect bursty read behavior. `read_samples(...)` accepts any output buffer size.

### Spans

Streaming sources sometimes change format while they are still producing samples. For example, a playlist-like source may play one file at 44.1 kHz stereo and then another at 48 kHz mono. The streaming resampler models those format regions as spans. You can start a new span with `new_span()`. When a new span starts, writes go to the new span immediately, and reads continue draining the previous span first before switching to the next. 

Input spans and output spans are non-synchronous. After calling `new_span`, you should query `samples_left_in_span()` to see how many samples are left on the output side before the output will switch to a new span.

To end the stream early, you can always just stop and call `reset()` on the stream.

```rust
use ardftsrc::{PRESET_GOOD, StreamingResampler};

fn resample_streaming(span_1_input: Vec<f32>, span_2_input: Vec<f32>) -> Vec<f32> {
    // Span 1 is 44.1 kHz stereo. Span 2 is 48 kHz mono.
    // Both spans are resampled to the same 48 kHz output rate.
    assert!(span_1_input.len().is_multiple_of(2));

    let config = PRESET_GOOD
        .with_input_rate(44_100)
        .with_output_rate(48_000)
        .with_channels(2);

    let mut resampler = StreamingResampler::<f32>::new(config).unwrap();
    let mut output = Vec::<f32>::new();
    let mut read_sample = [0.0_f32; 1];

    // This intentionally writes one sample at a time. Larger slices are more efficient,
    // but single-sample writes are valid.
    for sample in span_1_input {
        resampler.write_samples(&[sample]).unwrap();

        let written = resampler.read_samples(&mut read_sample);
        output.extend_from_slice(&read_sample[..written]);

        if resampler.samples_left_in_span() == Some(0) {
            // New span detected, maybe switch channel count in output.
        }
    }

    // Starting a new span implicitly finalizes span 1. Span 1 must be frame aligned.
    resampler.new_span(48_000, 1).unwrap();

    for sample in span_2_input {
        resampler.write_samples(&[sample]).unwrap();

        let written = resampler.read_samples(&mut read_sample);
        output.extend_from_slice(&read_sample[..written]);

        if resampler.samples_left_in_span() == Some(0) {
            // New span detected, maybe switch channel count in output.
        }
    }

    // Finalization can produce delayed tail output, so keep reading until the stream is drained.
    resampler.finalize_samples().unwrap();
    loop {
        let written = resampler.read_samples(&mut read_sample);
        if written == 0 {
            break;
        }
        output.extend_from_slice(&read_sample[..written]);
    }

    output
}
```

## Batching

Use batching when you have multiple full tracks to convert with the same configuration.

- `BatchResampler::batch(...)`: processes each input as an independent stream (no context shared between tracks).
- `BatchResampler::batch_gapless(...)`: preserves adjacent-track context for gapless album-style playback.

Enable the `rayon` feature to parallelize work across tracks.

```rust
use ardftsrc::{BatchResampler, PRESET_GOOD, PlanarVecs};
use ardftsrc::audioadapter::Adapter;
use ardftsrc::audioadapter_buffers::direct::InterleavedSlice;

fn resample_tracks(
    inputs: &[&[f64]],
    in_rate: usize,
    out_rate: usize,
    channels: usize,
) -> Vec<PlanarVecs<f64>> {
    let config = PRESET_GOOD
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let driver = BatchResampler::<f64>::new(config).unwrap();
    let adapters = inputs
        .iter()
        .map(|input| InterleavedSlice::new(input, channels, input.len() / channels).unwrap())
        .collect::<Vec<_>>();
    let adapter_refs = adapters
        .iter()
        .map(|input| input as &dyn Adapter<'_, f64>)
        .collect::<Vec<_>>();

    // Independent tracks (podcasts, unrelated files, etc.).
    let _independent = driver.batch(&adapter_refs).unwrap();

    // Gapless sequence (album tracks played back-to-back).
    let gapless = driver.batch_gapless(&adapter_refs).unwrap();

    // Return one of the two results based on your use case.
    gapless
}
```

## Quality Tuning and Presets

ARDFTSRC is built for quality over speed, and despite supporting both `f32` and `f64` should almost always be run as `f64`. To resample `f32` audio, it is recommended to convert `f32` samples to `f64`, resample them using `Resampler<f64>`, then convert back to `f32`.

If you want better performance than what this project offers, consider using a sinc resampler such as [`rubato`](https://crates.io/crates/rubato).

Presets are pre-vetted `Config` for various quality levels. 

```rust
let config = ardftsrc::PRESET_GOOD
  .with_input_rate(44_100)
  .with_output_rate(48_000)
  .with_channels(2);
```

| Preset           |                             Parameters | Recommended use                                                                                                  | Quality Metrics  |
| ---------------- | -------------------------------------: | ---------------------------------------------------------------------------------------------------------------- | ---------------- |
| `PRESET_FAST`    | `quality=512` `bandwidth=0.8323`       | Fast preset for realtime workloads. Prefer a sinc resampler such as [`rubato`](https://crates.io/crates/rubato). | TODO             |
| `PRESET_GOOD`    | `quality=2048` `bandwidth=0.95`        | Balanced preset for realtime quality.                                                                            | TODO             |
| `PRESET_HIGH`    | `quality=65536` `bandwidth=0.97`       | High quality for offline or quality-focused realtime use.                                                        | TODO             |
| `PRESET_EXTREME` | `quality=524288` `bandwidth=0.9932`    | Maximum quality, intended for offline use.                                                                       | TODO             |


## Feature Flags

| Flag        | Enables                                                     | Default |
| ----------- | ----------------------------------------------------------- | ------- |
| `rayon`     | Parallel processing (channel and track parallelism)         | No      |
| `avx`       | FFT AVX SIMD                                                | Yes     |
| `sse`       | FFT SSE SIMD                                                | Yes     |
| `neon`      | FFT NEON SIMD    for ARM / Mac                              | Yes     |
| `wasm_simd` | FFT WebAssembly SIMD                                        | Yes     |

Runtime feature detection is in place for all SIMD except webassembly. 

## Command Line

The workspace includes a small utility cli, `ardftsrc-rs`, for WAV/FLAC sample-rate conversion. 

You can use this as a utlity, or use it to benchmark this project.

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/ardftsrc-rs --help
./target/release/ardftsrc-rs --input in.wav --output out.flac --output-rate 48000 --preset high
```

## Contributing

Contributions are welcome!

### Architectural Overview

At a high level there are two layers:

- `ArdftsrcCore<T>` is the core DSP engine. It owns FFT and runs the core ARDFTSRC algorithm. It is private.
- `ChunkResampler<T>` is the fixed-size chunk resampler. It owns one `ArdftsrcCore` per channel and routes incoming interleaved audio to channel-specific cores.
- `StreamingResampler<T>` wraps `ChunkResampler<T>` with arbitrary-size sample buffering for live resampling.
- `BatchResampler<T>` wraps `ChunkResampler<T>` for batching processing multiple files at once.

### Golden Hashes

The `golden_hashes` test validates resampler determinism against checked-in golden outputs in `test_wavs/golden_hashes.<arch>.json`. It is intended to catch unintended behavior changes.

Run it with:

```bash
cargo test -p ardftsrc --release --features=rayon golden_hashes -- --nocapture
```

To regenerate `test_wavs/golden_hashes.<arch>.json`:

```bash
rust-script scripts/generate_golden_hashes.rs
```

Updates to `test_wavs/golden_hashes.<arch>.json` are allowed, but only when accompanied by verifiable quality improvements demonstrated with the HydrogenAudio SRC test suite.


### AI Usage Policy

AI use is allowed for the following:

1. Code exploration and understanding
2. Generating tests
3. Creating normal code / function blocks as long as the code is then manually and carefully hand-edited by a human

### Development TODOs:

1. Use `mut_input_ref` and write input directly, removing `ChunkResampler::input_staging`, this will avoid a copy.  Similar pattern for pre and post (avoid an allocation here).
2. Add support for `phase` config.
3. Add `tanh` taper.
4. Calc performance metrics and post links
5. Add bindings to other languages, python, ts (wasm) etc.
6. Right now `StreamingResampler` does all it's processing on the main audio thread, investigate if this should be moved off-thread. Probably, we are allocating...
7. Investigate why audiodapater interface is SOOO SLLOOWWW. 
8. In the StreamingResampler, there could be some minor performance improvements when going from f32 -> f64 -> f32 by delaying conversion until there's enough to take advantage of autovectorization on the conversion. Converting one scalar at a time is inefficient. 