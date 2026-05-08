# ARDFTSRC

[![Crates.io](https://img.shields.io/crates/v/ardftsrc.svg)](https://crates.io/crates/ardftsrc)
[![Docs.rs](https://docs.rs/ardftsrc/badge.svg)](https://docs.rs/ardftsrc)

A rust implementation of the Arbitrary Rate Discrete Fourier Transform Sample Rate Converter (ARDFTSRC) algorithm.

`ardftsrc` is a streaming audio sample-rate converter for interleaved audio streams, and is appropriate for both realtime and offline resampling. 

Generally `ardftsrc` is preferred over other resamplers when quality is paramount.  Although it is generic over both `f32` and `f64`, it is highly recommended to use it with `f64`, even when processing an `f32` audio stream. 

It is more compute intensive than other resamplers, so consider sinc [rubato](https://crates.io/crates/rubato) if you want more efficiency. See [PERFORMANCE.md](https://github.com/phayes/ardftsrc-rs/blob/master/PERFORMANCE.md) for a detailed speed and quality comparison vs rubato.

## Quick Start

Use `InterleavedResampler::process_all` to resample a complete interleaved audio stream for a single track.

```rust
use ardftsrc::{InterleavedResampler, PRESET_HIGH};

fn resample_all(input: &[f32], in_rate: usize, out_rate: usize, channels: usize) -> Vec<f32> {
    // When using a preset other than "FAST", f64 processing is preferred.
    let input_f64: Vec<f64> = input.iter().map(|v| *v as f64).collect();

    let config = PRESET_HIGH
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = InterleavedResampler::<f64>::new(config).unwrap();

    let output = resampler.process_all(&input_f64).unwrap();

    // Convert back to the original interleaved f32
    output.interleave().into_iter().map(|v| v as f32).collect()
}
```

## Chunk Resampling

Use chunk resampling when you can control both read and write buffer sizes. Query `input_buffer_size()` and `output_buffer_size()` and size your input and output slices to the sizes required. The chunk API is more efficient than the streaming API and is preferred when you are not doing live resampling.

There are two chunked resamplers depending on the shape of your audio:

1. `InterleavedResampler` - for interleaved audio
2. `PlanarResampler` - for planar audio. 

Internally ardftsrc uses planar representation, so `PlanarResampler` is more efficient, but if you're already working with interleaved audio, prefer `InterleavedResampler` since it has an optimized de-interleave / re-interleave path. Working with all chunked resamplers is the same:

1. Create the resampler with `let resampler = Resampler::new(config)`
2. Query the required input buffer size and output buffer size with `resampler.input_buffer_size()` and `resampler.output_buffer_size()`
3. Call  `process_chunk(...)` for each chunk, using the appropriate buffer sizes.
4. Call `process_chunk_final(...)` for the final chunk, it can be undersized. 
5. Finally, call `finalize(...)` once per stream to emit delayed tail samples and reset stream state.

To end the stream early, you may simply call `reset()`. 

```rust
use ardftsrc::{InterleavedResampler, PRESET_GOOD};

fn resample_chunked(input: Vec<f32>, in_rate: usize, out_rate: usize, channels: usize) -> Vec<f32> {
    // When using a preset other than "FAST", f64 processing is preferred.
    let input_f64: Vec<f64> = input.into_iter().map(|v| v as f64).collect();

    let config = PRESET_GOOD
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = InterleavedResampler::<f64>::new(config).unwrap();

    // Get the input and output chunk sizes
    // You must read and write in these buffer sizes
    let input_chunk_size = resampler.input_buffer_size();
    let output_chunk_size = resampler.output_buffer_size();
    let mut out_buf = vec![0.0_f64; output_chunk_size];
    let mut out_f64 = Vec::<f64>::new();
    let mut offset = 0;

    // Process whole chunks in the size of input_chunk_size
    while offset + input_chunk_size <= input_f64.len() {
        let chunk = &input_f64[offset..offset + input_chunk_size];

        // Process the chunk
        let written = resampler.process_chunk(chunk, &mut out_buf).unwrap();

        // Process output
        out_f64.extend_from_slice(&out_buf[..written]);
        offset += input_chunk_size;
    }

    // The final chunk can be undersized (or even zero sized)
    let final_chunk = &input_f64[offset..];

    // Process Output
    let written = resampler.process_chunk_final(final_chunk, &mut out_buf).unwrap();
    out_f64.extend_from_slice(&out_buf[..written]);

    // After processing the final chunk, you must call "finalize()" to get tail content.
    // finalize() also resets the resampler instance so it can be used again.
    let written = resampler.finalize(&mut out_buf).unwrap();
    out_f64.extend_from_slice(&out_buf[..written]);

    // Convert back into f32
    out_f64.into_iter().map(|v| v as f32).collect()
}
```

### Gapless Context

For adjacent tracks, you can set edge context before processing:

- `pre(Vec<T>)`: tail frames from the previous track
- `post(Vec<T>)`: head frames from the next track

`post(...)` may be called any time while the current stream is still active, but it must be
set before `process_chunk_final(...)`.

This enables live gapless handoff: while track A is streaming, once track B is known you can
call `post(...)` on A with B's head samples so A's stop-edge uses real next-track context.

## Realtime Resampler

Enable the `realtime` feature to use `RealtimeResampler` for live resampling. It accepts interleaved samples one-at-a-time and runs the chunk resampler on a worker thread.

1. Call `write_sample(...)` with each incoming interleaved sample and `read_sample(...)` at your output cadence.
2. For multichannel streams, samples must be written interleaved.
3. Call `new_span(input_sample_rate, channels)` when the input sample rate or channel count changes.
4. Call `finalize()` at end-of-stream, then keep calling `read_sample(...)` until it returns `None`.

`RealtimeResampler` has some startup delay and will emit `Some(-0.0)` (negative-zero silence) until the off-thread resampler
is warmed up and producing samples. You may do nothing (it will play as silence), or check (`x.is_zero() && x.is_sign_negative()`) for this specific circumstance.

### Spans

Streaming sources sometimes change format while they are still producing samples. For example, a playlist-like source may play one file at 44.1 kHz stereo and then another at 48 kHz mono. The realtime resampler models those format regions as spans. You can start a new span with `new_span()`. When a new span starts, writes go to the new span immediately, and reads continue draining the previous span first before switching to the next.

Input spans and output spans are non-synchronous. After calling `new_span`, query `current_span_len()` to see how many samples are left on the output side before the output will switch to a new span.

To end the stream early, stop writing input samples, call `finalize()`, then drain with `read_sample()` until it returns `None`.

```rust
 #[cfg(feature = "realtime")]
fn resample_streaming(span_1_input: Vec<f32>, span_2_input: Vec<f32>) -> Vec<f32> {
    use ardftsrc::{PRESET_GOOD, RealtimeResampler, StreamingConfig};

    // Span 1 is 44.1 kHz stereo. Span 2 is 48 kHz mono.
    // Both spans are resampled to the same 48 kHz output rate.
    assert!(span_1_input.len().is_multiple_of(2));

    let config = PRESET_GOOD
        .with_input_rate(44_100)
        .with_output_rate(48_000)
        .with_channels(2);

    let mut resampler = RealtimeResampler::<f32>::new(config, StreamingConfig::default());
    let mut output = Vec::<f32>::new();

    // This intentionally writes one sample at a time. Larger slices are more efficient,
    // but single-sample writes are valid.
    for sample in span_1_input {
        resampler.write_sample(sample);

        if let Some(sample) = resampler.read_sample() {
            output.push(sample);
        }

        if resampler.current_span_len() == Some(0) {
            // New span detected, maybe switch channel count in output.
        }
    }

    resampler.new_span(48_000, 1);

    for sample in span_2_input {
        resampler.write_sample(sample);

        if let Some(sample) = resampler.read_sample() {
            output.push(sample);
        }

        if resampler.current_span_len() == Some(0) {
            // New span detected, maybe switch channel count in output.
        }
    }

    // Finalization can produce delayed tail output, so keep reading until the stream is drained.
    resampler.finalize();
    while let Some(sample) = resampler.read_sample() {
        output.push(sample);
    }

    resampler.shutdown().unwrap();
    output
}
```

### Rodio integration

Enable the `rodio` feature to use `rodio::RodioResampler` to wrap a `rodio::Source` and resample it in realtime in your rodio pipeline.

- Basic rodio example: [`examples/rodio_adapter.rs`](https://github.com/phayes/ardftsrc-rs/blob/master/ardftsrc/examples/rodio_adapter.rs)
- Span-switching rodio example: [`examples/rodio_adapter_with_spans.rs`](https://github.com/phayes/ardftsrc-rs/blob/master/ardftsrc/examples/rodio_adapter_with_spans.rs)

## Batching

Use batching when you have multiple full tracks to convert with the same configuration.

- `InterleavedResampler::batch(...)`: processes each interleaved input as an independent stream (no context shared between tracks).
- `InterleavedResampler::batch_gapless(...)`: preserves adjacent-track context for gapless album-style playback.
- `PlanarResampler` exposes the same `batch(...)` and `batch_gapless(...)` APIs for already-planar inputs.

Enable the `rayon` feature to parallelize work across tracks.

```rust
use ardftsrc::{InterleavedResampler, PRESET_GOOD, PlanarVecs};

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

    let driver = InterleavedResampler::<f64>::new(config).unwrap();

    // Independent tracks (podcasts, unrelated files, etc.).
    let _independent = driver.batch(inputs).unwrap();

    // Gapless sequence (album tracks played back-to-back).
    let gapless = driver.batch_gapless(inputs).unwrap();

    // Return one of the two results based on your use case.
    gapless
}
```

## Quality Tuning and Presets

ARDFTSRC is built for quality over speed, and despite supporting both `f32` and `f64` should almost always be run as `f64`. To resample `f32` audio, it is recommended to convert `f32` samples to `f64`, resample them using `InterleavedResampler<f64>` or `PlanarResampler<f64>`, then convert back to `f32`.

If you want better performance than what this project offers, consider using a sinc resampler such as [`rubato`](https://crates.io/crates/rubato).

Presets are pre-vetted `Config` for various quality levels. 

```rust
let config = ardftsrc::PRESET_GOOD
  .with_input_rate(44_100)
  .with_output_rate(48_000)
  .with_channels(2);
```

| Preset           |                             Parameters | Recommended use                                            | Quality Metrics  |
| ---------------- | -------------------------------------: | -----------------------------------------------------------| ---------------- |
| `PRESET_FAST`    | `quality=512` `bandwidth=0.832`        | Fast preset for realtime workloads.                        | [f32](https://src.hydrogenaudio.org/compareresults?id1=c527356d-3566-46f8-8dea-dc2065b11e46&id2=0), [f64](https://src.hydrogenaudio.org/compareresults?id1=8e59a5bd-8147-470c-9501-44ab81718b8f&id2=0)|
| `PRESET_GOOD`    | `quality=1878` `bandwidth=0.911`       | Balanced preset for realtime quality.                      | [f64](https://src.hydrogenaudio.org/compareresults?id1=e12d7fe0-dfa2-4c49-bbdd-51c16a931cb5&id2=0)             |
| `PRESET_HIGH`    | `quality=73622` `bandwidth=0.987`      | High quality for offline use. | [f64](https://src.hydrogenaudio.org/compareresults?id1=43a72723-7f35-4318-bbd1-44cdfaa6df88&id2=0)             |
| `PRESET_EXTREME` | `quality=524514` `bandwidth=0.995`     | Maximum quality, intended for offline use.                 | [f64](https://src.hydrogenaudio.org/compareresults?id1=dbdbdd66-d8b8-4b8b-b217-b71162cb1f2f&id2=0)             |


## Feature Flags

| Flag           | Enables                                                                           | Default |
| -------------- | --------------------------------------------------------------------------------- | ------- |
| `audioadapter` | Experimental [`audioadapter`](https://crates.io/crates/audioadapter) support      | No      |
| `realtime`     | `RealtimeResampler` streaming API backed by lock-free ring buffers                | No      |
| `rayon`        | Parallel processing (channel and track parallelism)                               | No      |
| `rodio`        | [`rodio`](https://crates.io/crates/rodio) integration via `rodio::RodioResampler` | No      |
| `avx`          | FFT AVX SIMD                                                                      | Yes     |
| `sse`          | FFT SSE SIMD                                                                      | Yes     |
| `neon`         | FFT NEON SIMD for ARM / Mac                                                       | Yes     |
| `wasm_simd`    | FFT WebAssembly SIMD                                                              | Yes     |

Runtime feature detection is in place for all SIMD except webassembly. 
