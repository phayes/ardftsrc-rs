# ardftsrc-rs

A rust implementation of the Real Discrete Fourier Transform Sample Rate Conversion (ARDFTSRC) algorithm.

`ardftsrc-rs` is a streaming audio sample-rate converter for interleaved audio streams, and is appropriate for both realtime and offline resampling. 

Generally `ardftsrc-rs` is preferred over other resamplers when quality is paramount.  Although it is generic over both `f32` and `f64`, it is highly recommended to use it with `f64`, even when processing an `f32` audio stream. 

It is more compute and memory intensive than other resamplers, so consider [rubato](https://crates.io/crates/rubato) if you want more efficiency. 

## Quick Start

Use `process_all` to resample a complete interleaved audio stream.

```rust
use ardftsrc::{Ardftsrc, PRESET_HIGH};

fn resample_all(input: &[f32], in_rate: usize, out_rate: usize, channels: usize) -> Vec<f32> {
    // When using a preset other than "FAST", f64 processing is preferred.
    let input_f64: Vec<f64> = input.iter().map(|v| *v as f64).collect();

    let config = PRESET_HIGH
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = Ardftsrc::<f64>::new(config).unwrap();

    let output = resampler.process_all(&input_f64).unwrap();

    // Convert back to the original f32
    output.into_iter().map(|v| v as f32).collect()
}
```

## Streaming Example

```rust
use ardftsrc::{Ardftsrc, PRESET_GOOD};

fn resample_streaming(input: Vec<f32>, in_rate: usize, out_rate: usize, channels: usize) -> Vec<f32> {
    // When using a preset other than "FAST", f64 processing is preferred.
    let input_f64: Vec<f64> = input.into_iter().map(|v| v as f64).collect();

    let config = PRESET_GOOD
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = Ardftsrc::<f64>::new(config).unwrap();

    // Get the input and output buffer sizes
    // You must read and write in these buffer sizes
    let input_buffer_size = resampler.input_buffer_size();
    let output_buffer_size = resampler.output_buffer_size();
    let mut out_buf = vec![0.0_f64; output_buffer_size];
    let mut out_f64 = Vec::<f64>::new();
    let mut offset = 0;

    while offset + input_buffer_size <= input_f64.len() {
        let chunk = &input_f64[offset..offset + input_buffer_size];
        let written = resampler.process_chunk(chunk, &mut out_buf).unwrap();
        out_f64.extend_from_slice(&out_buf[..written]);
        offset += input_buffer_size;
    }

    // The final chunk can be undersized (or even zero sized)
    let final_chunk = &input_f64[offset..];
    let written = resampler.process_chunk_final(final_chunk, &mut out_buf).unwrap();
    out_f64.extend_from_slice(&out_buf[..written]);

    // After processing the final chunk, you must call "finalize()" to get tail content.
    // finalize() also resets the resampler instance so it can be used again.
    let written = resampler.finalize(&mut out_buf).unwrap();
    out_f64.extend_from_slice(&out_buf[..written]);

    out_f64.into_iter().map(|v| v as f32).collect()
}
```

## Gapless Context

For adjacent tracks, you can set edge context before processing:

- `pre(...)`: tail samples from the previous track
- `post(...)`: head samples from the next track

Both buffers must be interleaved and channel-aligned.

## Presets

Presets are pre-vetted `Config` for various quality levels. 

```rust
let config = ardftsrc::PRESET_GOOD.with_input_rate(44_100).with_output_rate(48_000).with_channels(2);
```

| Preset           |                             Parameters | Recommended use                                                                                                  | Performance  |
| ---------------- | -------------------------------------: | ---------------------------------------------------------------------------------------------------------------- | ------------ |
| `PRESET_FAST`    | `quality=512` `bandwidth=0.8323`       | Fast preset for realtime workloads. Prefer a sinc resampler such as [`rubato`](https://crates.io/crates/rubato). | TODO         |
| `PRESET_GOOD`    | `quality=2048` `bandwidth=0.95`        | Balanced preset for realtime quality.                                                                            | TODO         |
| `PRESET_HIGH`    | `quality=65536` `bandwidth=0.97`       | High quality for offline or quality-focused realtime use.                                                        | TODO         |
| `PRESET_EXTREME` | `quality=524288` `bandwidth=0.9932`    | Maximum quality, intended for offline use.                                                                       | TODO         |


## Feature Flags

| Flag        | Enables                                                     | Default |
| ----------- | ----------------------------------------------------------- | ------- |
| `batch`     | Parallel APIs via Rayon: `batch(...)`, `batch_gapless(...)` | No      |
| `avx`       | `realfft` AVX backend                                       | No      |
| `sse`       | `realfft` SSE backend                                       | No      |
| `neon`      | `realfft` NEON backend                                      | No      |
| `wasm_simd` | `realfft` WebAssembly SIMD backend                          | No      |


## API Notes

- Buffers are interleaved by channel.
- Non-final `process_chunk(...)` input length must equal `input_buffer_size()`.
- `process_chunk_final(...)` accepts the trailing partial chunk (or empty slice).
- Call `finalize(...)` exactly once per stream to emit delayed tail samples.

## TODOs:

1. ardftsrc-rs has pathological RSS metrics. It's doing something that's making the allocator very unhappy. Need to track this down. 
2. Add support for `phase` config.
3. Refactor internal API so that `process_all` can do parallel processing via rayon. We can make this work by re-using `pre()` and `post()` for intra-track parallel processing. Will require significant internal API refactor (but perhaps worth it).
4. Calc performance metrics and post link
5. Investigate moving to a an [audioadapter](https://docs.rs/audioadapter/latest/audioadapter/) based interface, instead of always assuming interleaved.
6. Refactor `Ardftsrc` struct into `ArdftsrcCore` (single channel only, streaming) and `Ardftsrc` (top level orchestrator), do this in combination with todo#3. We could parallelize over both channels and within a single track, while organizing the internal API into a cleaner shape. 