# ARDFTSRC

A rust implementation of the Arbitrary Rate Discrete Fourier Transform Sample Rate Converter (ARDFTSRC) algorithm.

`ardftsrc` is a streaming audio sample-rate converter for interleaved audio streams, and is appropriate for both realtime and offline resampling. 

Generally `ardftsrc` is preferred over other resamplers when quality is paramount.  Although it is generic over both `f32` and `f64`, it is highly recommended to use it with `f64`, even when processing an `f32` audio stream. 

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

## Chunk Streaming

There are two streaming modes, chunk based and sample based. 

Use chunk streaming when you can control both read and write buffer sizes. Query `input_chunk_size()` and `output_chunk_size()` and size your input and output slices to the sizes required. The chunk API is more efficient and is preferred when you are able to control the buffer sizes.

1. `process_chunk(...)` for each chunk, input and output slice sizes should match `input_chunk_size()` and `output_chunk_size()`
2. Call `process_chunk_final(...)` for the final chunk, it can be undersized. 
3. `finalize(...)` must be called once per stream to emit delayed tail samples and reset stream state.

To end the stream early, you can always just stop and call `reset()` on the stream.

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

    // Get the input and output chunk sizes
    // You must read and write in these buffer sizes
    let input_chunk_size = resampler.input_chunk_size();
    let output_chunk_size = resampler.output_chunk_size();
    let mut out_buf = vec![0.0_f64; output_chunk_size];
    let mut out_f64 = Vec::<f64>::new();
    let mut offset = 0;

    while offset + input_chunk_size <= input_f64.len() {
        let chunk = &input_f64[offset..offset + input_chunk_size];
        let written = resampler.process_chunk(chunk, &mut out_buf).unwrap();
        out_f64.extend_from_slice(&out_buf[..written]);
        offset += input_chunk_size;
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

## Sample Streaming

Use sample streaming when you do not control buffer sizes. This API supports arbitrary input lengths (including single frames / samples), and handles internal chunking for you.

1. Call `write_samples(...)` with any incoming input size and call `read_samples(...)` to drain available output.
2. For multichannel streams, samples must be written interleaved.
3. Before calling `finalize_samples(...)` all previously written samples must be frame aligned.
3. Call `finalize_samples(...)` once at end-of-stream, then keep calling `read_samples(...)` until it returns `0`.

Expect bursty read behavior when writing small numbers of samples at a time. To end the stream early, you can always just stop and call `reset()` on the stream.

```rust
use ardftsrc::{Ardftsrc, PRESET_GOOD};

fn resample_sample_streaming(
    input: &[f32],
    in_rate: usize,
    out_rate: usize,
    channels: usize,
) -> Vec<f32> {
    let config = PRESET_GOOD
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let mut resampler = Ardftsrc::<f32>::new(config).unwrap();
    let mut output = Vec::<f32>::new();
    let mut read_buf = vec![0.0_f32; resampler.output_chunk_size()];

    // write_samples accepts interleaved input of any length.
    let mut offset = 0;
    let write_step = channels * 32;
    while offset < input.len() {
        let end = (offset + write_step).min(input.len());
        resampler.write_samples(&input[offset..end]).unwrap();
        offset = end;

        loop {
            let written = resampler.read_samples(&mut read_buf);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&read_buf[..written]);
        }
    }

    // Finalize stream tail and drain remaining output.
    resampler.finalize_samples().unwrap();
    loop {
        let written = resampler.read_samples(&mut read_buf);
        if written == 0 {
            break;
        }
        output.extend_from_slice(&read_buf[..written]);
    }

    output
}
```

## Gapless Context

For adjacent tracks, you can set edge context before processing:

- `pre(...)`: tail samples from the previous track
- `post(...)`: head samples from the next track

`post(...)` may be called any time while the current stream is still active, but it must be
set before `process_chunk_final(...)` (chunk API) or `finalize_samples()` (samples API).

This enables live gapless handoff: while track A is streaming, once track B is known you can
call `post(...)` on A with B's head samples so A's stop-edge uses real next-track context.

Both buffers must be interleaved and channel-aligned.

## Batching

Use batching when you have multiple full tracks to convert with the same configuration.

- `batch(...)`: processes each input as an independent stream (no context shared between tracks).
- `batch_gapless(...)`: preserves adjacent-track context for gapless album-style playback.

Both APIs accept `&[&[T]]` (a list of interleaved, channel-aligned tracks) and return `Vec<Vec<T>>` in the same order as input.

Enable the `rayon` feature to parallelize work across tracks.

```rust
use ardftsrc::{Ardftsrc, PRESET_GOOD};

fn resample_tracks(inputs: &[&[f64]], in_rate: usize, out_rate: usize, channels: usize) -> Vec<Vec<f64>> {
    let config = PRESET_GOOD
        .with_input_rate(in_rate)
        .with_output_rate(out_rate)
        .with_channels(channels);

    let driver = Ardftsrc::<f64>::new(config).unwrap();

    // Independent tracks (podcasts, unrelated files, etc.).
    let _independent = driver.batch(inputs).unwrap();

    // Gapless sequence (album tracks played back-to-back).
    let gapless = driver.batch_gapless(inputs).unwrap();

    // Return one of the two results based on your use case.
    gapless
}
```

## Quality Tuning and Presets

ARDFTSRC is built for quality over speed, and despite supporting both `f32` and `f64` should almost always be run as `f64`. To resample `f32` audio, it is recommended to convert `f32` samples to `f64`, resample them using ARDFTSRC as `f64`, then convert back to `f32`. 

If you want speed over quality, consider using a sinc resampler such as [`rubato`](https://crates.io/crates/rubato).

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
| `rayon`     | Parallel processing (channel and track parallelism)         | No      |
| `avx`       | `realfft` AVX backend                                       | No      |
| `sse`       | `realfft` SSE backend                                       | No      |
| `neon`      | `realfft` NEON backend                                      | No      |
| `wasm_simd` | `realfft` WebAssembly SIMD backend                          | No      |


## TODOs:

1. ardftsrc-rs has pathological RSS metrics. It's doing something that's making the allocator very unhappy. Need to track this down. 
2. Add support for `phase` config.
3. Calc performance metrics and post link
4. Investigate moving to a an [audioadapter](https://docs.rs/audioadapter/latest/audioadapter/) based interface, instead of always assuming interleaved.
5. Add bindings to other languages, python, ts (wasm) etc. 

## Contributing

Contributions are welcome!

### Architectural Overview

At a high level there are two layers:

- `ArdftsrcCore<T>` is the core DSP engine. It owns FFT and runs the core ARDFTSRC algorithim. It is private.
- `Ardftsrc<T>` is the public orchestrator. It owns one `ArdftsrcCore` per channel, and routes incomming interleaved audio to channel-specific `ArdftsrcCore<T>` cores. 

Interaction model:

1. Caller uses `Ardftsrc` APIs (`process_chunk`, `write_samples`, `process_all`, etc.) with interleaved audio.
2. `Ardftsrc` maps that stream into per-channel slices and routes each slice to the corresponding `ArdftsrcCore`. There is one core per channel.
3. Each `ArdftsrcCore` advances independently (but in sync), then `Ardftsrc` combines channel outputs back into interleaved form.

### Golden Hashes

The `golden_hashes` test validates resampler determinism against checked-in golden outputs in `test_wavs/golden_hashes.<arch>.json`. It is intended to catch unintended behavior changes.

Run it with:

```bash
cargo test --release --features=rayon golden_hashes -- --nocapture
```

To regenerate `test_wavs/golden_hashes.<arch>.json`:

```bash
rust-script script/generate_golden_hashes.rs
```

Updates to `test_wavs/golden_hashes.<arch>.json` are allowed, but only when accompanied by verifiable quality improvements demonstrated with the HydrogenAudio SRC test suite.

### AI Usage Policy

AI use is allowed for the following:

1. Code exploration and understanding
2. Generating tests
3. Creating normal code / function blocks as long as the code is then manually and carefully hand-edited by a human
