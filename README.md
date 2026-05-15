# ARDFTSRC

[![Crates.io](https://img.shields.io/crates/v/ardftsrc.svg)](https://crates.io/crates/ardftsrc)
[![Docs.rs](https://docs.rs/ardftsrc/badge.svg)](https://docs.rs/ardftsrc)

A rust implementation of the Arbitrary Rate Discrete Fourier Transform Sample Rate Converter (ARDFTSRC) algorithm.

`ardftsrc` is a streaming audio sample-rate converter, and is appropriate for both realtime and offline resampling. 

Generally `ardftsrc` is preferred over other resamplers when quality is paramount.  Although it is generic over both `f32` and `f64`, it is highly recommended to use it with `f64`, even when processing an `f32` audio stream. 

It is among the top scoring resampling libraries on [HydrogenAudio Sample Rate Conversion Comparison](https://src.hydrogenaudio.org/), topping out with a balanced score of 99.73%.  It is more compute intensive than other resamplers, so consider sinc [rubato](https://crates.io/crates/rubato) if you want more efficiency. See [PERFORMANCE.md](https://github.com/phayes/ardftsrc-rs/blob/master/PERFORMANCE.md) for a detailed speed and quality comparison vs rubato.

## Quick Start

Use [`InterleavedResampler::process_all`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.InterleavedResampler.html#method.process_all) to resample a complete interleaved audio stream for a single track.

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

1. [`InterleavedResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.InterleavedResampler.html) - for interleaved audio
2. [`PlanarResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.PlanarResampler.html) - for planar audio. 

Internally ardftsrc uses planar representation, so [`PlanarResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.PlanarResampler.html) is more efficient, but if you're already working with interleaved audio, prefer [`InterleavedResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.InterleavedResampler.html) since it has an optimized de-interleave / re-interleave path. Working with all chunked resamplers is the same:

1. Create the resampler with `let resampler = Resampler::new(config)`
2. Query the required input buffer size and output buffer size with `resampler.input_buffer_size()` and `resampler.output_buffer_size()`
3. Call  `process_chunk(...)` for each chunk, using the appropriate buffer sizes.
4. Call `process_chunk_final(...)` for the final chunk, it can be undersized. 
5. Finally, call `finalize(...)` once per stream to emit delayed tail samples and reset stream state.

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

## Realtime Resampling

ardftsrc-rs provides both [rodio](https://crates.io/crates/rodio) integration via [`RodioResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.RodioResampler.html) (`rodio` feature) and the ability to build your own custom realtime audio resampling pipline via [`RealtimeResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.RealtimeResampler.html). 

## Rodio integration

Enable the `rodio` feature to use [`RodioResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.RodioResampler.html) to wrap a [`rodio::Source`](https://docs.rs/rodio/latest/rodio/source/trait.Source.html) and resample it in realtime in your rodio pipeline.

When playing from a buffered audio source such as a file or a buffered stream, it is recommended to use [`config.with_rodio_fast_start(true)`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.Config.html#method.with_rodio_fast_start), which will 
avoid initial output delay by pulling samples from the upstream source to prime the resampler. For very-realtime sources such as microphones or similar, 
do not enable fast-start.

```rust
#[cfg(feature = "rodio")]
{
    let stream = rodio::DeviceSinkBuilder::open_default_sink()?;
    let mixer = stream.mixer();

    let tone = rodio::source::SignalGenerator::new(
        NonZero::new(44_100 as u32).unwrap(),
        400, // 400 Hz
        rodio::source::Function::Sine,
    )
    .take_duration(Duration::from_secs(3.0));

    let config = PRESET_FAST.with_channels(1).with_input_rate(44_100).with_output_rate(48_000);
    let resampled_tone = RodioResampler::new(tone, config)?;

    mixer.add(resampled_tone);
    std::thread::sleep(Duration::from_secs(4));
}
```

More examples can be found:

- Basic rodio example: [`examples/rodio_adapter.rs`](https://github.com/phayes/ardftsrc-rs/blob/master/ardftsrc/examples/rodio_adapter.rs)
- Span-switching rodio example: [`examples/rodio_adapter_with_spans.rs`](https://github.com/phayes/ardftsrc-rs/blob/master/ardftsrc/examples/rodio_adapter_with_spans.rs)

## Batching

Use batching when you have multiple full tracks to convert with the same configuration.

- [`InterleavedResampler::batch(...)`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.InterleavedResampler.html#method.batch): processes each interleaved input as an independent stream (no context shared between tracks).
- [`InterleavedResampler::batch_gapless(...)`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.InterleavedResampler.html#method.batch_gapless): preserves adjacent-track context for gapless album-style playback.
- [`PlanarResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.PlanarResampler.html) exposes the same [`batch(...)`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.PlanarResampler.html#method.batch) and [`batch_gapless(...)`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.PlanarResampler.html#method.batch_gapless) APIs for already-planar inputs.

Enable the [`rayon` feature](https://docs.rs/crate/ardftsrc/latest/features) to parallelize work across tracks.

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

ARDFTSRC is built for quality over speed, and despite supporting both `f32` and `f64` should almost always be run as `f64`. To resample `f32` audio, it is recommended to convert `f32` samples to `f64`, resample them using [`InterleavedResampler<f64>`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.InterleavedResampler.html) or [`PlanarResampler<f64>`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.PlanarResampler.html), then convert back to `f32`.

If you want better performance than what this project offers, consider using a sinc resampler such as [`rubato`](https://crates.io/crates/rubato).

Presets are pre-vetted [`Config`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.Config.html) for various quality levels. 

```rust
let config = ardftsrc::PRESET_GOOD
  .with_input_rate(44_100)
  .with_output_rate(48_000)
  .with_channels(2);
```

| Preset                                                                                    |  Quality | Bandwidth | Recommended use                            | Quality metrics                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------- | -------: | --------: | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`PRESET_FAST`](https://docs.rs/ardftsrc/latest/ardftsrc/constant.PRESET_FAST.html)       |    `512` |   `0.832` | Fast preset for realtime workloads.        | [f32](https://src.hydrogenaudio.org/compareresults?id1=c527356d-3566-46f8-8dea-dc2065b11e46&id2=0), [f64](https://src.hydrogenaudio.org/compareresults?id1=8e59a5bd-8147-470c-9501-44ab81718b8f&id2=0) |
| [`PRESET_GOOD`](https://docs.rs/ardftsrc/latest/ardftsrc/constant.PRESET_GOOD.html) †       |   `1878` |   `0.911` | Balanced preset for realtime quality.      | [f64](https://src.hydrogenaudio.org/compareresults?id1=e12d7fe0-dfa2-4c49-bbdd-51c16a931cb5&id2=0)                                                                                                     |
| [`PRESET_HIGH`](https://docs.rs/ardftsrc/latest/ardftsrc/constant.PRESET_HIGH.html)       |  `73622` |   `0.987` | High quality for offline use.              | [f64](https://src.hydrogenaudio.org/compareresults?id1=43a72723-7f35-4318-bbd1-44cdfaa6df88&id2=0)                                                                                                     |
| [`PRESET_EXTREME`](https://docs.rs/ardftsrc/latest/ardftsrc/constant.PRESET_EXTREME.html) | `524514` |   `0.995` | Maximum quality, intended for offline use. | [f64](https://src.hydrogenaudio.org/compareresults?id1=dbdbdd66-d8b8-4b8b-b217-b71162cb1f2f&id2=0)                                                                                                     |

† You should probably use [`PRESET_GOOD`](https://docs.rs/ardftsrc/latest/ardftsrc/constant.PRESET_GOOD.html). It's fast,  has very high quality metrics, and has lower pre-ringing artefact as compared PRESET_HIGH and PRESET_EXTREME.

## Feature Flags

| Flag           | Enables                                                                           | Default |
| -------------- | --------------------------------------------------------------------------------- | ------- |
| `rodio`        | [`rodio`](https://crates.io/crates/rodio) integration via [`rodio::RodioResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.RodioResampler.html) | No      |
| `rayon`        | Parallelized resampling (`batch()` and `process_all()` APIs)                      | No      |
| `avx`          | FFT AVX SIMD                                                                      | Yes     |
| `sse`          | FFT SSE SIMD                                                                      | Yes     |
| `neon`         | FFT NEON SIMD for ARM / Mac                                                       | Yes     |
| `wasm_simd`    | FFT WebAssembly SIMD                                                              | Yes     |
| `audioadapter` | Experimental [`audioadapter`](https://crates.io/crates/audioadapter) support      | No      |

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
- `PlanarResampler<T>` and `InterleavedResampler<T>` are fixed-size chunk resamplers. They own one `ArdftsrcCore` per channel and expose full-buffer, chunked, and batch processing APIs for planar or interleaved audio.
- `AdapterResampler<T>` is optional behind the `audioadapter` feature and adapts generic audioadapter inputs and outputs onto the chunk resampling core. Right now there are performance issues with this.
- `RealtimeResampler<T>` provides arbitrary-size sample buffering for live resampling.

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

1. Add `tanh` taper.
2. Add bindings to other languages, python, cpp, c#, ts (wasm) etc.
3. Investigate why the optional audioadapter interface appears to be much slower than other paths.
