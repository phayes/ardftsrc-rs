# Changelog

## 0.4.4 - 2026-10-14

### Added

- `RealtimeResampler::samples_outstanding()` reports how many output samples the input written so far has yet to hand out, ready or not. Added to what a caller has read, it is where the next written sample will come out, to within a frame, so a caller can place a boundary such as a gapless track join without finalizing the stream.

## 0.4.3 - 2026-10-14

### Added

- `RealtimeResampler::write_span()` and `RealtimeResampler::read_span()` replace `write_rodio_buffer()` and `read_rodio_buffer()`. They work without the `rodio` feature and allocate nothing. `write_span()` takes the samples from an iterator, starting a new span when their rate or channel count changes. `read_span()` hands back a `SpanSamples` iterator over one span's output, with its channel count and rate, so a caller can collect it into a buffer it reuses.

### Removed

- `RealtimeResampler::write_rodio_buffer()` and `RealtimeResampler::read_rodio_buffer()`, added in 0.4.2. Pass a `rodio::buffer::SamplesBuffer`'s rate, channel count and samples to `write_span()`, and build one from a `read_span()` result's `channels()`, `sample_rate()` and collected samples.

## 0.4.2 - 2026-10-13

### Added

- `RealtimeResampler::write_rodio_buffer()` and `RealtimeResampler::read_rodio_buffer()`, with the `rodio` feature, for push-style pipelines that resample finished packets as they are decoded and queue what comes out. A packet in another rate or channel count starts a new span by itself, and a read never straddles two spans, so each buffer has one channel count.
- `RealtimeResampler::is_empty()` reports whether the resampler holds any audio, including the delayed tail of a stream not yet finalized, so a caller can send audio around it without reordering.
- `RealtimeResampler::pre()` and `RealtimeResampler::post()` set previous- and next-track edge context for gapless material, as `InterleavedResampler` and `PlanarResampler` already do.
- `RealtimeResampler::write_iter()` accepts interleaved samples from an iterator, without collecting them into a slice first.

## 0.4.1 - 2026-10-12

### Added

- `RealtimeResampler::is_input_span_primed()` reports whether the span being written has been fed enough input to produce output. Unlike `is_primed()`, it goes `false` again after `new_span()`, so callers that fast-start can re-prime at a format change.

### Fixed

- `RodioResampler` no longer drops out for up to ~27 ms (at `PRESET_GOOD`) when the inner source changes sample rate or channel count mid-stream. Input pacing is now counted in frames, so a channel-count change no longer under-feeds the new span while the previous one drains, and with fast-start the new span is primed as soon as it starts.
- `RealtimeResampler::estimate_priming_duration()` divides by the channel count, so it is no longer twice too long for stereo. `RodioResampler::total_duration()` without fast-start, which adds it, is corrected too.
- The MIT licence names its copyright year and holder, and both licence files are now included in the published `ardftsrc` and `ardftsrc-rs` crates.

## 0.4.0 - 2026-10-12

### Breaking

- High precision is now a sample type instead of a config option. With the `high_precision` feature, the new public types `DoubleDouble`, `F128`, and `F256` can be used as any resampler's sample type, e.g. `InterleavedResampler::<F128>`. Convert samples in with `From<f64>`/`From<f32>` (exact) and out with `to_f64()`.
- Every stage now runs in the sample type: decimation, edge extrapolation, both FFTs, the spectral taper and phase, overlap-add, and output scaling. Previously only the FFTs ran in the wide type, and the spectrum was rounded to `f64` between them, so high-precision output differs slightly from 0.3.1.
- Removed `Config::high_precision`, `Config::with_high_precision`, `HighPrecision`, and `Error::HighPrecisionIncompatibleWithF32`.
- `InterleavedResampler`, `PlanarResampler`, and `RealtimeResampler` are bounded by the new `Sample` trait instead of `num_traits::Float + realfft::FftNum`. Every `Float + FftNum` type implements `Sample`, so `f32` and `f64` code is unaffected unless it names the bounds.
- `ardftsrc-rs` replaces `--high-precision double-double|f128|f256` with `--dd`, `--f128`, and `--f256`, alongside `--f32` and `--f64`. High-precision runs decode and encode in `f64`.

### Added

- `RodioResampler::reset()` discards buffered audio and resamples afresh from wherever the inner source is, without seeking it. Use it when the inner source jumps on its own, such as a cleared queue. With fast-start enabled it re-primes at once. `try_seek()` now calls it after seeking the inner source.

### Changed

- The CPU resampler now reuses per-core FFT scratch buffers instead of allocating them on every chunk.

### Documentation

- `PERFORMANCE.md` is regenerated. On one core, `PRESET_HIGH` converts 44.1 kHz → 48 kHz at 662× realtime (was 374×). rubato and `PRESET_FAST` are now within 3% of each other when upsampling.

### Benchmarks

- The throughput, chunk-time, and rubato benches share one input: 4 minutes of deterministic stereo white noise per input rate, replacing the `test_wavs` fixtures, which differed in length and channel count between rates.

## 0.3.1 - 2026-10-11

Doc updates. No library changes.

### Documentation

- `PERFORMANCE.md` is regenerated for the 0.3.0 presets, with throughput, latency and per-chunk compute measurements, and a comparison with rubato.
- `PERFORMANCE.md` has a new section on which presets suit realtime monitoring (5 ms), playback (100 ms) and offline use. No preset suits monitoring. `PRESET_FAST` and `PRESET_GOOD` suit playback from sources that can't be read ahead; every preset suits playback from files and other buffered sources.
- The READMEs link to `PERFORMANCE.md`.

### Benchmarks

- New `ardftsrc_latency` bench. It measures latency through `RealtimeResampler`, the per-chunk compute time distribution and CPU load, and the latency floor at minimum `quality`.
- `ardftsrc_throughput` includes `PRESET_HIGH` with `HighPrecision::DoubleDouble` when built with `--features high_precision`.

## 0.3.0 - 2026-10-11

### Breaking

- All presets and the `Config` default have been retuned, so output differs from 0.2.0 for every preset. Each preset now scores higher on the HydrogenAudio SRC test suite and runs faster; `PRESET_HIGH` is much faster.

  | Preset | Quality | Bandwidth | Taper |
  | --- | ---: | ---: | --- |
  | `PRESET_FAST` | `512` | `0.92` | `Kaiser(30.0)` |
  | `PRESET_GOOD` (default) | `2304` | `0.975` | `Kaiser(30.0)` |
  | `PRESET_HIGH` | `7056` | `0.996` | `Kaiser(34.0)`, alias floor `0` dB |

- `PRESET_EXTREME` is removed, along with `ardftsrc-rs --preset extreme`. `PRESET_HIGH` now permits aliasing at the lower Nyquist (alias floor `0` dB) and scores higher than `PRESET_EXTREME` did, with a quarter of its FFT size. Use `PRESET_HIGH` instead.
- `PRESET_HIGH` now supports `f32` processing (its quality is within the `f32` limit of 8192).

- `TaperType::default()` is now `Kaiser(30.0)` (was `Cosine(3.4375)`).
- The `bessel` feature is removed. The Kaiser taper no longer depends on `pxfm` and is always available.
- `TaperType::Bessel(alpha)` is replaced by `TaperType::Kaiser(beta)`, which takes the Kaiser window's `beta` as MATLAB and SciPy do: `Bessel(alpha)` corresponds to `Kaiser(PI * alpha)`. It integrates the window exactly and samples it at bin centres, so it is exactly amplitude-complementary and centred on its transition. When the transition straddles Nyquist it samples from unity to zero inclusive, putting −6 dB on Nyquist.
- `Kaiser` accepts `beta` in `(0, 300]` and otherwise fails validation with the new `Error::InvalidKaiserBeta`.
- `TaperType::Kbd` and `TaperType::HalfKaiser` are removed.
- `ardftsrc-rs`: `--taper-type bessel` is now `kaiser` and takes `--beta` (default `30`) instead of `--alpha`. `kbd` and `half_kaiser` are removed.
- `CpuCore` is no longer exported. It could not be constructed outside the crate, since its constructor takes a crate-internal configuration type. Use `PlanarResampler` with `channels = 1` for single-channel resampling.

### Fixed

- With a lowered alias floor, the transition is now centred on the lower Nyquist, so every pair of bins mirrored about it sums to unity gain and Nyquist sits at −6 dB. Taper samples that had underflowed to zero were trimmed, which pushed the transition above Nyquist: with the default taper and the widest floor at 44.1 kHz → 96 kHz, −6 dB sat 9 bins high and Nyquist was at −3.4 dB. This affects `PRESET_HIGH`, which now uses an alias floor of `0` dB. Without aliasing the layout is unchanged.
- Taper profiles are now computed in `f64` for every sample type. In `f32`, `Cosine` and `Planck` tails underflowed earlier and shifted the transition toward Nyquist (by up to 280 bins for `Cosine(3.4375)` over 5000 transition bins).
- End-of-stream stop extrapolation no longer reads its input history from a buffer the forward FFT had already used as scratch. With the default FFT backend, the synthesized tail was built from corrupted samples, degrading the final chunk; outputs now match the high-precision backends.
- With `decimate` enabled, output is now exactly time-aligned with the input. The decimation cascade's group delay was rounded to whole output samples, leaving a fractional-sample offset; the remainder is now compensated in the spectral stage.

### Documentation

- Quality reports regenerated for the new presets, with an `f32` report for `PRESET_FAST`.
- Added guidance on compiler optimization settings for realtime resampling.

## 0.2.0 - 2026-10-08

### Fixed

- `RodioResampler` no longer over-reads input from packetized sources such as rodio's decoders. When upsampling, the buffered audio grew by about 88 ms per second of playback, delaying seeks by several seconds and allocating on the audio thread.
- `RodioResampler::try_seek()` now discards audio buffered from before the seek and re-primes with fast-start, instead of playing up to two chunks of stale audio first (seconds at `PRESET_HIGH` and above).
- `RodioResampler` no longer panics on sources whose `current_span_len()` is `None`.
- `RodioResampler` no longer writes the first sample of a new span into the previous span when the source exposes the new format only after that sample is pulled, which panicked or swapped channels on format changes.
- `RealtimeResampler::reset()` no longer allocates.

## 0.1.0 - 2026-10-06

First stable release. The resampling API covered by the default features is now considered mostly stable.

