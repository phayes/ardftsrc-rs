# Changelog

## 0.3.0 - 2026-10-11

### Breaking

- All presets and the `Config` default have been retuned, so output differs from 0.2.0 for every preset. Each preset now scores higher on the HydrogenAudio SRC test suite and runs faster; `PRESET_HIGH` and `PRESET_EXTREME` are much faster.

  | Preset | Quality | Bandwidth | Taper |
  | --- | ---: | ---: | --- |
  | `PRESET_FAST` | `512` | `0.86` | `BetaCdf { alpha: 17.0, beta: 17.0 }` |
  | `PRESET_GOOD` (default) | `2304` | `0.95` | `BetaCdf { alpha: 18.0, beta: 18.0 }` |
  | `PRESET_HIGH` | `9216` | `0.986` | `BetaCdf { alpha: 19.0, beta: 19.0 }` |
  | `PRESET_EXTREME` | `28224` | `0.9975` | `BetaCdf { alpha: 12.0, beta: 12.0 }`, alias floor `0` dB |

- `TaperType::default()` is now `BetaCdf { alpha: 18.0, beta: 18.0 }` (was `Cosine(3.4375)`).
- The `bessel` feature is renamed to `kaiser`.
- `TaperType::Bessel` is renamed to `TaperType::KaiserCdf`, and `TaperType::Kbd` to `TaperType::KaiserBesselDerived`. `TaperType::HalfKaiser` is removed.
- `ardftsrc-rs`: `--taper-type bessel` is now `kaiser_cdf` (`bessel` remains an alias), `kbd` also accepts `kaiser_bessel_derived`, and `half_kaiser` is removed.

### Fixed

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

