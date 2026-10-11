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
- The `bessel` feature is removed. The Kaiser tapers no longer depend on `pxfm` and are always available.
- `TaperType::Bessel(alpha)` is replaced by `TaperType::Kaiser(beta)`, which takes the Kaiser window's `beta` as MATLAB and SciPy do: `Bessel(alpha)` corresponds to `Kaiser(PI * alpha)`. It integrates the window exactly and samples it at bin centres, so it is exactly amplitude-complementary and centred on its transition. When the transition straddles Nyquist it samples from unity to zero inclusive, putting −6 dB on Nyquist.
- `TaperType::Kbd(alpha)` is renamed to `TaperType::KaiserBesselDerived(beta)` and also takes `beta` (`Kbd(alpha)` corresponds to `KaiserBesselDerived(PI * alpha)`; AAC's alpha 4 and 6 are beta 4π and 6π). It is now the square root of `Kaiser`, exactly power-complementary.
- Both Kaiser tapers accept `beta` in `(0, 300]` and otherwise fail validation with the new `Error::InvalidKaiserBeta`. `TaperType::HalfKaiser` is removed.
- `ardftsrc-rs`: `--taper-type bessel` is now `kaiser`, `kbd` also accepts `kaiser_bessel_derived`, and both take `--beta` (default 6π ≈ 18.85) instead of `--alpha`. `half_kaiser` is removed.

### Fixed

- With a lowered alias floor, the transition is now centred on the lower Nyquist, so every pair of bins mirrored about it sums to unity gain and Nyquist sits at −6 dB. Taper samples that had underflowed to zero were trimmed, which pushed the transition above Nyquist: with the default taper and the widest floor at 44.1 kHz → 96 kHz, −6 dB sat 9 bins high and Nyquist was at −3.4 dB. This changes `PRESET_EXTREME` output. Without aliasing the layout is unchanged.
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

