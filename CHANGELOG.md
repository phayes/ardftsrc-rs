# Changelog

## Unreleased

### Fixed

- `RodioResampler` no longer over-reads input from packetized sources such as rodio's decoders. When upsampling, the buffered audio grew by about 88 ms per second of playback, delaying seeks by several seconds and allocating on the audio thread.
- `RodioResampler::try_seek()` now discards audio buffered from before the seek and re-primes with fast-start, instead of playing up to two chunks of stale audio first (seconds at `PRESET_HIGH` and above).
- `RodioResampler` no longer panics on sources whose `current_span_len()` is `None`.
- `RodioResampler` no longer writes the first sample of a new span into the previous span when the source exposes the new format only after that sample is pulled, which panicked or swapped channels on format changes.
- `RealtimeResampler::reset()` no longer allocates.

## 0.1.0 - 2026-10-06

First stable release. The resampling API covered by the default features is now considered mostly stable.

