# Performance

This document measures ardftsrc's speed and latency for each preset, compares its throughput with
[rubato](https://crates.io/crates/rubato), and says which presets suit which workloads. For quality
measurements, see the per-preset [quality reports](reports/) and the HydrogenAudio links in the
[`Config` docs](https://docs.rs/ardftsrc/latest/ardftsrc/struct.Config.html).

## Summary

1. **No preset is suitable for live monitoring (5 ms).** The latency comes from FFT block size, not
   from compute. Even `fast` adds 21–27 ms. For 44.1 kHz ↔ 48 kHz the shortest possible chunk is
   294 frames, so the latency floor is 10 ms at any quality setting. Use a sinc or polynomial
   resampler for monitoring.
2. **Every preset suits playback when samples can be pulled ahead.** For files and buffered
   streams, the resampler is primed by reading ahead on start and after every seek (rodio
   fast-start), so the listener doesn't hear its latency.
3. **`fast` and `good` suit playback when samples can't be pulled ahead,** such as live or
   network sources that arrive in realtime. `fast` adds 21–27 ms and `good` 96–107 ms, around the
   100 ms limit. `high` adds 294–320 ms, which is too much.
4. **Every preset suits offline work.** On one core, `high` converts 44.1 kHz → 48 kHz at 374× realtime,
   about 0.6 s for a 4-minute stereo track. `high` with `HighPrecision::DoubleDouble` runs at 31–115×
   realtime and gives the best quality.
5. **CPU load is never the constraint for realtime use.** Median per-chunk compute is under 0.6% of the
   chunk's audio duration for every preset except `DoubleDouble` (2–6%).
6. **rubato is faster at comparable settings.** rubato's throughput is 4–6% higher than `fast` for
   44.1 kHz → 48/96 kHz, and up to 51% higher when downsampling from 96 kHz. When input and output
   rates match, ardftsrc is 10–16× faster because it passes samples through.

## Test setup

| | |
| --- | --- |
| Machine | Apple M1 Max (10 cores), 64 GB, macOS 26 (Darwin 25.5) |
| Toolchain | rustc 1.95.0, `release`/`bench` profile (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`) |
| ardftsrc | 0.3.0, default features (SIMD), `rayon` off, single thread, `mimalloc` |
| rubato | 2.x, `Fft` synchronous resampler, `chunk_size = 512`, `sub_chunks = 1`, `FixedSync::Input` |

The presets measured:

| Preset | `quality` | `bandwidth` | Taper | HydrogenAudio balanced score |
| --- | ---: | ---: | --- | ---: |
| `fast` | 512 | 0.92 | Kaiser(30) | 71.83% (`f32`), 97.69% (`f64`) |
| `good` | 2304 | 0.975 | Kaiser(30) | 99.26% |
| `high` | 7056 | 0.996 | Kaiser(34), alias floor 0 dB | 99.86% |
| `high+dd` | `high` with `HighPrecision::DoubleDouble` | | | 99.92% |

`high+dd` requires the `high_precision` feature and `f64`.

## Use-case suitability

The three use cases have different limits:

| Use case | Example | Limit | What matters |
| --- | --- | --- | --- |
| **Realtime monitoring** | Live input monitoring, DAW tracking, effects in a performer's headphones | ≤ 5 ms added latency | Latency, and per-chunk compute bursts fitting inside a small audio callback |
| **Playback** | Music players, streaming clients | ≤ 100 ms added latency | Whether the source can be read ahead; latency and compute bursts fitting inside the audio callback |
| **Offline** | Batch-resampling a library, mastering exports | None | Throughput and quality |

Latency in this section is the delay that the resampler adds, measured through
[`RealtimeResampler`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.RealtimeResampler.html). The
output device's buffer adds to it.

| Preset | Monitoring (≤ 5 ms) | Playback, can't pull ahead (≤ 100 ms) | Playback, can pull ahead | Offline |
| --- | --- | --- | --- | --- |
| `fast` | ❌ 21–27 ms | ✅ 21–27 ms | ✅ | ✅ but lowest quality; prefer a higher preset |
| `good` | ❌ 96–107 ms | ✅ 96–107 ms | ✅ | ✅ |
| `high` | ❌ 294–320 ms | ❌ 294–320 ms | ✅ | ✅ **recommended** |
| `high+dd` | ❌ 294–320 ms | ❌ 294–320 ms | ✅ with a large device buffer | ✅ highest quality, 4–9× slower than `high` |

### 1. Realtime monitoring (5 ms limit)

**No preset is suitable, and lowering `quality` does not fix this for common rate pairs.**

ardftsrc transforms audio in fixed-size input chunks. It can't output a sample until the whole chunk
containing that sample has arrived, and the overlapping windows add half a chunk of algorithmic
delay on top. `RealtimeResampler` also waits for two chunks before it starts emitting output, so its
latency is two chunks. A pipeline built directly on `process_chunk()` can get down to 1.5 chunks.

`fast` has 10.7–13.3 ms chunks, which gives 21–27 ms of latency. The chunk length also has a lower
limit set by the rate ratio: 44.1 kHz ↔ 48 kHz needs at least 294 input frames (6.7 ms) per chunk at
any `quality`. At `quality = 64`, the lowest setting measured, and far below `fast` in quality:

| Scenario | Chunk (frames) | Chunk (ms) | `RealtimeResampler` latency (ms) | Chunk-API minimum (ms) |
| --- | ---: | ---: | ---: | ---: |
| 44.1k → 48k | 294 | 6.67 | 13.32 | 10.00 |
| 48k → 44.1k | 320 | 6.67 | 13.32 | 10.00 |
| 44.1k → 96k | 294 | 6.67 | 13.31 | 10.00 |
| 96k → 48k | 128 | 1.33 | 2.67 | 2.00 |
| 192k → 48k | 256 | 1.33 | 2.66 | 2.00 |

Only integer-ratio conversions can get under 5 ms, and only at a quality setting no preset uses.
Compute speed is not the problem: `fast` processes a chunk in about 20 µs.

For monitoring, use a time-domain resampler such as rubato's sinc or polynomial resamplers. Their
latency depends on the filter length and chunk size you choose, and can stay within a few
milliseconds.

### 2. Playback (100 ms limit)

How much the resampler's latency matters depends on whether samples can be pulled from the source
ahead of when they're played.

#### Sources that can be pulled ahead: all presets

Files, decoded buffers and buffered network streams can be read faster than realtime. The
resampler can then be primed by reading ahead on start and after every seek, so it produces output
straight away and the listener doesn't hear its latency. rodio does this with fast-start
([`Config::with_rodio_fast_start`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.Config.html#method.with_rodio_fast_start)).
With `RealtimeResampler`, write samples until
[`is_primed()`](https://docs.rs/ardftsrc/latest/ardftsrc/struct.RealtimeResampler.html#method.is_primed)
returns `true`.

Every preset works for this kind of playback, including `high`:

- `fast`, `good` and `high` process a chunk in at most about 1 ms at p99 (`high` at 192 kHz input),
  well within a 10 ms audio callback.
- `high+dd` processes a chunk in 3–8.5 ms, and `RealtimeResampler` does that work on the audio
  thread. At 192 kHz → 48 kHz a chunk takes 8.5 ms at p99, longer than a 256-frame callback at
  48 kHz (5.3 ms). Use a larger device buffer, or resample off the audio thread.

The latency is still there, so it matters if playback has to stay in sync with something else, such
as video or a UI that reacts to the sound. In that case, follow the next subsection.

#### Sources that can't be pulled ahead: `fast` and `good`

Live inputs and network streams that arrive at realtime pace can't be read ahead, so the listener
hears all of the resampler's latency.

- **`fast`: suitable.** It adds 21–27 ms of latency, and chunk compute bursts stay under 35 µs at p99.
  `fast` is the only preset where `f32` is reasonable: with `f32` its HydrogenAudio score drops to
  71.83%, against 97.69% with `f64`, and the `f64` compute cost is small.
- **`good`: suitable.** Through `RealtimeResampler` it adds 96 ms for 48 kHz-family inputs and 107 ms
  for 44.1 kHz-family inputs, close to the 100 ms limit. A custom pipeline on
  `InterleavedResampler::process_chunk()` that starts reading after 1.5 chunks instead of 2 brings
  this down to 72–80 ms. Compute bursts stay under 250 µs at p99.
- **`high` and `high+dd`: not suitable.** They add 294–320 ms.

### 3. Offline (batch resampling)

All presets are suitable. Throughput is the only constraint, so the choice comes down to quality
versus time:

- **`high` is the recommended offline preset.** It runs at 256–768× realtime on one core for
  resampling conversions, about 0.6 s for a 4-minute stereo track at 44.1 kHz → 48 kHz.
- **`high+dd`** gives the best quality (99.92% HydrogenAudio balanced score) at 31–115× realtime,
  about 5 s for the same track. Use it when quality matters more than time.
- **`good`** is 1.4–2.2× faster than `high`, at a small quality cost.

These figures are for one core. A batch job runs independent files on separate threads, so total
throughput scales with core count.
The `rayon` feature parallelizes channels within one stream, which helps single long files but
gives little extra when files already run in parallel.

## Benchmarks

### Throughput (`process_all`, × realtime, one core)

The realtime factor is the duration of the input audio divided by wall-clock processing time.
Higher is better. Same-rate rows (22.05k → 22.05k, 96k → 96k) use passthrough and do no resampling.

**`f64`** (recommended for every preset except `fast`):

| Scenario | `fast` | `good` | `high` | `high+dd` |
| --- | ---: | ---: | ---: | ---: |
| 44.1k → 22.05k | 1,353× | 610× | 281× | 56× |
| 44.1k → 48k | 769× | 533× | 374× | 47× |
| 44.1k → 96k | 495× | 387× | 285× | 31× |
| 22.05k → 22.05k | 92,325× | 90,114× | 83,886× | 84,021× |
| 22.05k → 48k | 1,901× | 1,329× | 768× | 115× |
| 22.05k → 96k | 1,161× | 889× | 585× | 67× |
| 96k → 22.05k | 955× | 531× | 256× | 59× |
| 96k → 48k | 1,350× | 705× | 362× | 55× |
| 96k → 96k | 21,553× | 21,366× | 21,060× | 21,028× |

**`f32`**:

| Scenario | `fast` | `good` | `high` |
| --- | ---: | ---: | ---: |
| 44.1k → 22.05k | 1,956× | 843× | 352× |
| 44.1k → 48k | 1,046× | 717× | 530× |
| 44.1k → 96k | 734× | 546× | 418× |
| 22.05k → 22.05k | 95,515× | 94,645× | 91,840× |
| 22.05k → 48k | 2,730× | 1,765× | 1,090× |
| 22.05k → 96k | 1,702× | 1,250× | 818× |
| 96k → 22.05k | 1,333× | 691× | 324× |
| 96k → 48k | 2,024× | 996× | 469× |
| 96k → 96k | 21,833× | 21,704× | 21,545× |

`f32` is 27–50% faster than `f64`, but loses accuracy at the higher presets. For `f32` audio, convert
to `f64`, resample, and convert back.

The 44.1 kHz fixture is stereo; the 22.05 kHz and 96 kHz fixtures are mono.

### Throughput: `fast` vs rubato

Output kilosamples per second; higher is better. rubato settings are listed in [Test setup](#test-setup).

| Scenario | Type | ardftsrc `fast` | rubato | Faster |
| --- | --- | ---: | ---: | --- |
| 44.1k → 22.05k | `f32` | 86,260 | 120,565 | rubato (+40%) |
| 44.1k → 22.05k | `f64` | 59,659 | 83,089 | rubato (+39%) |
| 44.1k → 48k | `f32` | 100,458 | 105,744 | rubato (+5%) |
| 44.1k → 48k | `f64` | 73,871 | 77,149 | rubato (+4%) |
| 44.1k → 96k | `f32` | 140,948 | 145,965 | rubato (+4%) |
| 44.1k → 96k | `f64` | 95,107 | 100,562 | rubato (+6%) |
| 22.05k → 22.05k | `f32` | 2,106,102 | 201,472 | ardftsrc (+945%) |
| 22.05k → 22.05k | `f64` | 2,035,756 | 129,150 | ardftsrc (+1476%) |
| 22.05k → 48k | `f32` | 131,035 | 155,980 | rubato (+19%) |
| 22.05k → 48k | `f64` | 91,267 | 105,166 | rubato (+15%) |
| 22.05k → 96k | `f32` | 163,420 | 186,468 | rubato (+14%) |
| 22.05k → 96k | `f64` | 111,443 | 119,448 | rubato (+7%) |
| 96k → 22.05k | `f32` | 29,385 | 44,125 | rubato (+50%) |
| 96k → 22.05k | `f64` | 21,060 | 31,838 | rubato (+51%) |
| 96k → 48k | `f32` | 97,157 | 138,154 | rubato (+42%) |
| 96k → 48k | `f64` | 64,779 | 90,607 | rubato (+40%) |
| 96k → 96k | `f32` | 2,095,942 | 200,584 | ardftsrc (+945%) |
| 96k → 96k | `f64` | 2,069,094 | 128,093 | ardftsrc (+1515%) |

### Latency and per-chunk compute

These measurements use stereo white noise.

- **Latency:** the measured delay of an impulse through `RealtimeResampler`, with input written at the
  input rate and output read at the output rate.
- **Chunk-API min:** the smallest fixed delay at which a pipeline built on `process_chunk()` never
  runs out of output.
- **Compute:** the wall-clock time of each `process_chunk()` call, sampled over up to 3 s.
- **Load:** median compute time as a share of the chunk's audio duration, on one core.

The max column includes OS scheduling noise.

| Preset | Type | Scenario | Chunk (frames) | Chunk (ms) | Latency (ms) | Chunk-API min (ms) | Compute median (µs) | Compute p99 (µs) | Compute max (µs) | Load (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | `f32` | 44.1k → 48k | 588 | 13.33 | 26.64 | 20.00 | 12.9 | 19.9 | 93.2 | 0.10 |
| fast | `f64` | 44.1k → 48k | 588 | 13.33 | 26.64 | 20.00 | 17.5 | 22.4 | 86.9 | 0.13 |
| fast | `f32` | 48k → 44.1k | 640 | 13.33 | 26.67 | 20.00 | 12.6 | 17.7 | 57.2 | 0.09 |
| fast | `f64` | 48k → 44.1k | 640 | 13.33 | 26.67 | 20.00 | 17.8 | 24.3 | 110.2 | 0.13 |
| fast | `f32` | 44.1k → 96k | 588 | 13.33 | 26.65 | 20.00 | 19.0 | 23.9 | 97.9 | 0.14 |
| fast | `f64` | 44.1k → 96k | 588 | 13.33 | 26.65 | 20.00 | 26.9 | 34.2 | 130.3 | 0.20 |
| fast | `f32` | 96k → 48k | 1024 | 10.67 | 21.34 | 16.00 | 10.3 | 14.1 | 53.4 | 0.10 |
| fast | `f64` | 96k → 48k | 1024 | 10.67 | 21.32 | 16.00 | 15.1 | 20.2 | 94.5 | 0.14 |
| fast | `f32` | 192k → 48k | 2048 | 10.67 | 21.32 | 16.00 | 17.9 | 22.4 | 74.4 | 0.17 |
| fast | `f64` | 192k → 48k | 2048 | 10.67 | 21.32 | 16.00 | 25.9 | 30.8 | 121.1 | 0.24 |
| good | `f64` | 44.1k → 48k | 2352 | 53.33 | 106.64 | 80.00 | 84.0 | 99.8 | 185.7 | 0.16 |
| good | `f64` | 48k → 44.1k | 2560 | 53.33 | 106.67 | 80.00 | 83.3 | 111.4 | 183.7 | 0.16 |
| good | `f64` | 44.1k → 96k | 2352 | 53.33 | 106.65 | 80.00 | 123.1 | 192.5 | 287.7 | 0.23 |
| good | `f64` | 96k → 48k | 4608 | 48.00 | 96.00 | 72.00 | 98.7 | 134.3 | 233.0 | 0.21 |
| good | `f64` | 192k → 48k | 9216 | 48.00 | 96.00 | 72.00 | 174.0 | 247.8 | 368.3 | 0.36 |
| high | `f64` | 44.1k → 48k | 7056 | 160.00 | 319.98 | 240.00 | 247.4 | 338.0 | 432.3 | 0.15 |
| high | `f64` | 48k → 44.1k | 7680 | 160.00 | 320.00 | 240.00 | 244.8 | 346.6 | 419.3 | 0.15 |
| high | `f64` | 44.1k → 96k | 7056 | 160.00 | 319.98 | 240.00 | 384.3 | 492.4 | 2986.7 | 0.24 |
| high | `f64` | 96k → 48k | 14112 | 147.00 | 294.00 | 220.50 | 422.4 | 550.6 | 611.6 | 0.29 |
| high | `f64` | 192k → 48k | 28224 | 147.00 | 294.00 | 220.50 | 820.0 | 999.5 | 1130.0 | 0.56 |
| high+dd | `f64` | 44.1k → 48k | 7056 | 160.00 | 319.98 | 240.00 | 3064.5 | 3223.9 | 3662.2 | 1.92 |
| high+dd | `f64` | 48k → 44.1k | 7680 | 160.00 | 320.00 | 240.00 | 3059.2 | 3196.6 | 4641.8 | 1.91 |
| high+dd | `f64` | 44.1k → 96k | 7056 | 160.00 | 319.98 | 240.00 | 4760.2 | 4913.5 | 4961.0 | 2.98 |
| high+dd | `f64` | 96k → 48k | 14112 | 147.00 | 294.00 | 220.50 | 4713.3 | 4881.6 | 5039.2 | 3.21 |
| high+dd | `f64` | 192k → 48k | 28224 | 147.00 | 294.00 | 220.50 | 8309.0 | 8536.1 | 8683.4 | 5.65 |

## Reproducing

Run from the `ardftsrc/` crate directory. Include `--features high_precision` to add the `high+dd`
rows. Don't enable `rayon`.

```sh
# Latency, per-chunk compute, and the quality-64 latency floor (prints Markdown tables)
cargo bench --features high_precision --bench ardftsrc_latency

# Throughput (criterion; results in target/criterion/)
cargo bench --features high_precision --bench ardftsrc_throughput
cargo bench --bench rubato_sync

# Criterion per-chunk timing (mean only; ardftsrc_latency reports the distribution)
cargo bench --bench ardftsrc_chunk_time
```

Run benchmarks one at a time on an otherwise idle machine.

## Follow-ups

1. rubato could add a passthrough fast path for matching sample rates.
2. rubato could add synthetic pre/post samples at stream edges.
3. Measure quality with test suites other than HydrogenAudio SRC.
4. Investigate starting `RealtimeResampler` output after 1.5 chunks instead of 2. The chunk-API
   measurements show 1.5 chunks is enough, and it would bring `good` within the 100 ms playback
   budget.
