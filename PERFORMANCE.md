# Performance

This document measures ardftsrc's speed and latency for each preset. For quality
measurements, see the per-preset [quality reports](reports/) and the HydrogenAudio links in the
[`Config` docs](https://docs.rs/ardftsrc/latest/ardftsrc/struct.Config.html).

## Summary

1. **No preset is suitable for live monitoring (5 ms).** Latency from FFT block sizes are too large. 
   Use a sinc resampler for monitoring.
2. **Every preset suits playback when samples can be pulled ahead.** For files and buffered
   streams, the resampler is primed by reading ahead on start and after every seek (rodio
   fast-start), so the listener doesn't hear its latency.
3. **`fast` and `good` suit playback when samples can't be pulled ahead,** such as live or
   network sources that arrive in realtime. `fast` adds 21–27 ms and `good` 96–107 ms.
4. **Every preset suits offline work.** On one core, `high` converts 44.1 kHz → 48 kHz at 662× realtime,
   about 0.4 s for a 4-minute stereo track. 
5. **CPU load is never the constraint for realtime use.** Median per-chunk compute is under 0.6% of the
   chunk's audio duration for every preset.
6. **rubato and `fast` are close in speed.** They are within 3% of each other when upsampling.
   rubato is 13–17% faster when downsampling in `f64` (5–6% in `f32`). When input and output rates
   match, ardftsrc is 11–17× faster because it passes samples through.

## Test setup

| | |
| --- | --- |
| Machine | Apple M1 Max (10 cores), 64 GB, macOS 26 (Darwin 25.5) |
| Toolchain | rustc 1.95.0, `release`/`bench` profile (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`) |
| ardftsrc | 0.3.1, default features (SIMD), `rayon` off, single thread, `mimalloc` |
| rubato | 2.0.0, `Fft` synchronous resampler, `chunk_size = 512`, `sub_chunks = 1`, `FixedSync::Input` |
| Input | Deterministic stereo white noise for every benchmark; 4 minutes per input rate for throughput |

The presets measured:

| Preset | `quality` | `bandwidth` | Taper | HydrogenAudio balanced score |
| --- | ---: | ---: | --- | ---: |
| `fast` | 512 | 0.92 | Kaiser(30) | 71.83% (`f32`), 97.69% (`f64`) |
| `good` | 2304 | 0.975 | Kaiser(30) | 99.26% |
| `high` | 7056 | 0.996 | Kaiser(34), alias floor 0 dB | 99.86% |
| `high+dd` | `high` in `DoubleDouble` | | | 99.92% |

`high+dd` requires the `high_precision` feature.

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
| `high+dd` | ❌ 294–320 ms | ❌ 294–320 ms | ✅ with a large device buffer | ✅ highest quality, 11–14× slower than `high` |

### 1. Realtime monitoring (5 ms limit)

**No preset is suitable, and lowering `quality` does not fix this for common rate pairs.**

ardftsrc transforms audio in fixed-size input chunks. It can't output a sample until the whole chunk
containing that sample has arrived, and the overlapping windows add half a chunk of algorithmic
delay on top. `RealtimeResampler` also waits for two chunks before it starts emitting output, so its
latency is two chunks. A pipeline built directly on `process_chunk()` can get down to 1.5 chunks.

| Scenario | Chunk (frames) | Chunk (ms) | `RealtimeResampler` latency (ms) | Chunk-API minimum (ms) |
| --- | ---: | ---: | ---: | ---: |
| 44.1k → 48k | 294 | 6.67 | 13.32 | 10.00 |
| 48k → 44.1k | 320 | 6.67 | 13.32 | 10.00 |
| 44.1k → 96k | 294 | 6.67 | 13.31 | 10.00 |
| 96k → 48k | 128 | 1.33 | 2.67 | 2.00 |
| 192k → 48k | 256 | 1.33 | 2.66 | 2.00 |

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

#### Sources that can't be pulled ahead: `fast` and `good`

Live inputs and network streams that arrive at realtime pace can't be read ahead, so the listener
hears all of the resampler's latency.

- **`fast`: suitable.** It adds 21–27 ms of latency, and chunk compute bursts stay under 40 µs at p99.
  `fast` is the only preset where `f32` is reasonable: with `f32` its HydrogenAudio score drops to
  71.83%, against 97.69% with `f64`, and the `f64` compute cost is small.
- **`good`: suitable.** Through `RealtimeResampler` it adds 96 ms for 48 kHz-family inputs and 107 ms
  for 44.1 kHz-family inputs. 
- **`high`: not suitable.** They add 294–320 ms latency.

### 3. Offline (batch resampling)

All presets are suitable. Throughput is the only constraint, so the choice comes down to quality
versus time:

- **`high` is the recommended offline preset.** It runs at 362–874× realtime on one core for
  resampling conversions, about 0.4 s for a 4-minute stereo track at 44.1 kHz → 48 kHz.
- **`high+dd`** gives the best quality (99.92% HydrogenAudio balanced score) at 31–67× realtime,
  about 5 s for the same track. Use it when quality matters more than time.
- **`good`** saves little time offline. It is no faster than `high` for 44.1 kHz → 48 kHz and
  22.05 kHz → 48 kHz, and at most 1.4× faster for the other conversions. Per second of audio, FFT
  cost grows only slowly with block size.

`process_all` also has a fixed cost per stream, mostly for extrapolating the signal past each edge.
Per stereo stream it is about 0.6 ms for `fast`, 2–4 ms for `good`, 7–13 ms for `high` and
60–120 ms for `high+dd`. That is 2% or less of a 4-minute track, but it dominates for clips of a
second or two, which resample more slowly than the table below shows.

These figures are for one core. A batch job runs independent files on separate threads, so total
throughput scales with core count.
The `rayon` feature parallelizes channels within one stream, which helps single long files but
gives little extra when files already run in parallel.

## Benchmarks

### Throughput (`process_all`, × realtime, one core)

Each input is 4 minutes of stereo white noise. The realtime factor is the duration of the input
audio divided by wall-clock processing time, so higher is better. Same-rate rows
(22.05k → 22.05k, 96k → 96k) use passthrough and do no resampling; they only deinterleave, and are
limited by memory bandwidth.

**`f64`** (recommended for every preset except `fast`):

| Scenario | `fast` | `good` | `high` | `high+dd` |
| --- | ---: | ---: | ---: | ---: |
| 44.1k → 22.05k | 1,638× | 1,054× | 753× | 65× |
| 44.1k → 48k | 826× | 653× | 662× | 51× |
| 44.1k → 96k | 524× | 451× | 423× | 34× |
| 22.05k → 22.05k | 44,384× | 44,455× | 44,508× | 33,582× |
| 22.05k → 48k | 1,047× | 799× | 874× | 67× |
| 22.05k → 96k | 605× | 554× | 510× | 38× |
| 96k → 22.05k | 607× | 530× | 492× | 37× |
| 96k → 48k | 776× | 502× | 362× | 31× |
| 96k → 96k | 10,545× | 10,540× | 10,402× | 8,090× |

**`f32`**:

| Scenario | `fast` | `good` | `high` |
| --- | ---: | ---: | ---: |
| 44.1k → 22.05k | 2,607× | 1,607× | 1,029× |
| 44.1k → 48k | 1,133× | 913× | 995× |
| 44.1k → 96k | 783× | 644× | 671× |
| 22.05k → 22.05k | 46,394× | 46,270× | 46,174× |
| 22.05k → 48k | 1,565× | 1,316× | 1,361× |
| 22.05k → 96k | 929× | 800× | 778× |
| 96k → 22.05k | 889× | 775× | 714× |
| 96k → 48k | 1,201× | 752× | 502× |
| 96k → 96k | 10,708× | 10,831× | 10,778× |

`f32` is 37–65% faster than `f64` when resampling, but loses accuracy at the higher presets. For
`f32` audio, convert to `f64`, resample, and convert back.

### Throughput: `fast` vs rubato

Realtime factor on the same inputs as above; higher is better. rubato settings are listed in
[Test setup](#test-setup).

| Scenario | Type | ardftsrc `fast` | rubato | Faster |
| --- | --- | ---: | ---: | --- |
| 44.1k → 22.05k | `f32` | 2,607× | 2,772× | rubato (+6%) |
| 44.1k → 22.05k | `f64` | 1,638× | 1,921× | rubato (+17%) |
| 44.1k → 48k | `f32` | 1,133× | 1,117× | ardftsrc (+1%) |
| 44.1k → 48k | `f64` | 826× | 805× | ardftsrc (+3%) |
| 44.1k → 96k | `f32` | 783× | 758× | ardftsrc (+3%) |
| 44.1k → 96k | `f64` | 524× | 524× | even |
| 22.05k → 22.05k | `f32` | 46,394× | 4,050× | ardftsrc (+1046%) |
| 22.05k → 22.05k | `f64` | 44,384× | 2,718× | ardftsrc (+1533%) |
| 22.05k → 48k | `f32` | 1,565× | 1,515× | ardftsrc (+3%) |
| 22.05k → 48k | `f64` | 1,047× | 1,051× | even |
| 22.05k → 96k | `f32` | 929× | 911× | ardftsrc (+2%) |
| 22.05k → 96k | `f64` | 605× | 604× | even |
| 96k → 22.05k | `f32` | 889× | 933× | rubato (+5%) |
| 96k → 22.05k | `f64` | 607× | 689× | rubato (+13%) |
| 96k → 48k | `f32` | 1,201× | 1,272× | rubato (+6%) |
| 96k → 48k | `f64` | 776× | 883× | rubato (+14%) |
| 96k → 96k | `f32` | 10,708× | 930× | ardftsrc (+1051%) |
| 96k → 96k | `f64` | 10,545× | 623× | ardftsrc (+1592%) |

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
| fast | `f32` | 44.1k → 48k | 588 | 13.33 | 26.64 | 20.00 | 14.7 | 31.0 | 92.6 | 0.11 |
| fast | `f64` | 44.1k → 48k | 588 | 13.33 | 26.64 | 20.00 | 17.4 | 20.9 | 85.4 | 0.13 |
| fast | `f32` | 48k → 44.1k | 640 | 13.33 | 26.67 | 20.00 | 12.5 | 15.4 | 71.7 | 0.09 |
| fast | `f64` | 48k → 44.1k | 640 | 13.33 | 26.67 | 20.00 | 17.3 | 22.9 | 116.4 | 0.13 |
| fast | `f32` | 44.1k → 96k | 588 | 13.33 | 26.65 | 20.00 | 18.9 | 23.1 | 134.6 | 0.14 |
| fast | `f64` | 44.1k → 96k | 588 | 13.33 | 26.65 | 20.00 | 26.4 | 36.4 | 139.4 | 0.20 |
| fast | `f32` | 96k → 48k | 1024 | 10.67 | 21.34 | 16.00 | 10.0 | 11.2 | 41.0 | 0.09 |
| fast | `f64` | 96k → 48k | 1024 | 10.67 | 21.32 | 16.00 | 14.9 | 16.6 | 107.0 | 0.14 |
| fast | `f32` | 192k → 48k | 2048 | 10.67 | 21.32 | 16.00 | 17.3 | 22.1 | 115.6 | 0.16 |
| fast | `f64` | 192k → 48k | 2048 | 10.67 | 21.32 | 16.00 | 25.2 | 34.3 | 125.0 | 0.24 |
| good | `f64` | 44.1k → 48k | 2352 | 53.33 | 106.64 | 80.00 | 81.0 | 120.6 | 210.4 | 0.15 |
| good | `f64` | 48k → 44.1k | 2560 | 53.33 | 106.67 | 80.00 | 80.7 | 124.8 | 217.4 | 0.15 |
| good | `f64` | 44.1k → 96k | 2352 | 53.33 | 106.65 | 80.00 | 119.8 | 193.2 | 269.5 | 0.22 |
| good | `f64` | 96k → 48k | 4608 | 48.00 | 96.00 | 72.00 | 96.4 | 140.5 | 234.1 | 0.20 |
| good | `f64` | 192k → 48k | 9216 | 48.00 | 96.00 | 72.00 | 169.1 | 259.0 | 353.8 | 0.35 |
| high | `f64` | 44.1k → 48k | 7056 | 160.00 | 319.98 | 240.00 | 240.5 | 375.2 | 439.2 | 0.15 |
| high | `f64` | 48k → 44.1k | 7680 | 160.00 | 320.00 | 240.00 | 238.4 | 372.5 | 433.0 | 0.15 |
| high | `f64` | 44.1k → 96k | 7056 | 160.00 | 319.98 | 240.00 | 374.5 | 511.9 | 681.5 | 0.23 |
| high | `f64` | 96k → 48k | 14112 | 147.00 | 294.00 | 220.50 | 408.9 | 569.8 | 652.5 | 0.28 |
| high | `f64` | 192k → 48k | 28224 | 147.00 | 294.00 | 220.50 | 796.4 | 993.4 | 1096.7 | 0.54 |
| high+dd | `DoubleDouble` | 44.1k → 48k | 7056 | 160.00 | 319.98 | 240.00 | 3024.6 | 3145.8 | 3284.8 | 1.89 |
| high+dd | `DoubleDouble` | 48k → 44.1k | 7680 | 160.00 | 320.00 | 240.00 | 3021.7 | 3132.8 | 3162.4 | 1.89 |
| high+dd | `DoubleDouble` | 44.1k → 96k | 7056 | 160.00 | 319.98 | 240.00 | 4714.5 | 4837.0 | 4870.3 | 2.95 |
| high+dd | `DoubleDouble` | 96k → 48k | 14112 | 147.00 | 294.00 | 220.50 | 4650.8 | 4803.5 | 4878.8 | 3.16 |
| high+dd | `DoubleDouble` | 192k → 48k | 28224 | 147.00 | 294.00 | 220.50 | 8155.3 | 8386.5 | 8464.2 | 5.55 |

## Reproducing

Run from the `ardftsrc/` crate directory. Include `--features high_precision` to add the `high+dd`
rows. Don't enable `rayon`.

```sh
# Latency, per-chunk compute, and the quality-64 latency floor (prints Markdown tables)
cargo bench --features high_precision --bench ardftsrc_latency

# Throughput (criterion; results in target/criterion/). Each iteration resamples a 4-minute
# stereo track, so this takes about 25 minutes.
cargo bench --features high_precision --bench ardftsrc_throughput
cargo bench --bench rubato_sync

# Criterion per-chunk timing (mean only; ardftsrc_latency reports the distribution)
cargo bench --bench ardftsrc_chunk_time
```

Criterion reports throughput in input frames per second (`elem/s`). Divide it by the input rate to
get the realtime factor.

Run benchmarks one at a time on an otherwise idle machine.

## Follow-ups

1. rubato could add a passthrough fast path for matching sample rates.
2. rubato could add synthetic pre/post samples at stream edges.
3. Measure quality with test suites other than HydrogenAudio SRC.
4. Investigate starting `RealtimeResampler` output after 1.5 chunks instead of 2. The chunk-API
   measurements show 1.5 chunks is enough, and it would bring `good` within the 100 ms playback
   budget.
