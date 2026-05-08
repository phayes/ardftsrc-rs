# Performance Testing

This doc compares ardftsrc-rs against the best-in-class sinc resampler rubato. 
It's meant to help gauge where ardftsrc-rs might fit in the rust-audio ecosystem.

We use the HydrogenAudio SRC Test Suite as our "quality" proxy.

## Summary

1. Rubato is faster than ardftsrc in all instances, even when rubato is running at "high quality sync"
2. At higher quality levels (> fast) ardftsrc beats out rubato in quality
3. Pre-ringing is an issue for ardftsrc. This can be ameliorated by introducing a "phase" param (not done yet)
4. They both use about the same amount of memory. 
5. Rubato `f64` performs significantly better than rubato `f32`. We should investigate using rubato with `f64` even in `f32` pipelines (`f32` -> `f64` -> rubato::<`f64`> -> `f64` -> `f32`). 

## Conclusion

When occupying aproximately the same "samples per second" space, the two are competitive: rubato generally retains a throughput advantage, while ardftsrc tends to edge ahead on several quality metrics.

When higher quality resampling is desired, ardftsrc has the advantage. This higher quality comes at the cost of more compute, although still easily within realtime limits.
The "good" preset sits in the goldilocks zone for realtime high-quality resampling for ardftsrc.

## Test 1: Fast (vs rubato)

```
ardftsrc: --preset=fast, quality=512, bandwidth=0.8323
rubato:   FixedSync, chunk-size=512, sub-chunks=1, FixedSync::Both
```

### HydrogenAudio Scores

**ardftsrc vs rubato:**

1. `f32`: https://src.hydrogenaudio.org/compareresults?id1=c527356d-3566-46f8-8dea-dc2065b11e46&id2=3e45fdeb-152f-4abd-97d6-1848b86243c8
2. `f64`: https://src.hydrogenaudio.org/compareresults?id1=8e59a5bd-8147-470c-9501-44ab81718b8f&id2=6b05a1f8-87db-4e3e-8aa0-aefacfda7a3e

*`f32`*:
1. Spectogram: both display significant aliasing, but ardftsrc-rs aliasing is mildly worse.
3. ardftsrc wins Gapless Sine (45% vs 5%)
4. All other metrics roughly equivalent. 

*`f64`*
1. Spectogram: both are excellent, with rubato mildly better.
2. ardftsrc wins Aliasing, Intermodulation Harmonic Distortion, Impulse Frequency, Impulse Response, Gapless Sine
3. rubato wins Nyquist Filter (but only above 20Khz - audible frequencies they are equivalent)

Overall, if the goal is speed over quality, rubato wins. But at a cost of more computing cost, ardftsrc edges it out, especially once we move to `f64`. 

### Bench

| Scenario        | Type | ardftsrc (kilosamples/s) | rubato (kilosamples/s) | ardftsrc overhead (%) | rubato overhead (%) | Faster            |
| --------------- | ---- | -----------------------: | ---------------------: | --------------------: | ------------------: | ----------------- |
| 44.1k → 22.05k  | `f32`  |                   89,780 |                124,060 |               0.0491% |             0.0355% | rubato (+38%)     |
| 44.1k → 22.05k  | `f64`  |                   60,160 |                 85,720 |               0.0733% |             0.0514% | rubato (+42%)     |
| 44.1k → 48k     | `f32`  |                  102,680 |                108,760 |               0.0429% |             0.0405% | rubato (+6%)      |
| 44.1k → 48k     | `f64`  |                   75,430 |                 78,960 |               0.0585% |             0.0558% | rubato (+5%)      |
| 44.1k → 96k     | `f32`  |                  146,020 |                148,050 |               0.0302% |             0.0298% | rubato (+1%)      |
| 44.1k → 96k     | `f64`  |                   98,770 |                103,240 |               0.0446% |             0.0427% | rubato (+5%)      |
| 22.05k → 22.05k | `f32`  |                2,252,100 |                203,610 |               0.0010% |             0.0108% | ardftsrc (+1006%) |
| 22.05k → 22.05k | `f64`  |                2,127,500 |                130,270 |               0.0010% |             0.0169% | ardftsrc (+1533%) |
| 22.05k → 48k    | `f32`  |                  136,330 |                158,080 |               0.0162% |             0.0139% | rubato (+16%)     |
| 22.05k → 48k    | `f64`  |                   94,330 |                106,020 |               0.0234% |             0.0208% | rubato (+12%)     |
| 22.05k → 96k    | `f32`  |                  169,650 |                190,590 |               0.0130% |             0.0116% | rubato (+12%)     |
| 22.05k → 96k    | `f64`  |                  112,600 |                122,260 |               0.0196% |             0.0180% | rubato (+9%)      |
| 96k → 22.05k    | `f32`  |                   30,500 |                 44,660 |               0.3148% |             0.2149% | rubato (+46%)     |
| 96k → 22.05k    | `f64`  |                   21,590 |                 32,210 |               0.4447% |             0.2980% | rubato (+49%)     |
| 96k → 48k       | `f32`  |                  100,860 |                140,050 |               0.0952% |             0.0686% | rubato (+39%)     |
| 96k → 48k       | `f64`  |                   67,820 |                 91,790 |               0.1416% |             0.1046% | rubato (+35%)     |
| 96k → 96k       | `f32`  |                2,159,300 |                202,340 |               0.0044% |             0.0475% | ardftsrc (+967%)  |
| 96k → 96k       | `f64`  |                2,141,700 |                130,100 |               0.0045% |             0.0738% | ardftsrc (+1546%) |

overhead = input-rate / throughput

### Bench Memory Usage (`f32`)

| Implementation | Peak memory (bytes)    |
| -------------- | ---------------------- |
| rubato         | 31,130,816             |
| ardftsrc-rs    | 31,409,344             |

## Test 2: Good

```
ardftsrc: f64, preset=good, quality=1878, bandwidth=0.911
```

### HydrogenAudio Scores 

**ardftsrc vs ideal:**
https://src.hydrogenaudio.org/compareresults?id1=e12d7fe0-dfa2-4c49-bbdd-51c16a931cb5&id2=0

### Bench

Overhead is appropriate for realtime use, although the bursty nature of chunk processing may require off-thread resampling.

| Scenario        | Type | ardftsrc (kilosamples/s) | ardftsrc overhead (%) |
| --------------- | ---- | -----------------------: | --------------------: |
| 44.1k → 22.05k  | `f64`  |                   15,154 |               0.2910% |
| 44.1k → 48k     | `f64`  |                   51,471 |               0.0857% |
| 44.1k → 96k     | `f64`  |                   73,150 |               0.0603% |
| 22.05k → 22.05k | `f64`  |                2,027,900 |               0.0011% |
| 22.05k → 48k    | `f64`  |                   62,787 |               0.0351% |
| 22.05k → 96k    | `f64`  |                   79,562 |               0.0277% |
| 96k → 22.05k    | `f64`  |                   11,699 |               0.8206% |
| 96k → 48k       | `f64`  |                   16,311 |               0.5886% |
| 96k → 96k       | `f64`  |                2,115,400 |               0.0045% |

overhead = input-rate / throughput

## Test 3: High

```
ardftsrc: f64, preset=high, quality=73622, bandwidth=0.987
```

### HydrogenAudio Scores

**ardftsrc vs ideal:**
https://src.hydrogenaudio.org/compareresults?id1=43a72723-7f35-4318-bbd1-44cdfaa6df88&id2=0

### Bench

Still appropriate for realtime on systems with ample compute as long as we can move the resampling off-thread.

| Scenario        | Type | ardftsrc (kilosamples/s) | ardftsrc overhead (%) |
| --------------- | ---- | -----------------------: | --------------------: |
| 44.1k → 22.05k  | `f64`  |                    1,450 |               3.0405% |
| 44.1k → 48k     | `f64`  |                    4,923 |               0.8958% |
| 44.1k → 96k     | `f64`  |                    7,829 |               0.5633% |
| 22.05k → 22.05k | `f64`  |                  832,910 |               0.0026% |
| 22.05k → 48k    | `f64`  |                    4,288 |               0.5142% |
| 22.05k → 96k    | `f64`  |                    6,656 |               0.3313% |
| 96k → 22.05k    | `f64`  |                      609 |              15.7757% |
| 96k → 48k       | `f64`  |                    2,285 |               4.2008% |
| 96k → 96k       | `f64`  |                1,687,100 |               0.0057% |

## Test 4: Extreme

```
ardftsrc: f64, preset=extreme, quality=524514, bandwidth=0.995
```

### HydrogenAudio Scores

**ardftsrc vs ideal:**
https://src.hydrogenaudio.org/compareresults?id1=dbdbdd66-d8b8-4b8b-b217-b71162cb1f2f&id2=0

### Bench

These parameters, despite producing excellent quality results are not appropriate for realtime and should only be used in an offline workflow.

| Scenario        | Type | ardftsrc (kilosamples/s) | ardftsrc overhead (%) |
| --------------- | ---- | -----------------------: | --------------------: |
| 44.1k → 22.05k  | `f64`  |                      297 |              14.8260% |
| 44.1k → 48k     | `f64`  |                    1,281 |               3.4434% |
| 44.1k → 96k     | `f64`  |                    2,377 |               1.8556% |
| 22.05k → 22.05k | `f64`  |                  147,490 |               0.0150% |
| 22.05k → 48k    | `f64`  |                    1,126 |               1.9590% |
| 22.05k → 96k    | `f64`  |                    1,903 |               1.1589% |
| 96k → 22.05k    | `f64`  |                      140 |              68.7531% |
| 96k → 48k       | `f64`  |                      487 |              19.7190% |
| 96k → 96k       | `f64`  |                  435,520 |               0.0220% |

# Test 5: Goodhart's Law

*“When a measure becomes a target, it ceases to be a good measure.”*

```
ardftsrc: f64 --quality=61656210 --bandwidth=0.9951796875           
```

https://src.hydrogenaudio.org/compareresults?id1=f5d9a9c0-0019-43d4-8b39-6dba547fed98&id2=0

This was not benchmarked. It's stupidly slow, but it looks pretty!


# Realtime Assessment

FFT's are naturally bursty. This measures the amount of time required to process a full chunk.
For live / realtime use, a resampler should ideally use a very small fraction of the total
inter-sample budget (the peroid of the sample rate). 

While ardftsrc-rs has enough *throughput* for realtime use, the burstly nature of FFT's
means that it's chunk resamplers cannot be used for on-thread resampling between samples.

Realtime use of ardftsrc-rs requires using `RealtimeResampler` to use off-thread resampling.
Enable the `realtime` feature to use the real-time resampler. 

*Chunk Resampler chunk time*

| scenario               | preset | type | chunk time (µs) | budget (µs) |
| :--------------------- | :----- | :--- | --------------: | ----------: |
| 44.1 kHz -> 22.05 kHz  | fast   | f32  |          9.7287 |       45.35 |
| 96 kHz -> 22.05 kHz    | fast   | f32  |          16.003 |       45.35 |
| 22.05 kHz -> 48 kHz    | fast   | f32  |          9.5585 |       20.83 |
| 44.1 kHz -> 48 kHz     | fast   | f32  |          12.318 |       20.83 |
| 96 kHz -> 48 kHz       | fast   | f32  |          5.1023 |       20.83 |
| 22.05 kHz -> 96 kHz    | fast   | f32  |          16.254 |       10.42 |
| 44.1 kHz -> 96 kHz     | fast   | f32  |          18.751 |       10.42 |
| 44.1 kHz -> 22.05 kHz  | fast   | f64  |          15.093 |       45.35 |
| 96 kHz -> 22.05 kHz    | fast   | f64  |          23.123 |       45.35 |
| 22.05 kHz -> 48 kHz    | fast   | f64  |          13.493 |       20.83 |
| 44.1 kHz -> 48 kHz     | fast   | f64  |          17.217 |       20.83 |
| 96 kHz -> 48 kHz       | fast   | f64  |          7.8326 |       20.83 |
| 22.05 kHz -> 96 kHz    | fast   | f64  |          23.438 |       10.42 |
| 44.1 kHz -> 96 kHz     | fast   | f64  |          26.005 |       10.42 |
| 22.05 kHz -> 22.05 kHz | good   | f64  |          3.3358 |       45.35 |
| 44.1 kHz -> 22.05 kHz  | good   | f64  |          200.15 |       45.35 |
| 96 kHz -> 22.05 kHz    | good   | f64  |          101.87 |       45.35 |
| 22.05 kHz -> 48 kHz    | good   | f64  |          58.623 |       20.83 |
| 44.1 kHz -> 48 kHz     | good   | f64  |          73.415 |       20.83 |
| 96 kHz -> 48 kHz       | good   | f64  |          103.49 |       20.83 |
| 22.05 kHz -> 96 kHz    | good   | f64  |          102.54 |       10.42 |
| 44.1 kHz -> 96 kHz     | good   | f64  |          111.33 |       10.42 |
| 96 kHz -> 96 kHz       | good   | f64  |          3.3233 |       10.42 |

# Misc Observations to follow up on:

1. Rubato could benefit from a fast-path no-op for matching sample rates.
2. Rubato could benefit from synthetic pre/post samples.
3. Expanding this analysis with other test suites other than HydrogenAudio SRC would be a good idea. 
