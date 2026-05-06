# Performance Testing

This doc compares ardftsrc-rs against the best-in-class sinc resampler rubato. 
It's meant to help gauge where ardftsrc-rs might fit in the rust-audio ecosystem.

## Summary

1. Rubato is faster than ardftsrc in all instances, even when rubato is running at "high quality sync"
2. At higher quality levels (> fast) ardftsrc beats out rubato in quality
3. Pre-ringing is an issue for ardftsrc. This can be ameliorated by introducing a "phase" param (not done yet)
4. They both use about the same amount of memory. 
5. Rubato f64 performs significantly better than rubato f32. We should investigate using rubato with f64 even in a f32 pipeline (f32 -> f64 -> rubato::<f64> -> f64 -> f32). 
6. Rubato could benifit from a fast-path no-op for matching sample rates.

## Conclusion

ardftsrc works best as an offline resampler, or as a realtime resampler that prioritizes quality over speed. 

ardftsrc is not competitive against rubato in terms of raw speed, but beats it out in quality (with the exception of pre-ringing) when running at higher quality levels. A lower quality settings, ardftsrc might benefit from a low-pass filter. 

## Test 1: Fast

ardftsrc: --preset=fast, quality=512, bandwidth=0.8323
rubato:   FixedSync, chunk-size=512, sub-chunks=1, FixedSync::Both

### Quality

https://src.hydrogenaudio.org/compareresults?id1=9167aa7f-92f4-4953-9642-8a472b4115e9&id2=3e45fdeb-152f-4abd-97d6-1848b86243c8

1. Spectogram: both display significant aliasing, but ardftsrc-rs aliasing is mildly worse. ardftsrc-rs also produces noise at audible frequencies when resampling frequencies above 20 kHz. 
2. Pre-ringing: rubato wins out (24% vs 19.7%)
3. Gapless Sine: ardftsrc wins (62% vs 5%)
4. All other metrics roughly equivalent. 

Note: Using f64, neither produce any noticable aliasing, however ardftsrc-rs still produces noise at high frequencies: https://src.hydrogenaudio.org/compareresults?id1=1cba453d-a6c2-4257-b4de-289a8cb363cf&id2=6b05a1f8-87db-4e3e-8aa0-aefacfda7a3e

Overall, ardftsrc is not competitive with rubato at settings tuned for speed over quality.

### Bench

| Scenario        | Type | ardftsrc (kilosamples/s) | rubato (kilosamples/s) | ardftsrc overhead (%) | rubato overhead (%) | Faster            |
| --------------- | ---- | -----------------------: | ---------------------: | --------------------: | ------------------: | ----------------- |
| 44.1k → 22.05k  | f32  |                   89,780 |                124,060 |               0.0491% |             0.0355% | rubato (+38%)     |
| 44.1k → 22.05k  | f64  |                   60,160 |                 85,720 |               0.0733% |             0.0514% | rubato (+42%)     |
| 44.1k → 48k     | f32  |                  102,680 |                108,760 |               0.0429% |             0.0405% | rubato (+6%)      |
| 44.1k → 48k     | f64  |                   75,430 |                 78,960 |               0.0585% |             0.0558% | rubato (+5%)      |
| 44.1k → 96k     | f32  |                  146,020 |                148,050 |               0.0302% |             0.0298% | rubato (+1%)      |
| 44.1k → 96k     | f64  |                   98,770 |                103,240 |               0.0446% |             0.0427% | rubato (+5%)      |
| 22.05k → 22.05k | f32  |                2,252,100 |                203,610 |               0.0010% |             0.0108% | ardftsrc (+1006%) |
| 22.05k → 22.05k | f64  |                2,127,500 |                130,270 |               0.0010% |             0.0169% | ardftsrc (+1533%) |
| 22.05k → 48k    | f32  |                  136,330 |                158,080 |               0.0162% |             0.0139% | rubato (+16%)     |
| 22.05k → 48k    | f64  |                   94,330 |                106,020 |               0.0234% |             0.0208% | rubato (+12%)     |
| 22.05k → 96k    | f32  |                  169,650 |                190,590 |               0.0130% |             0.0116% | rubato (+12%)     |
| 22.05k → 96k    | f64  |                  112,600 |                122,260 |               0.0196% |             0.0180% | rubato (+9%)      |
| 96k → 22.05k    | f32  |                   30,500 |                 44,660 |               0.3148% |             0.2149% | rubato (+46%)     |
| 96k → 22.05k    | f64  |                   21,590 |                 32,210 |               0.4447% |             0.2980% | rubato (+49%)     |
| 96k → 48k       | f32  |                  100,860 |                140,050 |               0.0952% |             0.0686% | rubato (+39%)     |
| 96k → 48k       | f64  |                   67,820 |                 91,790 |               0.1416% |             0.1046% | rubato (+35%)     |
| 96k → 96k       | f32  |                2,159,300 |                202,340 |               0.0044% |             0.0475% | ardftsrc (+967%)  |
| 96k → 96k       | f64  |                2,141,700 |                130,100 |               0.0045% |             0.0738% | ardftsrc (+1546%) |


overhead = input-rate / throughput

### Bench Memory Usage (f32)

| Implementation | Peak memory (bytes)    |
| -------------- | ---------------------- |
| rubato         | 31,130,816             |
| ardftsrc-rs    | 31,409,344             |

## Test 2: Good

ardftsrc: f64, preset=good, quality=2048, bandwidth=0.95

### Quality

https://src.hydrogenaudio.org/compareresults?id1=b587413c-4311-4018-ad05-266dab2daaf5&id2=6b05a1f8-87db-4e3e-8aa0-aefacfda7a3e

ardftsrc beats rubato in all scores (with the exception of pre-ringing).

### Bench

| Scenario        | Type | ardftsrc (kilosamples/s) | ardftsrc overhead (%) |
| --------------- | ---- | -----------------------: | --------------------: |
| 44.1k → 22.05k  | f64  |                   35,411 |               0.1245% |
| 44.1k → 48k     | f64  |                   52,587 |               0.0839% |
| 44.1k → 96k     | f64  |                   73,865 |               0.0597% |
| 22.05k → 22.05k | f64  |                2,078,600 |               0.0011% |
| 22.05k → 48k    | f64  |                   63,934 |               0.0345% |
| 22.05k → 96k    | f64  |                   79,647 |               0.0277% |
| 96k → 22.05k    | f64  |                   11,809 |               0.8130% |
| 96k → 48k       | f64  |                   43,299 |               0.2217% |
| 96k → 96k       | f64  |                2,108,500 |               0.0046% |

overhead = input-rate / throughput


## Test 3: High

ardftsrc: f64, preset=high, quality=65536, bandwidth=0.97

### Quality

https://src.hydrogenaudio.org/compareresults?id1=81504c6e-19ba-4d3e-b081-5f283b9ceaab&id2=0

Excellent quality. Link compares to a theoretical "perfect" result.

### Bench

These parameters, despite producing excellent quality results (still issue with pre-ringing)
are not appropriate for realtime and should only be used in an offline workflow.

| Scenario        | Type | ardftsrc (kilosamples/s) | ardftsrc overhead (%) |
| --------------- | ---- | -----------------------: | --------------------: |
| 44.1k → 22.05k  | f64  |                    2,106 |               2.0948% |
| 44.1k → 48k     | f64  |                    6,377 |               0.6913% |
| 44.1k → 96k     | f64  |                   10,685 |               0.4127% |
| 22.05k → 22.05k | f64  |                  764,040 |               0.0029% |
| 22.05k → 48k    | f64  |                    5,600 |               0.3937% |
| 22.05k → 96k    | f64  |                    8,931 |               0.2469% |
| 96k → 22.05k    | f64  |                      729 |              13.1687% |
| 96k → 48k       | f64  |                    3,735 |               2.5703% |
| 96k → 96k       | f64  |                1,694,100 |               0.0057% |

## Test 4: Extreme

ardftsrc: f64, preset=extreme, quality=524288, bandwidth=0.932

### Quality

https://src.hydrogenaudio.org/compareresults?id1=81504c6e-19ba-4d3e-b081-5f283b9ceaab&id2=0

Nearly perfect quality. Link compares to a theoretical "perfect" result.

### Bench

These parameters, despite producing excellent quality results are not appropriate for realtime and should only be used in an offline workflow.

| Scenario        | Type | ardftsrc (kilosamples/s) | ardftsrc overhead (%) |
| --------------- | ---- | -----------------------: | --------------------: |
| 44.1k → 22.05k  | f64  |                      370 |              11.9090% |
| 44.1k → 48k     | f64  |                    1,169 |               3.7725% |
| 44.1k → 96k     | f64  |                    2,080 |               2.1202% |
| 22.05k → 22.05k | f64  |                  116,980 |               0.0188% |
| 22.05k → 48k    | f64  |                      977 |               2.2569% |
| 22.05k → 96k    | f64  |                    1,556 |               1.4165% |
| 96k → 22.05k    | f64  |                      130 |              73.7255% |
| 96k → 48k       | f64  |                      579 |              16.5803% |
| 96k → 96k       | f64  |                  534,050 |               0.0180% |

# Test 5: Goodhart's Law

“When a measure becomes a target, it ceases to be a good measure.”

ardftsrc: f64 --quality=61656210 --bandwidth=0.9951796875           

This was not benchmarked, it's stupidly slow, but it looks pretty!

###  MISC NOTES


# Top planck score
cargo run --release -p ardftsrc_src_test -- --workdir=./workspace --f64 --local --quality=61656210 --bandwidth=0.99 --taper-type=planck

# Top alpha score cluster:

  score    quality    bandwidth     alpha
--------  ---------  -----------  --------
99.73475   61656210    0.9951797       3.5

99.73313   61656159    0.9951377  3.487196

99.72821   61647053    0.9961132  3.484811


OPTIMUM 512 / fast:

  "params": {
    "alpha": 2.6486085116993427,
    "bandwidth": 0.8719081527878331
  },
  "score": 69.19481931034483