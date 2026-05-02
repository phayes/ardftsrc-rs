# Performance Testing

This doc compares ardftsrc-rs against the best-in-class sinc resampler rubato. 
It's meant to help gauge where ardftsrc-rs might fit in the rust-audio ecosystem.

## Summary

1. rubato is faster than ardftsrc in all instances, even when rubato is running at "high quality sync"
2. at higher quality levels (> fast) ardftsrc beats out rubato in quality
3. pre-ringing is a minor issue for ardftsrc. This can be ameliorated by introducing a "phase" param (not done yet)
4. They both use about the same amount of memory, but ardftsrc has pathological RSS metrics. 
   It's doing something that's making the allocator very unhappy. Need to track this down.
5. rubato f64 performs significantly better than rubato f32. We should investigate using rubato with f64 even in a f32 pipeline (f32 -> f64 -> rubato::<f64> -> f64 -> f32). 

## Conclusion

ardftsrc works best as an offline resampler, or as a realtime resampler that prioritizes quality over speed. 

ardftsrc is not competitive against rubato in terms of raw speed, but beats it out in quality (with the exception of pre-ringing) when running at higher quality levels. A lower quality settings, ardftsrc might benefit from a low-pass filter. 

## Test 1: Fast

ardftsrc: f32, quality=512, bandwidth=0.95
rubato:   f32, FixedSync, chunk-size=512, sub-chunks=1, FixedSync::Both

### Quality

https://src.hydrogenaudio.org/compareresults?id1=9167aa7f-92f4-4953-9642-8a472b4115e9&id2=3e45fdeb-152f-4abd-97d6-1848b86243c8

1. Spectogram: both display significant aliasing, but ardftsrc-rs aliasing is mildly worse. ardftsrc-rs also produces noise at audible frequencies when resampling frequencies above 20 kHz. 
2. Pre-ringing: rubato wins out (24% vs 19.7%)
3. Gapless Sine: ardftsrc wins (62% vs 5%)
4. All other metrics roughly equivalent. 

Note: Using f64, neither produce any noticable aliasing, however ardftsrc-rs still produces noise at high frequencies: https://src.hydrogenaudio.org/compareresults?id1=1cba453d-a6c2-4257-b4de-289a8cb363cf&id2=6b05a1f8-87db-4e3e-8aa0-aefacfda7a3e

Overall, ardftsrc is not competitive with rubato at settings tuned for speed over quality.

### Bench

| Case                   | rubato (samples/s) | ardftsrc-rs (samples/s) | rubato overhead (%) | ardftsrc-rs overhead (%) |
| ---------------------- | ------------------ | ----------------------- | ------------------- | ------------------------ |
| example 44.1k → 22.05k |        139,150,000 |              97,222,000 |             0.0317% |                  0.0454% |
| example 44.1k → 48k    |        115,660,000 |             101,000,000 |             0.0381% |                  0.0437% |
| example 44.1k → 96k    |        154,350,000 |             131,330,000 |             0.0286% |                  0.0336% |
| sweep 22.05k → 48k     |        155,250,000 |             144,530,000 |             0.0142% |                  0.0153% |
| sweep 22.05k → 96k     |        184,570,000 |             177,110,000 |             0.0119% |                  0.0125% |
| sweep f32 96k → 22.05k |         43,883,000 |              36,429,000 |             0.2188% |                  0.2636% |
| sweep f32 96k → 48k    |        136,850,000 |             116,620,000 |             0.0702% |                  0.0823% |

overhead = input-rate / throughput

### Bench Memory Usage

| Implementation | Max RSS (bytes) | Peak memory (bytes)    |
| -------------- | --------------- | ---------------------- |
| rubato         | 45,858,816      | 27,706,560             |
| ardftsrc-rs    | 565,624,832     | 28,165,376             |

## Test 2: Good

ardftsrc: f64, quality=2048, bandwidth=0.95
rubato:   f64, FixedSync, chunk-size=512, sub-chunks=1, FixedSync::Both

### Quality

https://src.hydrogenaudio.org/compareresults?id1=b587413c-4311-4018-ad05-266dab2daaf5&id2=6b05a1f8-87db-4e3e-8aa0-aefacfda7a3e

ardftsrc beats rubato in all scores (with the exception of pre-ringing).

### Bench

| Case                   | ardftsrc-rs (samples/s) | ardftsrc-rs overhead (%) | rubato (samples/s) | rubato overhead (%) |
| ---------------------- | ----------------------- | ------------------------ | ------------------ | ------------------- |
| example 44.1k → 22.05k | 52,309,000              | 0.0843%                  | 89,406,000         | 0.0493%             |
| example 44.1k → 48k    | 54,675,000              | 0.0806%                  | 80,827,000         | 0.0545%             |
| example 44.1k → 96k    | 69,586,000              | 0.0634%                  | 103,330,000        | 0.0427%             |
| sweep 22.05k → 48k     | 75,587,000              | 0.0292%                  | 103,830,000        | 0.0212%             |
| sweep 22.05k → 96k     | 90,795,000              | 0.0243%                  | 119,540,000        | 0.0184%             |
| sweep f32 96k → 22.05k | 17,410,000              | 0.5514%                  | 31,746,000         | 0.3024%             |
| sweep f32 96k → 48k    | 58,072,000              | 0.1653%                  | 88,460,000         | 0.1085%             |

overhead = input-rate / throughput

| Implementation | Max RSS (bytes) | Peak memory (bytes) |
| -------------- | --------------: | ------------------: |
| ardftsrc-rs    |     566,099,968 |          28,050,624 |
| rubato         |      41,877,504 |          25,314,496 |


## Test 3: Best

ardftsrc: f64, quality=618568, bandwidth=0.995
rubato:   not included, "Test 2: Good" was maximum quality

### Quality

https://src.hydrogenaudio.org/compareresults?id1=81504c6e-19ba-4d3e-b081-5f283b9ceaab&id2=0

Excellent quality. Link compares to a theoretical "perfect" result.

### Bench

These parameters, despite producing excellent quality results (still issue with pre-ringing)
are not appropriate for realtime and should only be used in an offline workflow.

| Case                   | ardftsrc-rs (samples/s) | ardftsrc-rs overhead (%) |
| ---------------------- | ----------------------- | ------------------------ |
| example 44.1k → 22.05k | 508,480                 | 8.67%                    |
| example 44.1k → 48k    | 1,992,100               | 2.21%                    |
| example 44.1k → 96k    | 3,025,600               | 1.46%                    |
| sweep 22.05k → 48k     | 1,384,000               | 1.59%                    |
| sweep 22.05k → 96k     | 1,814,900               | 1.21%                    |
| sweep f32 96k → 22.05k | 260,380                 | 36.87%                   |
| sweep f32 96k → 48k    | 867,400                 | 11.07%                   |

| Implementation | Max RSS (bytes) | Peak memory (bytes) |
| -------------- | --------------: | ------------------: |
| ardftsrc-rs    |     765,034,496 |          27,640,960 |
