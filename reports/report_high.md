# ardftsrc Quality Report: High

Revision: df1993b

## Preset Configuration

| Field     |          Value |
|-----------|----------------|
| Quality   |          73622 |
| Bandwidth |         0.9874 |
| Taper     | Cosine(3.4375) |
| Phase     |              0 |

## HydrogenAudio Test Suite

Local HydrogenAudio Test Suite run, `f64`, no high-precision backend. See <https://src.hydrogenaudio.org/> for the test suite's own scoring methodology.

| Metric               |      Score |
|----------------------|------------|
| Balanced score       |     99.32% |
| Spectrogram          |     99.40% |
| Bandwidth            |     99.08% |
| Impulse frequency    |     98.79% |
| Average impulse freq | -333.89 dB |
| Aliasing             |    100.00% |
| Pre-ringing          |     17.87% |
| Gapless              |    100.00% |
| Intermodulation      |     98.80% |
| Delay                |    100.00% |

### Spectrogram of a sweep 1 to 22 kHz

A sine sweep from 1 to 22 kHz. The ideal plot is a single strong red line. Issues with aliasing effects or filter cutoff would show as extra lines. Noise would appear as dots across the plot (instead of a black background). [96 kHz source resampled to 44 kHz]

<img src="high_sweep-1-to-44KHz-1to11sec.png" alt="Spectrogram of a sweep 1 to 22 kHz" width="50%" />

### Spectrogram of a sweep 1 to 22 kHz (extended)

The source signal included a sine sweep all the way up to 44 kHz, however when downsampled to 44 kHz the highest frequency which can be represented is 22 kHz. This plot would show if the sine above 22 kHz is filtering down into the plot, there should be nothing plotted after 10 seconds. [96 kHz source resampled to 44 kHz]

<img src="high_sweep-1-to-44KHz.png" alt="Spectrogram of a sweep 1 to 22 kHz (extended)" width="50%" />

### Aliasing

A 23 kHz sine at -4 dBFS with a white noise floor of -150 dBFS over 30 seconds. The ideal plot is a continuous line touching the -150dB line, many SRC routines engage a gradual filter at 20 kHz which would be visible on this plot. [96 kHz source resampled to 44 kHz]

<img src="high_aliasing150db.png" alt="Aliasing" width="50%" />

### Nyquist Filter

Zoomed on nyquist frequency (22.050 kHz), the bandwidth of the SRC is displayed. A SRC with 100% bandwidth, that is full frequency preservation would be a straight line of noise. [96 kHz source resampled to 44 kHz]

<img src="high_nyquist-filter.png" alt="Nyquist Filter" width="50%" />

### Intermodulation Harmonic Distortion

Two sine waves one at 64.59 Hz, -6 dBFS and the second at 6998 Hz, -18.0412 dBFS, which equals quarter the amplitude of the first sine. This test will highlight aliasing and dynamic range of processing. It will also show if dither has been applied, look for high frequency signals on the plot. [96 kHz source resampled to 44 kHz]

<img src="high_intermodulation-harmonic-distortion.png" alt="Intermodulation Harmonic Distortion" width="50%" />

### Intermodulation Harmonic Distortion (difference)

The difference between the ideal and measured signals from the Intermodulation Harmonic Distortion test. [96 kHz source resampled to 44 kHz]

<img src="high_intermodulation-harmonic-distortion-difference.png" alt="Intermodulation Harmonic Distortion (difference)" width="50%" />

### Impulse Frequency

A frequency response displaying leakage beyond the ideal frequency response. Impulses are at sample positions n so that {n mod 320} is a permutation of {0, 1, 2, . . . , 319}. All fractional differences in sample positions between input and output signal are addressed exactly once. The output signal is upsampled to 14.112 MHz and the impulse responses are added to obtain a high resolution impulse response. [96 kHz source resampled to 44 kHz]

<img src="high_impulse-frequency.png" alt="Impulse Frequency" width="50%" />

### Impulse Response

Displays the phase response, most SRC balance the response with pre and post ringing, personal preference might prefer a minimum phase response where there is no pre-ringing. The ideal response is not shown because a post-ringing representation, would not tally with the other ideal plots (phase, etc). [96 kHz source resampled to 44 kHz]

<img src="high_impulse-response.png" alt="Impulse Response" width="50%" />

### Impulse Phase

The actual phase across the frequency range. [96 kHz source resampled to 44 kHz]

<img src="high_impulse-phase.png" alt="Impulse Phase" width="50%" />

### Impulse Passband

Displays the SRC filter used close to the nyquist frequency. [96 kHz source resampled to 44 kHz]

<img src="high_impulse-passband.png" alt="Impulse Passband" width="50%" />

### Impulse Transition

A zoomed plot of the nyquist frequency showing the filter response. [96 kHz source resampled to 44 kHz]

<img src="high_impulse-transition.png" alt="Impulse Transition" width="50%" />

### Gapless Sine

A sine wave is split into two, both are resampled independently. This plot is the two signals joined back together. The blue line shows the transition after resampling. Deviation from sine wave shape likely means an audible glitch. Any value over +1 would clip if later not corrected. [96 kHz source resampled to 44 kHz]

<img src="high_gaplesstestsine.png" alt="Gapless Sine" width="50%" />

### Gapless Sine (frequency plot)

A frequency plot of the gapless sine test. [96 kHz source resampled to 44 kHz]

<img src="high_gaplesstest-frequency.png" alt="Gapless Sine (frequency plot)" width="50%" />

## THD+N

`f64` only. Broadband THD+N covers DC-Nyquist (time-domain sine fit, no FFT); audio-band THD+N is limited to 20 Hz-20000 Hz. Values at or below the -300 dB noise floor reflect the plain-`f64` analyzer's own precision ceiling.

† Gain error is below 1e-9 dB. It is below f64 round-off error.

### Summary (worst case per configuration)

| Rate pair       | Decimate | high_precision | Worst broadband THD+N (dB) | Worst audio-band THD+N (dB) | Worst gain error (dB) | Worst spur (dB) |
|-----------------|----------|----------------|----------------------------|-----------------------------|-----------------------|-----------------|
| 44100 -> 48000  | false    | off            |                    -222.86 |                     -223.57 |                   0 † |         -233.14 |
| 48000 -> 44100  | false    | off            |                    -216.05 |                     -218.43 |                   0 † |         -221.96 |
| 44100 -> 96000  | false    | off            |                    -222.86 |                     -226.00 |                   0 † |         -233.14 |
| 96000 -> 44100  | false    | off            |                    -217.81 |                     -219.92 |                   0 † |         -224.15 |
| 192000 -> 48000 | false    | off            |                    -219.27 |                     -219.27 |                   0 † |         -224.44 |
| 192000 -> 48000 | true     | off            |                    -219.27 |                     -219.27 |              -4.65e-7 |         -224.44 |
### 44100 -> 48000, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -236.62 |               -239.86 |             0 † |       -241.71 @ 20 |
|        20 |              -20 |              -236.63 |               -239.87 |             0 † |       -241.71 @ 20 |
|        20 |              -60 |              -236.62 |               -239.87 |             0 † |       -241.71 @ 20 |
|       100 |               -1 |              -231.31 |               -231.31 |             0 † |      -236.46 @ 100 |
|       100 |              -20 |              -231.31 |               -231.31 |             0 † |      -236.46 @ 100 |
|       100 |              -60 |              -231.30 |               -231.30 |             0 † |      -236.46 @ 100 |
|      1000 |               -1 |              -229.36 |               -229.38 |             0 † |     -234.57 @ 1000 |
|      1000 |              -20 |              -229.35 |               -229.37 |             0 † |     -234.57 @ 1000 |
|      1000 |              -60 |              -229.35 |               -229.37 |             0 † |     -234.56 @ 1000 |
|      5000 |               -1 |              -236.08 |               -236.76 |             0 † |     -245.39 @ 3954 |
|      5000 |              -20 |              -235.96 |               -236.62 |             0 † |     -245.39 @ 3954 |
|      5000 |              -60 |              -236.06 |               -236.74 |             0 † |     -245.39 @ 3954 |
|     10000 |               -1 |              -230.06 |               -230.76 |             0 † |     -239.37 @ 1046 |
|     10000 |              -20 |              -230.06 |               -230.76 |             0 † |     -239.37 @ 1046 |
|     10000 |              -60 |              -230.05 |               -230.75 |             0 † |     -239.37 @ 1046 |
|     15000 |               -1 |              -224.29 |               -224.91 |             0 † |    -235.75 @ 15000 |
|     15000 |              -20 |              -224.29 |               -224.91 |             0 † |    -235.75 @ 15000 |
|     15000 |              -60 |              -224.29 |               -224.91 |             0 † |    -235.75 @ 15000 |
|     18000 |               -1 |              -222.88 |               -223.86 |             0 † |    -233.14 @ 18000 |
|     18000 |              -20 |              -222.88 |               -223.86 |             0 † |    -233.14 @ 18000 |
|     18000 |              -60 |              -222.88 |               -223.86 |             0 † |    -233.14 @ 18000 |
|     20000 |               -1 |              -222.86 |               -223.57 |             0 † |    -233.35 @ 19046 |
|     20000 |              -20 |              -222.86 |               -223.57 |             0 † |    -233.35 @ 19046 |
|     20000 |              -60 |              -222.86 |               -223.57 |             0 † |    -233.35 @ 19046 |
### 48000 -> 44100, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -229.37 |               -232.61 |             0 † |       -234.45 @ 20 |
|        20 |              -20 |              -229.37 |               -232.61 |             0 † |       -234.45 @ 20 |
|        20 |              -60 |              -229.37 |               -232.61 |             0 † |       -234.45 @ 20 |
|       100 |               -1 |              -239.73 |               -239.73 |             0 † |      -244.89 @ 100 |
|       100 |              -20 |              -239.73 |               -239.73 |             0 † |      -244.89 @ 100 |
|       100 |              -60 |              -239.73 |               -239.73 |             0 † |      -244.89 @ 100 |
|      1000 |               -1 |              -230.57 |               -230.59 |             0 † |     -235.80 @ 1000 |
|      1000 |              -20 |              -230.57 |               -230.59 |             0 † |     -235.80 @ 1000 |
|      1000 |              -60 |              -230.57 |               -230.59 |             0 † |     -235.80 @ 1000 |
|      5000 |               -1 |              -235.84 |               -237.15 |             0 † |    -245.76 @ 13554 |
|      5000 |              -20 |              -235.84 |               -237.15 |             0 † |    -245.76 @ 13554 |
|      5000 |              -60 |              -235.84 |               -237.14 |             0 † |    -245.76 @ 13554 |
|     10000 |               -1 |              -223.83 |               -223.90 |             0 † |    -230.17 @ 10000 |
|     10000 |              -20 |              -223.83 |               -223.90 |             0 † |    -230.17 @ 10000 |
|     10000 |              -60 |              -223.83 |               -223.90 |             0 † |    -230.17 @ 10000 |
|     15000 |               -1 |              -221.56 |               -221.58 |             0 † |    -229.00 @ 15000 |
|     15000 |              -20 |              -221.56 |               -221.58 |             0 † |    -229.00 @ 15000 |
|     15000 |              -60 |              -221.56 |               -221.58 |             0 † |    -229.00 @ 15000 |
|     18000 |               -1 |              -224.39 |               -224.76 |             0 † |    -234.29 @ 17058 |
|     18000 |              -20 |              -224.39 |               -224.76 |             0 † |    -234.29 @ 17058 |
|     18000 |              -60 |              -224.39 |               -224.76 |             0 † |    -234.29 @ 17058 |
|     20000 |               -1 |              -216.05 |               -218.43 |             0 † |    -221.96 @ 20000 |
|     20000 |              -20 |              -216.05 |               -218.43 |             0 † |    -221.96 @ 20000 |
|     20000 |              -60 |              -216.05 |               -218.43 |             0 † |    -221.96 @ 20000 |
### 44100 -> 96000, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -236.63 |               -239.87 |             0 † |       -241.71 @ 20 |
|        20 |              -20 |              -236.62 |               -239.86 |             0 † |       -241.71 @ 20 |
|        20 |              -60 |              -236.63 |               -239.87 |             0 † |       -241.71 @ 20 |
|       100 |               -1 |              -231.31 |               -231.31 |             0 † |      -236.46 @ 100 |
|       100 |              -20 |              -231.29 |               -231.29 |             0 † |      -236.46 @ 100 |
|       100 |              -60 |              -231.28 |               -231.28 |             0 † |      -236.46 @ 100 |
|      1000 |               -1 |              -229.36 |               -229.40 |             0 † |     -234.57 @ 1000 |
|      1000 |              -20 |              -229.35 |               -229.39 |             0 † |     -234.57 @ 1000 |
|      1000 |              -60 |              -228.81 |               -228.85 |             0 † |     -234.57 @ 1000 |
|      5000 |               -1 |              -236.08 |               -242.77 |             0 † |    -245.40 @ 44046 |
|      5000 |              -20 |              -235.92 |               -242.08 |             0 † |    -245.40 @ 44046 |
|      5000 |              -60 |              -235.97 |               -242.28 |             0 † |    -245.40 @ 44046 |
|     10000 |               -1 |              -230.06 |               -236.84 |             0 † |    -239.37 @ 46954 |
|     10000 |              -20 |              -230.06 |               -236.84 |             0 † |    -239.37 @ 46954 |
|     10000 |              -60 |              -230.06 |               -236.84 |             0 † |    -239.37 @ 46954 |
|     15000 |               -1 |              -224.29 |               -227.23 |             0 † |    -235.76 @ 15000 |
|     15000 |              -20 |              -224.29 |               -227.23 |             0 † |    -235.76 @ 15000 |
|     15000 |              -60 |              -224.29 |               -227.22 |             0 † |    -235.76 @ 15000 |
|     18000 |               -1 |              -222.88 |               -226.23 |             0 † |    -233.14 @ 18000 |
|     18000 |              -20 |              -222.88 |               -226.22 |             0 † |    -233.14 @ 18000 |
|     18000 |              -60 |              -222.88 |               -226.22 |             0 † |    -233.14 @ 18000 |
|     20000 |               -1 |              -222.86 |               -226.00 |             0 † |    -233.35 @ 19046 |
|     20000 |              -20 |              -222.86 |               -226.00 |             0 † |    -233.35 @ 19046 |
|     20000 |              -60 |              -222.86 |               -226.00 |             0 † |    -233.35 @ 19046 |
### 96000 -> 44100, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -228.04 |               -231.28 |             0 † |       -233.13 @ 20 |
|        20 |              -20 |              -228.04 |               -231.28 |             0 † |       -233.13 @ 20 |
|        20 |              -60 |              -228.04 |               -231.28 |             0 † |       -233.13 @ 20 |
|       100 |               -1 |              -221.44 |               -221.44 |             0 † |      -226.59 @ 100 |
|       100 |              -20 |              -221.44 |               -221.44 |             0 † |      -226.59 @ 100 |
|       100 |              -60 |              -221.44 |               -221.44 |             0 † |      -226.59 @ 100 |
|      1000 |               -1 |              -227.28 |               -227.29 |             0 † |     -232.48 @ 1000 |
|      1000 |              -20 |              -227.28 |               -227.29 |             0 † |     -232.48 @ 1000 |
|      1000 |              -60 |              -227.28 |               -227.29 |             0 † |     -232.48 @ 1000 |
|      5000 |               -1 |              -224.11 |               -224.18 |             0 † |     -229.56 @ 5000 |
|      5000 |              -20 |              -224.11 |               -224.18 |             0 † |     -229.56 @ 5000 |
|      5000 |              -60 |              -224.11 |               -224.18 |             0 † |     -229.56 @ 5000 |
|     10000 |               -1 |              -229.82 |               -230.09 |             0 † |    -239.74 @ 15546 |
|     10000 |              -20 |              -229.82 |               -230.09 |             0 † |    -239.74 @ 15546 |
|     10000 |              -60 |              -229.82 |               -230.09 |             0 † |    -239.74 @ 15546 |
|     15000 |               -1 |              -221.56 |               -221.58 |             0 † |    -229.00 @ 15000 |
|     15000 |              -20 |              -221.56 |               -221.58 |             0 † |    -229.00 @ 15000 |
|     15000 |              -60 |              -221.56 |               -221.58 |             0 † |    -229.00 @ 15000 |
|     18000 |               -1 |              -224.39 |               -224.76 |             0 † |    -234.29 @ 17058 |
|     18000 |              -20 |              -224.39 |               -224.76 |             0 † |    -234.29 @ 17058 |
|     18000 |              -60 |              -224.39 |               -224.76 |             0 † |    -234.29 @ 17058 |
|     20000 |               -1 |              -217.81 |               -219.92 |             0 † |    -224.15 @ 20000 |
|     20000 |              -20 |              -217.81 |               -219.92 |             0 † |    -224.15 @ 20000 |
|     20000 |              -60 |              -217.81 |               -219.92 |             0 † |    -224.15 @ 20000 |
### 192000 -> 48000, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -222.31 |               -224.98 |             0 † |       -227.38 @ 20 |
|        20 |              -20 |              -222.31 |               -224.98 |             0 † |       -227.38 @ 20 |
|        20 |              -60 |              -222.31 |               -224.98 |             0 † |       -227.38 @ 20 |
|       100 |               -1 |              -224.35 |               -224.35 |             0 † |      -229.51 @ 100 |
|       100 |              -20 |              -224.35 |               -224.35 |             0 † |      -229.51 @ 100 |
|       100 |              -60 |              -224.35 |               -224.35 |             0 † |      -229.51 @ 100 |
|      1000 |               -1 |              -219.27 |               -219.27 |             0 † |     -224.44 @ 1000 |
|      1000 |              -20 |              -219.27 |               -219.27 |             0 † |     -224.44 @ 1000 |
|      1000 |              -60 |              -219.27 |               -219.27 |             0 † |     -224.44 @ 1000 |
|      5000 |               -1 |              -219.91 |               -219.92 |             0 † |     -225.18 @ 5000 |
|      5000 |              -20 |              -219.91 |               -219.92 |             0 † |     -225.18 @ 5000 |
|      5000 |              -60 |              -219.91 |               -219.92 |             0 † |     -225.18 @ 5000 |
|     10000 |               -1 |              -220.40 |               -220.47 |             0 † |    -226.06 @ 10000 |
|     10000 |              -20 |              -220.40 |               -220.47 |             0 † |    -226.06 @ 10000 |
|     10000 |              -60 |              -220.40 |               -220.47 |             0 † |    -226.06 @ 10000 |
|     15000 |               -1 |              -224.81 |               -225.52 |             0 † |     -237.04 @ 3716 |
|     15000 |              -20 |              -224.81 |               -225.52 |             0 † |     -237.04 @ 3716 |
|     15000 |              -60 |              -224.81 |               -225.52 |             0 † |     -237.04 @ 3716 |
|     18000 |               -1 |              -223.76 |               -224.99 |             0 † |    -234.58 @ 11658 |
|     18000 |              -20 |              -223.74 |               -224.97 |             0 † |    -234.58 @ 11658 |
|     18000 |              -60 |              -223.74 |               -224.97 |             0 † |    -234.58 @ 11658 |
|     20000 |               -1 |              -220.06 |               -221.72 |             0 † |    -227.42 @ 20000 |
|     20000 |              -20 |              -220.06 |               -221.72 |             0 † |    -227.42 @ 20000 |
|     20000 |              -60 |              -220.06 |               -221.72 |             0 † |    -227.42 @ 20000 |
### 192000 -> 48000, decimate=true, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -222.31 |               -224.98 |        -2.02e-9 |       -227.38 @ 20 |
|        20 |              -20 |              -222.31 |               -224.98 |        -2.01e-9 |       -227.38 @ 20 |
|        20 |              -60 |              -222.31 |               -224.98 |        -2.02e-9 |       -227.38 @ 20 |
|       100 |               -1 |              -224.35 |               -224.35 |        -4.76e-8 |      -229.51 @ 100 |
|       100 |              -20 |              -224.35 |               -224.35 |        -4.76e-8 |      -229.51 @ 100 |
|       100 |              -60 |              -224.35 |               -224.35 |        -4.76e-8 |      -229.51 @ 100 |
|      1000 |               -1 |              -219.27 |               -219.27 |        -2.15e-7 |     -224.44 @ 1000 |
|      1000 |              -20 |              -219.27 |               -219.27 |        -2.15e-7 |     -224.44 @ 1000 |
|      1000 |              -60 |              -219.27 |               -219.27 |        -2.15e-7 |     -224.44 @ 1000 |
|      5000 |               -1 |              -219.91 |               -219.92 |        -2.28e-7 |     -225.18 @ 5000 |
|      5000 |              -20 |              -219.91 |               -219.92 |        -2.28e-7 |     -225.18 @ 5000 |
|      5000 |              -60 |              -219.91 |               -219.92 |        -2.28e-7 |     -225.18 @ 5000 |
|     10000 |               -1 |              -220.40 |               -220.47 |        -2.22e-7 |    -226.06 @ 10000 |
|     10000 |              -20 |              -220.40 |               -220.47 |        -2.22e-7 |    -226.06 @ 10000 |
|     10000 |              -60 |              -220.40 |               -220.47 |        -2.22e-7 |    -226.06 @ 10000 |
|     15000 |               -1 |              -224.81 |               -225.52 |         1.22e-7 |     -237.04 @ 3716 |
|     15000 |              -20 |              -224.81 |               -225.52 |         1.22e-7 |     -237.04 @ 3716 |
|     15000 |              -60 |              -224.81 |               -225.52 |         1.22e-7 |     -237.04 @ 3716 |
|     18000 |               -1 |              -223.76 |               -225.00 |         1.94e-7 |    -234.58 @ 11658 |
|     18000 |              -20 |              -223.76 |               -225.00 |         1.94e-7 |    -234.58 @ 11658 |
|     18000 |              -60 |              -223.76 |               -225.00 |         1.94e-7 |    -234.58 @ 11658 |
|     20000 |               -1 |              -220.06 |               -221.71 |        -4.65e-7 |    -227.42 @ 20000 |
|     20000 |              -20 |              -220.06 |               -221.71 |        -4.65e-7 |    -227.42 @ 20000 |
|     20000 |              -60 |              -220.06 |               -221.72 |        -4.65e-7 |    -227.42 @ 20000 |

## Pre-ringing

`f64` only, no high-precision backend. Every measurement is repeated with the stimulus at 4 positions within the resampler's processing chunk; tables show the worst position.

### Impulse pre-ringing

A unit impulse is resampled, and the output is scanned backward from the impulse's ideal output time. Each "Pre-ring" column is how long before that time the response first reaches the given level, relative to the response's peak (worst position per column). "Ringing freq" is the strongest frequency in the pre-ringing tail, excluding the main lobe. The transition band runs from the passband edge (unity gain below it) to the lower Nyquist frequency.

| Rate pair       | Decimate | Transition band (Hz) | Pre-ring ≥ -60 dB (ms) | Pre-ring ≥ -100 dB (ms) | Pre-ring ≥ -140 dB (ms) | Ringing freq (Hz) |
|-----------------|----------|----------------------|------------------------|-------------------------|-------------------------|-------------------|
| 44100 -> 48000  | false    |        21771 - 22050 |                   6.47 |                   22.74 |                   36.88 |             21912 |
| 48000 -> 44100  | false    |        21771 - 22050 |                   7.13 |                   22.46 |                   36.88 |             21912 |
| 44100 -> 96000  | false    |        21771 - 22050 |                   5.51 |                   21.78 |                   36.04 |             21912 |
| 96000 -> 44100  | false    |        21771 - 22050 |                   7.13 |                   22.46 |                   36.88 |             21912 |
| 192000 -> 48000 | false    |        23696 - 24000 |                   6.55 |                   20.64 |                   33.86 |             23850 |
| 192000 -> 48000 | true     |        23696 - 24000 |                   6.55 |                   20.64 |                   33.86 |             23850 |

### Pre-ringing below the rolloff

Gaussian transients whose spectra are at least -300 dB down at the passband edge (a click, the part of an impulse below the rolloff, plus tone bursts) are resampled and compared sample-by-sample against their exact ideal output. The passband gain is unity, so an ideal linear-phase resampler would return them unchanged: any residual before the transient is pre-ringing that reached frequencies below the rolloff. A best-fit constant delay is removed first, so a timing offset isn't counted as ringing. Echo columns are the largest residual in dB relative to the transient's peak, @ ms before (pre) or after (post) its center; a lead well beyond the envelope σ is a true pre-echo, ahead of the transient's own rise. Values near the THD+N section's noise floor are `f64` rounding, not leakage.

Worst below-rolloff pre-echo: **-152.4 dB**, 1.356 ms ahead of the 19739 Hz burst (192000 -> 48000, decimate=true).

| Rate pair       | Decimate | Transient      | Envelope σ (ms) | Worst pre-echo (dB @ ms) | Worst post-echo (dB @ ms) |
|-----------------|----------|----------------|-----------------|--------------------------|---------------------------|
| 44100 -> 48000  | false    | Click          |           0.062 |           -271.2 @ 0.060 |            -271.1 @ 0.065 |
| 44100 -> 48000  | false    | 10668 Hz burst |           0.124 |           -255.1 @ 0.018 |            -254.6 @ 0.024 |
| 44100 -> 48000  | false    | 18135 Hz burst |           0.413 |           -250.1 @ 0.071 |            -249.9 @ 0.012 |
| 48000 -> 44100  | false    | Click          |           0.062 |           -270.4 @ 0.064 |            -270.5 @ 0.060 |
| 48000 -> 44100  | false    | 10668 Hz burst |           0.124 |           -254.2 @ 0.019 |            -254.1 @ 0.026 |
| 48000 -> 44100  | false    | 18135 Hz burst |           0.413 |           -249.1 @ 0.042 |            -249.2 @ 0.015 |
| 44100 -> 96000  | false    | Click          |           0.062 |           -277.0 @ 0.060 |            -277.2 @ 0.065 |
| 44100 -> 96000  | false    | 10668 Hz burst |           0.124 |           -260.9 @ 0.019 |            -260.6 @ 0.023 |
| 44100 -> 96000  | false    | 18135 Hz burst |           0.413 |           -256.0 @ 0.070 |            -255.9 @ 0.013 |
| 96000 -> 44100  | false    | Click          |           0.062 |           -270.4 @ 0.061 |            -270.5 @ 0.064 |
| 96000 -> 44100  | false    | 10668 Hz burst |           0.124 |           -254.2 @ 0.027 |            -254.2 @ 0.019 |
| 96000 -> 44100  | false    | 18135 Hz burst |           0.413 |           -249.2 @ 0.015 |            -249.2 @ 0.041 |
| 192000 -> 48000 | false    | Click          |           0.057 |           -270.4 @ 0.054 |            -270.5 @ 0.061 |
| 192000 -> 48000 | false    | 11611 Hz burst |           0.114 |           -253.9 @ 0.023 |            -253.9 @ 0.019 |
| 192000 -> 48000 | false    | 19739 Hz burst |           0.380 |           -249.1 @ 0.012 |            -249.4 @ 0.040 |
| 192000 -> 48000 | true     | Click          |           0.057 |           -155.6 @ 0.002 |            -155.7 @ 0.008 |
| 192000 -> 48000 | true     | 11611 Hz burst |           0.114 |           -155.7 @ 0.002 |            -156.4 @ 0.040 |
| 192000 -> 48000 | true     | 19739 Hz burst |           0.380 |           -152.4 @ 1.356 |            -152.4 @ 1.331 |
