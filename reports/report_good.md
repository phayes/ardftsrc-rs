# ardftsrc Quality Report: Good

Version: 0.3.0
Revision: 74b3157

## Preset Configuration

| Field     |        Value |
|-----------|--------------|
| Quality   |         2304 |
| Bandwidth |       0.9750 |
| Taper     | Kaiser(30.0) |
| Phase     |            0 |

## HydrogenAudio Test Suite

Local HydrogenAudio Test Suite run, `f64`, no high-precision backend. See <https://src.hydrogenaudio.org/> for the test suite's own scoring methodology.

| Metric               |      Score |
|----------------------|------------|
| Balanced score       |     99.26% |
| Spectrogram          |     98.91% |
| Bandwidth            |     98.14% |
| Impulse frequency    |     99.54% |
| Average impulse freq | -336.44 dB |
| Aliasing             |    100.00% |
| Pre-ringing          |     18.28% |
| Gapless              |    100.00% |
| Intermodulation      |    100.00% |
| Delay                |    100.00% |

### Spectrogram of a sweep 1 to 22 kHz

A sine sweep from 1 to 22 kHz. The ideal plot is a single strong red line. Issues with aliasing effects or filter cutoff would show as extra lines. Noise would appear as dots across the plot (instead of a black background). [96 kHz source resampled to 44 kHz]

<img src="good_sweep-1-to-44KHz-1to11sec.png" alt="Spectrogram of a sweep 1 to 22 kHz" width="50%" />

### Spectrogram of a sweep 1 to 22 kHz (extended)

The source signal included a sine sweep all the way up to 44 kHz, however when downsampled to 44 kHz the highest frequency which can be represented is 22 kHz. This plot would show if the sine above 22 kHz is filtering down into the plot, there should be nothing plotted after 10 seconds. [96 kHz source resampled to 44 kHz]

<img src="good_sweep-1-to-44KHz.png" alt="Spectrogram of a sweep 1 to 22 kHz (extended)" width="50%" />

### Aliasing

A 23 kHz sine at -4 dBFS with a white noise floor of -150 dBFS over 30 seconds. The ideal plot is a continuous line touching the -150dB line, many SRC routines engage a gradual filter at 20 kHz which would be visible on this plot. [96 kHz source resampled to 44 kHz]

<img src="good_aliasing150db.png" alt="Aliasing" width="50%" />

### Nyquist Filter

Zoomed on nyquist frequency (22.050 kHz), the bandwidth of the SRC is displayed. A SRC with 100% bandwidth, that is full frequency preservation would be a straight line of noise. [96 kHz source resampled to 44 kHz]

<img src="good_nyquist-filter.png" alt="Nyquist Filter" width="50%" />

### Intermodulation Harmonic Distortion

Two sine waves one at 64.59 Hz, -6 dBFS and the second at 6998 Hz, -18.0412 dBFS, which equals quarter the amplitude of the first sine. This test will highlight aliasing and dynamic range of processing. It will also show if dither has been applied, look for high frequency signals on the plot. [96 kHz source resampled to 44 kHz]

<img src="good_intermodulation-harmonic-distortion.png" alt="Intermodulation Harmonic Distortion" width="50%" />

### Intermodulation Harmonic Distortion (difference)

The difference between the ideal and measured signals from the Intermodulation Harmonic Distortion test. [96 kHz source resampled to 44 kHz]

<img src="good_intermodulation-harmonic-distortion-difference.png" alt="Intermodulation Harmonic Distortion (difference)" width="50%" />

### Impulse Frequency

A frequency response displaying leakage beyond the ideal frequency response. Impulses are at sample positions n so that {n mod 320} is a permutation of {0, 1, 2, . . . , 319}. All fractional differences in sample positions between input and output signal are addressed exactly once. The output signal is upsampled to 14.112 MHz and the impulse responses are added to obtain a high resolution impulse response. [96 kHz source resampled to 44 kHz]

<img src="good_impulse-frequency.png" alt="Impulse Frequency" width="50%" />

### Impulse Response

Displays the phase response, most SRC balance the response with pre and post ringing, personal preference might prefer a minimum phase response where there is no pre-ringing. The ideal response is not shown because a post-ringing representation, would not tally with the other ideal plots (phase, etc). [96 kHz source resampled to 44 kHz]

<img src="good_impulse-response.png" alt="Impulse Response" width="50%" />

### Impulse Phase

The actual phase across the frequency range. [96 kHz source resampled to 44 kHz]

<img src="good_impulse-phase.png" alt="Impulse Phase" width="50%" />

### Impulse Passband

Displays the SRC filter used close to the nyquist frequency. [96 kHz source resampled to 44 kHz]

<img src="good_impulse-passband.png" alt="Impulse Passband" width="50%" />

### Impulse Transition

A zoomed plot of the nyquist frequency showing the filter response. [96 kHz source resampled to 44 kHz]

<img src="good_impulse-transition.png" alt="Impulse Transition" width="50%" />

### Gapless Sine

A sine wave is split into two, both are resampled independently. This plot is the two signals joined back together. The blue line shows the transition after resampling. Deviation from sine wave shape likely means an audible glitch. Any value over +1 would clip if later not corrected. [96 kHz source resampled to 44 kHz]

<img src="good_gaplesstestsine.png" alt="Gapless Sine" width="50%" />

### Gapless Sine (frequency plot)

A frequency plot of the gapless sine test. [96 kHz source resampled to 44 kHz]

<img src="good_gaplesstest-frequency.png" alt="Gapless Sine (frequency plot)" width="50%" />

## THD+N

`f64` only. Broadband THD+N covers DC-Nyquist (time-domain sine fit, no FFT); audio-band THD+N is limited to 20 Hz-20000 Hz. Values at or below the -300 dB noise floor reflect the plain-`f64` analyzer's own precision ceiling.

† Gain error is below 1e-9 dB. It is below f64 round-off error.

### Summary (worst case per configuration)

| Rate pair       | Decimate | high_precision | Worst broadband THD+N (dB) | Worst audio-band THD+N (dB) | Worst gain error (dB) | Worst spur (dB) |
|-----------------|----------|----------------|----------------------------|-----------------------------|-----------------------|-----------------|
| 44100 -> 48000  | false    | off            |                    -222.86 |                     -223.57 |                   0 † |         -233.14 |
| 48000 -> 44100  | false    | off            |                    -216.05 |                     -218.43 |                   0 † |         -221.96 |
| 44100 -> 96000  | false    | off            |                    -222.85 |                     -225.96 |                   0 † |         -233.14 |
| 96000 -> 44100  | false    | off            |                    -217.81 |                     -219.92 |                   0 † |         -224.15 |
| 192000 -> 48000 | false    | off            |                    -219.27 |                     -219.27 |                   0 † |         -224.45 |
| 192000 -> 48000 | true     | off            |                    -219.27 |                     -219.27 |              -4.62e-7 |         -224.45 |
### 44100 -> 48000, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -236.62 |               -239.29 |             0 † |       -241.69 @ 20 |
|        20 |              -20 |              -236.63 |               -239.29 |             0 † |       -241.69 @ 20 |
|        20 |              -60 |              -236.62 |               -239.29 |             0 † |       -241.69 @ 20 |
|       100 |               -1 |              -231.31 |               -231.31 |             0 † |      -236.46 @ 100 |
|       100 |              -20 |              -231.30 |               -231.30 |             0 † |      -236.46 @ 100 |
|       100 |              -60 |              -231.30 |               -231.30 |             0 † |      -236.46 @ 100 |
|      1000 |               -1 |              -229.36 |               -229.38 |             0 † |     -234.56 @ 1000 |
|      1000 |              -20 |              -229.33 |               -229.35 |             0 † |     -234.57 @ 1000 |
|      1000 |              -60 |              -229.35 |               -229.37 |             0 † |     -234.57 @ 1000 |
|      5000 |               -1 |              -236.08 |               -236.76 |             0 † |     -245.39 @ 3954 |
|      5000 |              -20 |              -236.03 |               -236.70 |             0 † |     -245.39 @ 3954 |
|      5000 |              -60 |              -236.08 |               -236.76 |             0 † |     -245.39 @ 3954 |
|     10000 |               -1 |              -230.06 |               -230.76 |             0 † |     -239.37 @ 1046 |
|     10000 |              -20 |              -230.06 |               -230.76 |             0 † |     -239.37 @ 1046 |
|     10000 |              -60 |              -230.06 |               -230.76 |             0 † |     -239.37 @ 1046 |
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
|        20 |               -1 |              -229.37 |               -232.03 |             0 † |       -234.44 @ 20 |
|        20 |              -20 |              -229.37 |               -232.03 |             0 † |       -234.44 @ 20 |
|        20 |              -60 |              -229.37 |               -232.03 |             0 † |       -234.44 @ 20 |
|       100 |               -1 |              -239.72 |               -239.72 |             0 † |      -244.89 @ 100 |
|       100 |              -20 |              -239.73 |               -239.73 |             0 † |      -244.89 @ 100 |
|       100 |              -60 |              -239.72 |               -239.72 |             0 † |      -244.89 @ 100 |
|      1000 |               -1 |              -230.57 |               -230.59 |             0 † |     -235.80 @ 1000 |
|      1000 |              -20 |              -230.57 |               -230.59 |             0 † |     -235.80 @ 1000 |
|      1000 |              -60 |              -230.57 |               -230.59 |             0 † |     -235.80 @ 1000 |
|      5000 |               -1 |              -235.84 |               -237.15 |             0 † |    -245.76 @ 13554 |
|      5000 |              -20 |              -235.84 |               -237.15 |             0 † |    -245.76 @ 13554 |
|      5000 |              -60 |              -235.84 |               -237.15 |             0 † |    -245.76 @ 13554 |
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
|        20 |               -1 |              -236.63 |               -239.29 |             0 † |       -241.69 @ 20 |
|        20 |              -20 |              -236.61 |               -239.26 |             0 † |       -241.69 @ 20 |
|        20 |              -60 |              -236.62 |               -239.28 |             0 † |       -241.69 @ 20 |
|       100 |               -1 |              -231.30 |               -231.30 |             0 † |      -236.46 @ 100 |
|       100 |              -20 |              -231.31 |               -231.31 |             0 † |      -236.46 @ 100 |
|       100 |              -60 |              -231.30 |               -231.30 |             0 † |      -236.46 @ 100 |
|      1000 |               -1 |              -229.36 |               -229.39 |             0 † |     -234.57 @ 1000 |
|      1000 |              -20 |              -229.29 |               -229.33 |             0 † |     -234.57 @ 1000 |
|      1000 |              -60 |              -229.28 |               -229.32 |             0 † |     -234.57 @ 1000 |
|      5000 |               -1 |              -236.08 |               -242.77 |             0 † |    -245.40 @ 44046 |
|      5000 |              -20 |              -235.64 |               -241.03 |             0 † |    -245.40 @ 44046 |
|      5000 |              -60 |              -235.72 |               -241.32 |             0 † |    -245.40 @ 44046 |
|     10000 |               -1 |              -230.06 |               -236.84 |             0 † |    -239.37 @ 46954 |
|     10000 |              -20 |              -230.05 |               -236.80 |             0 † |    -239.37 @ 46954 |
|     10000 |              -60 |              -230.05 |               -236.79 |             0 † |    -239.37 @ 46954 |
|     15000 |               -1 |              -224.29 |               -227.23 |             0 † |    -235.76 @ 15000 |
|     15000 |              -20 |              -224.28 |               -227.22 |             0 † |    -235.76 @ 15000 |
|     15000 |              -60 |              -224.29 |               -227.23 |             0 † |    -235.76 @ 15000 |
|     18000 |               -1 |              -222.88 |               -226.23 |             0 † |    -233.14 @ 18000 |
|     18000 |              -20 |              -222.88 |               -226.23 |             0 † |    -233.14 @ 18000 |
|     18000 |              -60 |              -222.88 |               -226.23 |             0 † |    -233.14 @ 18000 |
|     20000 |               -1 |              -222.86 |               -226.00 |             0 † |    -233.35 @ 19046 |
|     20000 |              -20 |              -222.85 |               -225.96 |             0 † |    -233.35 @ 19046 |
|     20000 |              -60 |              -222.87 |               -226.00 |             0 † |    -233.35 @ 19046 |
### 96000 -> 44100, decimate=false, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -228.04 |               -230.70 |             0 † |       -233.11 @ 20 |
|        20 |              -20 |              -228.04 |               -230.70 |             0 † |       -233.11 @ 20 |
|        20 |              -60 |              -228.04 |               -230.70 |             0 † |       -233.11 @ 20 |
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
|       100 |               -1 |              -224.35 |               -224.35 |             0 † |      -229.50 @ 100 |
|       100 |              -20 |              -224.35 |               -224.35 |             0 † |      -229.50 @ 100 |
|       100 |              -60 |              -224.35 |               -224.35 |             0 † |      -229.50 @ 100 |
|      1000 |               -1 |              -219.27 |               -219.27 |             0 † |     -224.45 @ 1000 |
|      1000 |              -20 |              -219.27 |               -219.27 |             0 † |     -224.45 @ 1000 |
|      1000 |              -60 |              -219.27 |               -219.27 |             0 † |     -224.45 @ 1000 |
|      5000 |               -1 |              -219.91 |               -219.92 |             0 † |     -225.19 @ 5000 |
|      5000 |              -20 |              -219.91 |               -219.92 |             0 † |     -225.19 @ 5000 |
|      5000 |              -60 |              -219.91 |               -219.92 |             0 † |     -225.19 @ 5000 |
|     10000 |               -1 |              -220.40 |               -220.47 |             0 † |    -226.06 @ 10000 |
|     10000 |              -20 |              -220.39 |               -220.47 |             0 † |    -226.06 @ 10000 |
|     10000 |              -60 |              -220.40 |               -220.47 |             0 † |    -226.06 @ 10000 |
|     15000 |               -1 |              -224.81 |               -225.52 |             0 † |    -237.03 @ 14284 |
|     15000 |              -20 |              -224.81 |               -225.52 |             0 † |    -237.03 @ 14284 |
|     15000 |              -60 |              -224.81 |               -225.52 |             0 † |    -237.03 @ 14284 |
|     18000 |               -1 |              -223.76 |               -225.00 |             0 † |    -234.58 @ 11658 |
|     18000 |              -20 |              -223.77 |               -225.00 |             0 † |    -234.58 @ 11658 |
|     18000 |              -60 |              -223.76 |               -225.00 |             0 † |    -234.58 @ 11658 |
|     20000 |               -1 |              -220.06 |               -221.71 |             0 † |    -227.42 @ 20000 |
|     20000 |              -20 |              -220.06 |               -221.71 |             0 † |    -227.42 @ 20000 |
|     20000 |              -60 |              -220.06 |               -221.71 |             0 † |    -227.42 @ 20000 |
### 192000 -> 48000, decimate=true, high_precision=off

| Freq (Hz) | Amplitude (dBFS) | Broadband THD+N (dB) | Audio-band THD+N (dB) | Gain error (dB) | Max spur (dB @ Hz) |
|-----------|------------------|----------------------|-----------------------|-----------------|--------------------|
|        20 |               -1 |              -222.31 |               -224.98 |        -1.22e-9 |       -227.38 @ 20 |
|        20 |              -20 |              -222.31 |               -224.98 |        -1.22e-9 |       -227.38 @ 20 |
|        20 |              -60 |              -222.31 |               -224.98 |        -1.22e-9 |       -227.38 @ 20 |
|       100 |               -1 |              -224.35 |               -224.35 |        -2.89e-8 |      -229.50 @ 100 |
|       100 |              -20 |              -224.35 |               -224.35 |        -2.89e-8 |      -229.50 @ 100 |
|       100 |              -60 |              -224.35 |               -224.35 |        -2.89e-8 |      -229.50 @ 100 |
|      1000 |               -1 |              -219.27 |               -219.27 |        -1.25e-7 |     -224.45 @ 1000 |
|      1000 |              -20 |              -219.27 |               -219.27 |        -1.25e-7 |     -224.45 @ 1000 |
|      1000 |              -60 |              -219.27 |               -219.27 |        -1.25e-7 |     -224.45 @ 1000 |
|      5000 |               -1 |              -219.91 |               -219.92 |        -1.87e-7 |     -225.19 @ 5000 |
|      5000 |              -20 |              -219.91 |               -219.92 |        -1.87e-7 |     -225.19 @ 5000 |
|      5000 |              -60 |              -219.91 |               -219.92 |        -1.87e-7 |     -225.19 @ 5000 |
|     10000 |               -1 |              -220.40 |               -220.47 |        -1.23e-8 |    -226.06 @ 10000 |
|     10000 |              -20 |              -220.39 |               -220.46 |        -1.23e-8 |    -226.06 @ 10000 |
|     10000 |              -60 |              -220.39 |               -220.46 |        -1.23e-8 |    -226.06 @ 10000 |
|     15000 |               -1 |              -224.81 |               -225.52 |        -3.30e-8 |    -237.03 @ 14284 |
|     15000 |              -20 |              -224.80 |               -225.50 |        -3.30e-8 |    -237.03 @ 14284 |
|     15000 |              -60 |              -224.81 |               -225.52 |        -3.30e-8 |    -237.03 @ 14284 |
|     18000 |               -1 |              -223.77 |               -225.00 |        -6.05e-8 |    -234.58 @ 11658 |
|     18000 |              -20 |              -223.74 |               -224.97 |        -6.05e-8 |    -234.58 @ 11658 |
|     18000 |              -60 |              -223.75 |               -224.99 |        -6.05e-8 |    -234.58 @ 11658 |
|     20000 |               -1 |              -220.06 |               -221.71 |        -4.62e-7 |    -227.42 @ 20000 |
|     20000 |              -20 |              -220.06 |               -221.71 |        -4.62e-7 |    -227.42 @ 20000 |
|     20000 |              -60 |              -220.06 |               -221.71 |        -4.62e-7 |    -227.42 @ 20000 |

## Pre-ringing

`f64` only, no high-precision backend. Every measurement is repeated with the stimulus at 4 positions within the resampler's processing chunk; tables show the worst position.

### Impulse pre-ringing

A unit impulse is resampled, and the output is scanned backward from the impulse's ideal output time. Each "Pre-ring" column is how long before that time the response first reaches the given level, relative to the response's peak (worst position per column). "Ringing freq" is the strongest frequency in the pre-ringing tail, excluding the main lobe. The transition band runs from the passband edge (unity gain below it) to the lower Nyquist frequency.

| Rate pair       | Decimate | Transition band (Hz) | Pre-ring ≥ -60 dB (ms) | Pre-ring ≥ -100 dB (ms) | Pre-ring ≥ -140 dB (ms) | Ringing freq (Hz) |
|-----------------|----------|----------------------|------------------------|-------------------------|-------------------------|-------------------|
| 44100 -> 48000  | false    |        21499 - 22050 |                   3.52 |                    9.12 |                   12.29 |             21778 |
| 48000 -> 44100  | false    |        21499 - 22050 |                   3.20 |                    8.75 |                   12.27 |             21777 |
| 44100 -> 96000  | false    |        21499 - 22050 |                   3.61 |                    9.12 |                   12.29 |             21778 |
| 96000 -> 44100  | false    |        21499 - 22050 |                   3.20 |                    8.75 |                   12.27 |             21777 |
| 192000 -> 48000 | false    |        23400 - 24000 |                   2.92 |                    8.02 |                   11.23 |             23701 |
| 192000 -> 48000 | true     |        23400 - 24000 |                   2.92 |                    8.02 |                   11.23 |             23701 |

### Pre-ringing below the rolloff

Gaussian transients whose spectra are at least -300 dB down at the passband edge (a click, the part of an impulse below the rolloff, plus tone bursts) are resampled and compared sample-by-sample against their exact ideal output. The passband gain is unity, so an ideal linear-phase resampler would return them unchanged: any residual before the transient is pre-ringing that reached frequencies below the rolloff. A best-fit constant delay is removed first, so a timing offset isn't counted as ringing. Echo columns are the largest residual in dB relative to the transient's peak, @ ms before (pre) or after (post) its center; a lead well beyond the envelope σ is a true pre-echo, ahead of the transient's own rise. Values near the THD+N section's noise floor are `f64` rounding, not leakage.

Worst below-rolloff pre-echo: **-152.8 dB**, 1.398 ms ahead of the 19492 Hz burst (192000 -> 48000, decimate=true).

| Rate pair       | Decimate | Transient      | Envelope σ (ms) | Worst pre-echo (dB @ ms) | Worst post-echo (dB @ ms) |
|-----------------|----------|----------------|-----------------|--------------------------|---------------------------|
| 44100 -> 48000  | false    | Click          |           0.063 |           -271.4 @ 0.071 |            -271.5 @ 0.054 |
| 44100 -> 48000  | false    | 10534 Hz burst |           0.126 |           -255.4 @ 0.029 |            -256.5 @ 0.075 |
| 44100 -> 48000  | false    | 17908 Hz burst |           0.419 |           -250.1 @ 0.071 |            -250.1 @ 0.012 |
| 48000 -> 44100  | false    | Click          |           0.063 |           -270.7 @ 0.053 |            -270.6 @ 0.060 |
| 48000 -> 44100  | false    | 10534 Hz burst |           0.126 |           -255.1 @ 0.030 |            -255.4 @ 0.015 |
| 48000 -> 44100  | false    | 17908 Hz burst |           0.419 |           -249.5 @ 0.098 |            -249.3 @ 0.015 |
| 44100 -> 96000  | false    | Click          |           0.063 |           -277.4 @ 0.071 |            -277.4 @ 0.065 |
| 44100 -> 96000  | false    | 10534 Hz burst |           0.126 |           -261.1 @ 0.019 |            -260.7 @ 0.023 |
| 44100 -> 96000  | false    | 17908 Hz burst |           0.419 |           -256.2 @ 0.071 |            -256.1 @ 0.012 |
| 96000 -> 44100  | false    | Click          |           0.063 |           -270.7 @ 0.072 |            -270.5 @ 0.064 |
| 96000 -> 44100  | false    | 10534 Hz burst |           0.126 |           -254.2 @ 0.027 |            -254.3 @ 0.019 |
| 96000 -> 44100  | false    | 17908 Hz burst |           0.419 |           -249.6 @ 0.072 |            -249.3 @ 0.041 |
| 192000 -> 48000 | false    | Click          |           0.058 |           -270.6 @ 0.064 |            -270.6 @ 0.061 |
| 192000 -> 48000 | false    | 11466 Hz burst |           0.115 |           -254.0 @ 0.023 |            -254.1 @ 0.019 |
| 192000 -> 48000 | false    | 19492 Hz burst |           0.385 |           -249.3 @ 0.064 |            -249.4 @ 0.040 |
| 192000 -> 48000 | true     | Click          |           0.058 |           -159.7 @ 0.002 |            -160.1 @ 0.019 |
| 192000 -> 48000 | true     | 11466 Hz burst |           0.115 |           -159.8 @ 0.002 |            -160.5 @ 0.040 |
| 192000 -> 48000 | true     | 19492 Hz burst |           0.385 |           -152.8 @ 1.398 |            -152.7 @ 1.269 |
