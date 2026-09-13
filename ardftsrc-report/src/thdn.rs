//! THD+N (total harmonic distortion + noise) analysis for steady-state single-tone
//! `f64` signals.
//!
//! [`sine_fit`] fits a steady-state single-tone buffer against
//! `A*sin + B*cos + C` at a known frequency (time-domain, no FFT, no spectral
//! leakage) and reports THD+N (DC-Nyquist), gain, phase, and DC offset. The
//! fit residual `e[n]` it returns can be fed to [`residual_spectrum`] to get a
//! bandwidth-limited THD+N number and the largest discrete spur.

use realfft::RealFftPlanner;

pub mod report;

/// Below this, a measurement is reporting the analyzer's own numerical floor
/// rather than anything about the signal under test.
pub const NOISE_FLOOR_DB: f64 = -300.0;

fn db_ratio(numerator_rms: f64, denominator_rms: f64) -> f64 {
    if numerator_rms <= 0.0 || denominator_rms <= 0.0 {
        return NOISE_FLOOR_DB;
    }
    (20.0 * (numerator_rms / denominator_rms).log10()).max(NOISE_FLOOR_DB)
}

/// Result of fitting `samples` against `A*sin(2*pi*f*n/fs) + B*cos(2*pi*f*n/fs) + C`.
#[derive(Debug, Clone, PartialEq)]
pub struct SineFitResult {
    /// `20*log10(RMS(residual) / RMS(fundamental))`, i.e. THD+N over the full
    /// DC-Nyquist bandwidth. Clamped at [`NOISE_FLOOR_DB`].
    pub thdn_db: f64,
    /// `sqrt(A^2 + B^2)`: the fitted amplitude at the stimulus frequency.
    /// Divide by the known input amplitude to get gain error.
    pub gain: f64,
    /// `atan2(B, A)`: the fitted phase at the stimulus frequency, in radians.
    pub phase_rad: f64,
    /// `C`: the fitted DC offset.
    pub dc_offset: f64,
    /// RMS of the fitted fundamental (`A*sin + B*cos`, excluding DC).
    pub fundamental_rms: f64,
    /// RMS of the fit residual `e[n] = y[n] - (A*sin + B*cos + C)`.
    pub residual_rms: f64,
    /// The fit residual `e[n]`, in sample order. Feed this to
    /// [`residual_spectrum`] for band-limited THD+N and max-spur analysis.
    pub residual: Vec<f64>,
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Fits `samples` (assumed steady-state, single-tone at `freq`) against
/// `A*sin + B*cos + C` at `freq`/`sample_rate`.
///
/// The `C` (DC) term is fit rather than removed by pre-subtracting the
/// sample mean, so the result doesn't depend on `samples` holding a whole
/// number of cycles.
///
/// Panics if `samples` has fewer than 3 elements, or if the normal-equations
/// matrix is singular (degenerate input, e.g. all-zero `samples`).
pub fn sine_fit(samples: &[f64], sample_rate: f64, freq: f64) -> SineFitResult {
    assert!(
        samples.len() >= 3,
        "sine_fit: need at least 3 samples, got {}",
        samples.len()
    );

    let n = samples.len() as f64;
    let w = 2.0 * std::f64::consts::PI * freq / sample_rate;

    // Direct per-index sin/cos evaluation (not a recurrence/phase accumulator):
    // a recurrence accumulates rounding error per step, and the whole point here
    // is measuring residuals far below the direct-sin/cos rounding floor.
    let (mut ss, mut cc, mut sc, mut sd, mut cd, mut sy, mut cy, mut y_sum) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (i, &y) in samples.iter().enumerate() {
        let (s, c) = (w * i as f64).sin_cos();
        ss += s * s;
        cc += c * c;
        sc += s * c;
        sd += s;
        cd += c;
        sy += y * s;
        cy += y * c;
        y_sum += y;
    }

    // Solve the symmetric 3x3 normal-equations system for [A, B, C] by Cramer's rule:
    //   [ss sc sd] [A]   [sy]
    //   [sc cc cd] [B] = [cy]
    //   [sd cd n ] [C]   [y_sum]
    let m = [[ss, sc, sd], [sc, cc, cd], [sd, cd, n]];
    let rhs = [sy, cy, y_sum];
    let det = det3(m);
    assert!(
        det.abs() > 0.0,
        "sine_fit: singular normal-equations matrix (degenerate input?)"
    );

    let solve_col = |col: usize| -> f64 {
        let mut mm = m;
        for (row, r) in rhs.iter().enumerate() {
            mm[row][col] = *r;
        }
        det3(mm) / det
    };

    let a = solve_col(0);
    let b = solve_col(1);
    let c = solve_col(2);

    let mut residual = Vec::with_capacity(samples.len());
    let mut fundamental_sq_sum = 0.0;
    let mut residual_sq_sum = 0.0;
    for (i, &y) in samples.iter().enumerate() {
        let (s, cph) = (w * i as f64).sin_cos();
        let fundamental = a * s + b * cph;
        let e = y - (fundamental + c);
        fundamental_sq_sum += fundamental * fundamental;
        residual_sq_sum += e * e;
        residual.push(e);
    }

    let fundamental_rms = (fundamental_sq_sum / n).sqrt();
    let residual_rms = (residual_sq_sum / n).sqrt();

    SineFitResult {
        thdn_db: db_ratio(residual_rms, fundamental_rms),
        gain: (a * a + b * b).sqrt(),
        phase_rad: b.atan2(a),
        dc_offset: c,
        fundamental_rms,
        residual_rms,
        residual,
    }
}

/// The largest discrete spur found in a residual spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpurResult {
    /// Frequency of the largest spur bin, in Hz.
    pub freq_hz: f64,
    /// Level of that spur relative to the fundamental, in dB. Clamped at
    /// [`NOISE_FLOOR_DB`].
    pub level_db: f64,
}

/// Result of an FFT analysis of a [`SineFitResult::residual`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualSpectrumResult {
    /// THD+N limited to the given band (e.g. 20 Hz-20 kHz), relative to the
    /// fundamental. Clamped at [`NOISE_FLOOR_DB`].
    pub band_thdn_db: f64,
    /// The single largest discrete spur in the residual, anywhere from the
    /// first non-DC bin to Nyquist.
    pub max_spur: SpurResult,
}

/// FFTs `residual` (typically [`SineFitResult::residual`]) and reports
/// band-limited THD+N plus the largest discrete spur, both relative to
/// `fundamental_rms` (typically [`SineFitResult::fundamental_rms`]).
///
/// `band_hz` is `(low, high)` in Hz, inclusive. The DC bin is always excluded
/// from both the band sum and the spur search since it reflects fit/numerical
/// bias rather than a spectral feature.
///
/// No window is applied: this sums exact power-spectrum contributions (via
/// Parseval's theorem) rather than trading leakage rejection for amplitude
/// accuracy, which matters when values this far below the fundamental are
/// being compared.
///
/// Panics if `residual` has fewer than 2 samples.
pub fn residual_spectrum(
    residual: &[f64],
    fundamental_rms: f64,
    sample_rate: f64,
    band_hz: (f64, f64),
) -> ResidualSpectrumResult {
    let n = residual.len();
    assert!(n >= 2, "residual_spectrum: need at least 2 samples, got {n}");

    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut input = residual.to_vec();
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum)
        .expect("fixed-size real FFT process should not fail");

    let bin_hz = sample_rate / n as f64;
    let nyquist_bin = n / 2;
    let n_sq = (n as f64) * (n as f64);

    let mut band_power = 0.0;
    let mut max_power = 0.0;
    let mut max_bin = 0usize;

    // Bin 0 (DC) is skipped: sine_fit already fits DC separately, so any energy
    // left there is fit/numerical bias, not a spectral feature of the residual.
    for (k, bin) in spectrum.iter().enumerate().skip(1) {
        // One-sided power spectrum via Parseval: sum_n x[n]^2 == (1/N^2) * (|X[0]|^2 +
        // |X[N/2]|^2 + 2*sum_{0<k<N/2} |X[k]|^2) for real x of even length N (odd N has
        // no Nyquist bin to leave unscaled).
        let scale = if k == nyquist_bin && n.is_multiple_of(2) {
            1.0
        } else {
            2.0
        };
        let power = bin.norm_sqr() * scale / n_sq;

        let freq = k as f64 * bin_hz;
        if freq >= band_hz.0 && freq <= band_hz.1 {
            band_power += power;
        }
        if power > max_power {
            max_power = power;
            max_bin = k;
        }
    }

    ResidualSpectrumResult {
        band_thdn_db: db_ratio(band_power.sqrt(), fundamental_rms),
        max_spur: SpurResult {
            freq_hz: max_bin as f64 * bin_hz,
            level_db: db_ratio(max_power.sqrt(), fundamental_rms),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn synth_sine(n: usize, sample_rate: f64, freq: f64, amplitude: f64, phase: f64, dc: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f64 / sample_rate + phase).sin() + dc)
            .collect()
    }

    #[test]
    fn recovers_gain_phase_and_dc_of_a_clean_sine() {
        let sample_rate = 48_000.0;
        let freq = 1_000.0;
        let amplitude = 0.7;
        let phase = 0.9;
        let dc = 0.01;
        let samples = synth_sine(48_000, sample_rate, freq, amplitude, phase, dc);

        let fit = sine_fit(&samples, sample_rate, freq);

        assert!((fit.gain - amplitude).abs() < 1e-9, "gain={}", fit.gain);
        assert!((fit.dc_offset - dc).abs() < 1e-9, "dc_offset={}", fit.dc_offset);

        // atan2(B, A) with A = R*cos(phi'), B = R*sin(phi') recovers phi' = phase + pi/2
        // for a sin(x + phase) model reparameterized as A*sin(x) + B*cos(x); just check
        // it round-trips to the same waveform rather than pin an exact convention.
        let reconstructed: Vec<f64> = (0..samples.len())
            .map(|i| {
                let w = 2.0 * PI * freq / sample_rate;
                let x = w * i as f64;
                fit.gain * (x + fit.phase_rad).sin() + fit.dc_offset
            })
            .collect();
        let max_err = samples
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-9, "max_err={max_err}");

        // A clean, exactly-modeled sine should bottom out near the f64 summation floor
        // (empirically ~-248 dB for a 48000-sample fit; -200 dB leaves headroom for
        // platform-dependent summation order).
        assert!(fit.thdn_db <= -200.0, "thdn_db={}", fit.thdn_db);
    }

    #[test]
    fn detects_injected_harmonic_distortion() {
        let sample_rate = 48_000.0;
        let freq = 1_000.0;
        let fundamental_amp = 1.0;
        let second_harmonic_amp = fundamental_amp * 0.01; // -40 dB THD

        let n = 48_000;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                fundamental_amp * (2.0 * PI * freq * t).sin() + second_harmonic_amp * (2.0 * PI * 2.0 * freq * t).sin()
            })
            .collect();

        let fit = sine_fit(&samples, sample_rate, freq);

        // Expected THD+N ~= 20*log10(0.01/sqrt(2) / (1/sqrt(2))) = 20*log10(0.01) = -40 dB.
        assert!((fit.thdn_db - (-40.0)).abs() < 0.5, "thdn_db={}", fit.thdn_db);
        assert!((fit.gain - fundamental_amp).abs() < 1e-6, "gain={}", fit.gain);
    }

    #[test]
    fn detects_injected_broadband_noise() {
        let sample_rate = 48_000.0;
        let freq = 1_000.0;
        let amplitude = 1.0;
        let fundamental_rms = amplitude / 2.0_f64.sqrt();
        let target_ratio = 1e-3; // -60 dB target, referenced to the fundamental (not `amplitude`)
        // Scaled by sqrt(3) so a uniform-on-[-1,1) variate (population RMS = 1/sqrt(3))
        // contributes an RMS of exactly `fundamental_rms * target_ratio`.
        let noise_amplitude = fundamental_rms * target_ratio * 3.0_f64.sqrt();

        let n = 96_000;
        // Deterministic pseudo-noise (LCG) so the test has no external RNG dependency.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 11) & ((1u64 << 53) - 1); // top 53 bits
            2.0 * (bits as f64 / (1u64 << 53) as f64) - 1.0 // uniform on [-1, 1)
        };

        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * PI * freq * t).sin() + noise_amplitude * next()
            })
            .collect();

        let fit = sine_fit(&samples, sample_rate, freq);
        assert!((fit.thdn_db - (-60.0)).abs() < 1.0, "thdn_db={}", fit.thdn_db);
    }

    #[test]
    fn works_without_a_whole_number_of_cycles() {
        // 1013 Hz over a 48000-sample window at 48 kHz is not an integer number of cycles.
        let sample_rate = 48_000.0;
        let freq = 1_013.0;
        let samples = synth_sine(48_000, sample_rate, freq, 0.5, 0.3, -0.05);
        let fit = sine_fit(&samples, sample_rate, freq);
        assert!((fit.gain - 0.5).abs() < 1e-9, "gain={}", fit.gain);
        assert!((fit.dc_offset - (-0.05)).abs() < 1e-9, "dc_offset={}", fit.dc_offset);
        assert!(fit.thdn_db <= -200.0, "thdn_db={}", fit.thdn_db);
    }

    #[test]
    fn residual_spectrum_locates_an_injected_spur() {
        let sample_rate = 48_000.0;
        let freq = 1_000.0;
        let spur_freq = 6_000.0;
        let amplitude = 1.0;
        let spur_amplitude = amplitude * 1e-2; // -40 dB

        let n = 48_000;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * PI * freq * t).sin() + spur_amplitude * (2.0 * PI * spur_freq * t).sin()
            })
            .collect();

        let fit = sine_fit(&samples, sample_rate, freq);
        let spectrum = residual_spectrum(&fit.residual, fit.fundamental_rms, sample_rate, (20.0, 20_000.0));

        assert!(
            (spectrum.max_spur.freq_hz - spur_freq).abs() <= sample_rate / n as f64,
            "freq_hz={}",
            spectrum.max_spur.freq_hz
        );
        assert!(
            (spectrum.max_spur.level_db - (-40.0)).abs() < 0.5,
            "level_db={}",
            spectrum.max_spur.level_db
        );
        // Full-band and audio-band numbers should agree closely: nothing here is out-of-band.
        assert!((spectrum.band_thdn_db - fit.thdn_db).abs() < 0.5);
    }

    #[test]
    fn band_limiting_excludes_out_of_band_energy() {
        let sample_rate = 96_000.0;
        let freq = 1_000.0;
        let amplitude = 1.0;
        // A large spur well above a 20 kHz audio-band limit.
        let ultrasonic_spur_freq = 30_000.0;
        let spur_amplitude = amplitude * 0.1; // -20 dB, well above the fundamental-referenced floor

        let n = 96_000;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * PI * freq * t).sin() + spur_amplitude * (2.0 * PI * ultrasonic_spur_freq * t).sin()
            })
            .collect();

        let fit = sine_fit(&samples, sample_rate, freq);
        let audio_band = residual_spectrum(&fit.residual, fit.fundamental_rms, sample_rate, (20.0, 20_000.0));
        let full_band = residual_spectrum(
            &fit.residual,
            fit.fundamental_rms,
            sample_rate,
            (20.0, sample_rate / 2.0),
        );

        assert!(
            audio_band.band_thdn_db < full_band.band_thdn_db - 10.0,
            "audio={} full={}",
            audio_band.band_thdn_db,
            full_band.band_thdn_db
        );
        // The spur itself is still found (it's the largest bin) even though it's out of band.
        assert!((full_band.max_spur.freq_hz - ultrasonic_spur_freq).abs() <= sample_rate / n as f64);
    }
}
