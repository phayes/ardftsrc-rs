//! Adapts the vendored real FFT engine in [`vendor`](super::vendor) to the real `realfft`
//! crate's `RealToComplex`/`ComplexToReal` traits, for any [`HighPrecisionFloat`] backend.
//!
//! The vendored traits mirror `realfft`'s method for method, and the engine works in `T`
//! throughout, so the adapters only forward calls and map the error type. This lets `CpuCore<T>`
//! hold a high-precision engine in the same trait-object fields as a stock `realfft` one.

use std::sync::Arc;

use realfft::num_complex::Complex;

use super::numeric::HighPrecisionFloat;
use super::vendor::realfft as vendor_rfft;

/// Maps our vendored (but field-for-field identical) `FftError` to the real `realfft` crate's
/// `FftError`.
fn map_error(e: vendor_rfft::FftError) -> realfft::FftError {
    match e {
        vendor_rfft::FftError::InputBuffer(expected, got) => realfft::FftError::InputBuffer(expected, got),
        vendor_rfft::FftError::OutputBuffer(expected, got) => realfft::FftError::OutputBuffer(expected, got),
        vendor_rfft::FftError::ScratchBuffer(expected, got) => realfft::FftError::ScratchBuffer(expected, got),
        vendor_rfft::FftError::InputValues(first, last) => realfft::FftError::InputValues(first, last),
    }
}

/// Forward real-to-complex high-precision FFT for a fixed length.
struct HpRealToComplex<T: HighPrecisionFloat>(Arc<dyn vendor_rfft::RealToComplex<T>>);

impl<T: HighPrecisionFloat> realfft::RealToComplex<T> for HpRealToComplex<T> {
    fn process(&self, input: &mut [T], output: &mut [Complex<T>]) -> Result<(), realfft::FftError> {
        self.0.process(input, output).map_err(map_error)
    }

    fn process_with_scratch(
        &self,
        input: &mut [T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), realfft::FftError> {
        self.0.process_with_scratch(input, output, scratch).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        self.0.get_scratch_len()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn make_input_vec(&self) -> Vec<T> {
        self.0.make_input_vec()
    }

    fn make_output_vec(&self) -> Vec<Complex<T>> {
        self.0.make_output_vec()
    }

    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        self.0.make_scratch_vec()
    }
}

/// Inverse complex-to-real high-precision FFT for a fixed length.
struct HpComplexToReal<T: HighPrecisionFloat>(Arc<dyn vendor_rfft::ComplexToReal<T>>);

impl<T: HighPrecisionFloat> realfft::ComplexToReal<T> for HpComplexToReal<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [T]) -> Result<(), realfft::FftError> {
        self.0.process(input, output).map_err(map_error)
    }

    fn process_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), realfft::FftError> {
        self.0.process_with_scratch(input, output, scratch).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        self.0.get_scratch_len()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn make_input_vec(&self) -> Vec<Complex<T>> {
        self.0.make_input_vec()
    }

    fn make_output_vec(&self) -> Vec<T> {
        self.0.make_output_vec()
    }

    fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        self.0.make_scratch_vec()
    }
}

/// Plans a forward real FFT of `len` samples computed entirely in `T`.
pub(crate) fn plan_fft_forward<T: HighPrecisionFloat>(len: usize) -> Arc<dyn realfft::RealToComplex<T>> {
    let inner = vendor_rfft::RealFftPlanner::<T>::new().plan_fft_forward(len);
    Arc::new(HpRealToComplex(inner))
}

/// Inverse counterpart of [`plan_fft_forward`].
pub(crate) fn plan_fft_inverse<T: HighPrecisionFloat>(len: usize) -> Arc<dyn realfft::ComplexToReal<T>> {
    let inner = vendor_rfft::RealFftPlanner::<T>::new().plan_fft_inverse(len);
    Arc::new(HpComplexToReal(inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::high_precision::numeric::{DoubleDouble, F128, F256, twiddle_factor};
    use crate::high_precision::vendor::rustfft::FftDirection;

    fn widen<T: HighPrecisionFloat>(input: &[f64]) -> Vec<T> {
        input.iter().map(|&value| T::from_f64_exact(value)).collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Forward-then-inverse should return (a scaled copy of) the original signal, for both
    /// power-of-two and non-power-of-two lengths -- confirms the vendored mixed-radix/Bluestein
    /// selection logic does the right thing with twiddles backed by `T`.
    fn assert_round_trip_recovers_signal<T: HighPrecisionFloat>() {
        for len in [1usize, 2, 3, 4, 5, 7, 8, 12, 17, 32, 100, 101, 257, 1024, 4000] {
            let r2c = plan_fft_forward::<T>(len);
            let c2r = plan_fft_inverse::<T>(len);

            let original: Vec<f64> = (0..len).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();

            let mut input = widen::<T>(&original);
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            let mut hp_recovered = c2r.make_output_vec();
            c2r.process(&mut spectrum, &mut hp_recovered).unwrap();
            let recovered: Vec<f64> = hp_recovered.iter().map(|value| value.to_f64() / len as f64).collect();

            let diff = max_abs_diff(&original, &recovered);
            assert!(diff < 1e-12, "len={len}: round-trip diff {diff} too large");
        }
    }

    /// Forward-then-inverse round trips hide the precision benefit: forward and inverse rounding
    /// errors substantially cancel, and the result is narrowed back to `f64` here regardless of
    /// engine.
    ///
    /// The forward *spectrum*, before any such cancellation, is where the benefit shows. This
    /// compares both engines' forward spectra at a few bins against a brute-force O(N) direct DFT
    /// computed independently in `T` arithmetic (not via any FFT butterfly structure) -- for `len`
    /// this small, that reference is accurate to within a handful of `T` ULPs, many orders of
    /// magnitude tighter than either FFT engine's `f64`-representable output.
    fn assert_forward_spectrum_is_more_accurate_than_f64<T: HighPrecisionFloat>() {
        let len = 512usize;
        let original: Vec<f64> = (0..len)
            .map(|i| (i as f64 * 0.083).sin() * 0.9 + (i as f64 * 1.7).cos() * 0.1)
            .collect();

        let r2c = plan_fft_forward::<T>(len);
        let mut input = widen::<T>(&original);
        let mut wide_spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut wide_spectrum).unwrap();
        let hp_spectrum: Vec<Complex<f64>> = wide_spectrum
            .iter()
            .map(|bin| Complex::new(bin.re.to_f64(), bin.im.to_f64()))
            .collect();

        let mut f64_planner = realfft::RealFftPlanner::<f64>::new();
        let f64_r2c = f64_planner.plan_fft_forward(len);
        let mut f64_input = original.clone();
        let mut f64_spectrum = f64_r2c.make_output_vec();
        f64_r2c.process(&mut f64_input, &mut f64_spectrum).unwrap();

        let mut total_hp_error = 0.0f64;
        let mut total_f64_error = 0.0f64;
        for &k in &[0usize, 1, len / 4, len / 2 - 1, len / 2] {
            let reference = direct_dft_bin::<T>(&original, k, len);
            total_hp_error += (hp_spectrum[k] - reference).norm();
            total_f64_error += (f64_spectrum[k] - reference).norm();
        }

        assert!(
            total_hp_error <= total_f64_error,
            "expected the high-precision spectrum (total error {total_hp_error}) to be at least as \
             accurate as plain f64's (total error {total_f64_error})"
        );
        assert!(
            total_f64_error > 0.0,
            "test is vacuous if the f64 engine has no error to beat"
        );
    }

    /// Computes one DFT output bin directly (`O(len)`) in `T` arithmetic, independent of any FFT
    /// algorithm.
    fn direct_dft_bin<T: HighPrecisionFloat>(signal: &[f64], bin: usize, len: usize) -> Complex<f64> {
        let mut sum = Complex::new(T::zero(), T::zero());
        for (n, &sample) in signal.iter().enumerate() {
            let twiddle = twiddle_factor::<T>((bin * n) % len, len, FftDirection::Forward);
            let sample = T::from_f64_exact(sample);
            sum.re = sum.re + sample * twiddle.re;
            sum.im = sum.im + sample * twiddle.im;
        }
        Complex::new(sum.re.to_f64(), sum.im.to_f64())
    }

    #[test]
    fn double_double_round_trip_recovers_signal() {
        assert_round_trip_recovers_signal::<DoubleDouble>();
    }

    #[test]
    fn f128_round_trip_recovers_signal() {
        assert_round_trip_recovers_signal::<F128>();
    }

    #[test]
    fn f256_round_trip_recovers_signal() {
        assert_round_trip_recovers_signal::<F256>();
    }

    #[test]
    fn double_double_forward_spectrum_is_more_accurate_than_f64() {
        assert_forward_spectrum_is_more_accurate_than_f64::<DoubleDouble>();
    }

    #[test]
    fn f128_forward_spectrum_is_more_accurate_than_f64() {
        assert_forward_spectrum_is_more_accurate_than_f64::<F128>();
    }

    #[test]
    fn f256_forward_spectrum_is_more_accurate_than_f64() {
        assert_forward_spectrum_is_more_accurate_than_f64::<F256>();
    }
}
