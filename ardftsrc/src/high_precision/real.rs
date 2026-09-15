//! `f64`-in, `f64`-out real FFT built on the vendored engine in [`vendor`](super::vendor),
//! instantiated for any [`HighPrecisionFloat`] backend.
//!
//! Everything internal to a transform (twiddle factors, butterfly accumulation) happens in the
//! backend's precision; this module only converts at the boundary, so callers never see the
//! backend type. `f64 -> T` is exact (every backend's mantissa is a superset of `f64`'s), and
//! `T -> f64` rounds to nearest.
//!
//! This intentionally mirrors the shape of `realfft`'s `RealFftPlanner`/`RealToComplex`/
//! `ComplexToReal` (same method names/semantics), so it can implement the real `realfft` traits
//! and slot into `CpuCore`'s existing trait-object fields.

use std::fmt;
use std::sync::Arc;

use realfft::num_complex::Complex;

use super::numeric::HighPrecisionFloat;
use super::vendor::realfft as vendor_rfft;

/// Error returned by a [`HpRealToComplex`] or [`HpComplexToReal`] transform.
#[derive(Debug)]
pub(crate) struct HpFftError(vendor_rfft::FftError);

impl fmt::Display for HpFftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for HpFftError {}

fn widen<T: HighPrecisionFloat>(input: &[f64]) -> Vec<T> {
    input.iter().map(|&v| T::from_f64_exact(v)).collect()
}

fn widen_complex<T: HighPrecisionFloat>(input: &[Complex<f64>]) -> Vec<Complex<T>> {
    input
        .iter()
        .map(|c| Complex::new(T::from_f64_exact(c.re), T::from_f64_exact(c.im)))
        .collect()
}

fn narrow_into<T: HighPrecisionFloat>(dst: &mut [f64], src: &[T]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.to_f64();
    }
}

fn narrow_complex_into<T: HighPrecisionFloat>(dst: &mut [Complex<f64>], src: &[Complex<T>]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = Complex::new(s.re.to_f64(), s.im.to_f64());
    }
}

/// Plans high-precision real FFTs. See [`realfft::RealFftPlanner`] for the API this mirrors.
pub(crate) struct HpRealFftPlanner<T: HighPrecisionFloat> {
    inner: vendor_rfft::RealFftPlanner<T>,
}

impl<T: HighPrecisionFloat> HpRealFftPlanner<T> {
    pub(crate) fn new() -> Self {
        Self {
            inner: vendor_rfft::RealFftPlanner::new(),
        }
    }

    pub(crate) fn plan_fft_forward(&mut self, len: usize) -> Arc<HpRealToComplex<T>> {
        Arc::new(HpRealToComplex {
            inner: self.inner.plan_fft_forward(len),
        })
    }

    pub(crate) fn plan_fft_inverse(&mut self, len: usize) -> Arc<HpComplexToReal<T>> {
        Arc::new(HpComplexToReal {
            inner: self.inner.plan_fft_inverse(len),
        })
    }
}

/// Forward real-to-complex high-precision FFT for a fixed length. See
/// [`realfft::RealToComplex`] for the API this mirrors.
pub(crate) struct HpRealToComplex<T: HighPrecisionFloat> {
    inner: Arc<dyn vendor_rfft::RealToComplex<T>>,
}

impl<T: HighPrecisionFloat> HpRealToComplex<T> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub(crate) fn complex_len(&self) -> usize {
        self.inner.complex_len()
    }

    pub(crate) fn make_input_vec(&self) -> Vec<f64> {
        vec![0.0; self.len()]
    }

    pub(crate) fn make_output_vec(&self) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); self.complex_len()]
    }

    /// Transforms `input` (length [`len()`](Self::len)) into `output` (length
    /// [`complex_len()`](Self::complex_len)), converting to/from `T` at the boundary. `input` is
    /// left in an unspecified state after the call, matching `realfft`.
    pub(crate) fn process(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<(), HpFftError> {
        let mut hp_in = widen::<T>(input);
        let mut hp_out = vec![Complex::new(T::zero(), T::zero()); output.len()];
        self.inner.process(&mut hp_in, &mut hp_out).map_err(HpFftError)?;
        narrow_complex_into(output, &hp_out);
        Ok(())
    }
}

/// Inverse complex-to-real high-precision FFT for a fixed length. See
/// [`realfft::ComplexToReal`] for the API this mirrors.
pub(crate) struct HpComplexToReal<T: HighPrecisionFloat> {
    inner: Arc<dyn vendor_rfft::ComplexToReal<T>>,
}

impl<T: HighPrecisionFloat> HpComplexToReal<T> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub(crate) fn complex_len(&self) -> usize {
        self.inner.complex_len()
    }

    pub(crate) fn make_input_vec(&self) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); self.complex_len()]
    }

    pub(crate) fn make_output_vec(&self) -> Vec<f64> {
        vec![0.0; self.len()]
    }

    /// Transforms `input` (length [`complex_len()`](Self::complex_len)) into `output` (length
    /// [`len()`](Self::len)), converting to/from `T` at the boundary. `input` is left in an
    /// unspecified state after the call, matching `realfft`.
    pub(crate) fn process(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<(), HpFftError> {
        let mut hp_in = widen_complex::<T>(input);
        let mut hp_out = vec![T::zero(); output.len()];
        self.inner.process(&mut hp_in, &mut hp_out).map_err(HpFftError)?;
        narrow_into(output, &hp_out);
        Ok(())
    }
}

/// Maps our vendored (but field-for-field identical) `FftError` to the real `realfft` crate's
/// `FftError`, so the wrappers can implement the real `realfft` traits below.
fn map_error(e: HpFftError) -> realfft::FftError {
    match e.0 {
        vendor_rfft::FftError::InputBuffer(expected, got) => realfft::FftError::InputBuffer(expected, got),
        vendor_rfft::FftError::OutputBuffer(expected, got) => realfft::FftError::OutputBuffer(expected, got),
        vendor_rfft::FftError::ScratchBuffer(expected, got) => realfft::FftError::ScratchBuffer(expected, got),
        vendor_rfft::FftError::InputValues(first, last) => realfft::FftError::InputValues(first, last),
    }
}

// `get_scratch_len` returns 0 and `process_with_scratch` ignores the caller-provided scratch:
// the engine's own working buffers are `Vec<Complex<T>>`, not `Vec<Complex<f64>>`, so an
// `f64`-shaped scratch buffer from a caller can't be reused here. This only gives up the
// scratch-reuse optimization `process_with_scratch` normally provides, not correctness.

impl<T: HighPrecisionFloat> realfft::RealToComplex<f64> for HpRealToComplex<T> {
    fn process(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<(), realfft::FftError> {
        HpRealToComplex::process(self, input, output).map_err(map_error)
    }

    fn process_with_scratch(
        &self,
        input: &mut [f64],
        output: &mut [Complex<f64>],
        _scratch: &mut [Complex<f64>],
    ) -> Result<(), realfft::FftError> {
        HpRealToComplex::process(self, input, output).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        HpRealToComplex::len(self)
    }

    fn make_input_vec(&self) -> Vec<f64> {
        HpRealToComplex::make_input_vec(self)
    }

    fn make_output_vec(&self) -> Vec<Complex<f64>> {
        HpRealToComplex::make_output_vec(self)
    }

    fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
        Vec::new()
    }
}

impl<T: HighPrecisionFloat> realfft::ComplexToReal<f64> for HpComplexToReal<T> {
    fn process(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<(), realfft::FftError> {
        HpComplexToReal::process(self, input, output).map_err(map_error)
    }

    fn process_with_scratch(
        &self,
        input: &mut [Complex<f64>],
        output: &mut [f64],
        _scratch: &mut [Complex<f64>],
    ) -> Result<(), realfft::FftError> {
        HpComplexToReal::process(self, input, output).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        HpComplexToReal::len(self)
    }

    fn make_input_vec(&self) -> Vec<Complex<f64>> {
        HpComplexToReal::make_input_vec(self)
    }

    fn make_output_vec(&self) -> Vec<f64> {
        HpComplexToReal::make_output_vec(self)
    }

    fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
        Vec::new()
    }
}

/// Plans a forward real FFT in `T` precision, coerced to the real `realfft` crate's
/// `RealToComplex<f64>` trait object.
pub(crate) fn plan_fft_forward<T: HighPrecisionFloat>(len: usize) -> Arc<dyn realfft::RealToComplex<f64>> {
    HpRealFftPlanner::<T>::new().plan_fft_forward(len) as Arc<dyn realfft::RealToComplex<f64>>
}

/// Inverse counterpart of [`plan_fft_forward`].
pub(crate) fn plan_fft_inverse<T: HighPrecisionFloat>(len: usize) -> Arc<dyn realfft::ComplexToReal<f64>> {
    HpRealFftPlanner::<T>::new().plan_fft_inverse(len) as Arc<dyn realfft::ComplexToReal<f64>>
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::high_precision::numeric::{Dd, F128, F256, twiddle_factor};
    use crate::high_precision::vendor::rustfft::FftDirection;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Forward-then-inverse should return (a scaled copy of) the original signal, for both
    /// power-of-two and non-power-of-two lengths -- confirms the vendored mixed-radix/Bluestein
    /// selection logic does the right thing with twiddles backed by `T`.
    fn assert_round_trip_recovers_signal<T: HighPrecisionFloat>() {
        for len in [1usize, 2, 3, 4, 5, 7, 8, 12, 17, 32, 100, 101, 257, 1024, 4000] {
            let mut planner = HpRealFftPlanner::<T>::new();
            let r2c = planner.plan_fft_forward(len);
            let c2r = planner.plan_fft_inverse(len);

            let original: Vec<f64> = (0..len).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();

            let mut input = original.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            let mut recovered = c2r.make_output_vec();
            c2r.process(&mut spectrum, &mut recovered).unwrap();
            for value in recovered.iter_mut() {
                *value /= len as f64;
            }

            let diff = max_abs_diff(&original, &recovered);
            assert!(diff < 1e-12, "len={len}: round-trip diff {diff} too large");
        }
    }

    /// Forward-then-inverse round trips hide the precision benefit: forward and inverse rounding
    /// errors substantially cancel, and the result is narrowed back to `f64` regardless of engine.
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

        let mut planner = HpRealFftPlanner::<T>::new();
        let r2c = planner.plan_fft_forward(len);
        let mut input = original.clone();
        let mut hp_spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut hp_spectrum).unwrap();

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
        assert_round_trip_recovers_signal::<Dd>();
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
        assert_forward_spectrum_is_more_accurate_than_f64::<Dd>();
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
