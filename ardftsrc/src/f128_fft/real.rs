//! `f64`-in, `f64`-out real FFT built on the `f128` engine in [`vendor`](super::vendor).
//!
//! Everything internal to a transform (twiddle factors, butterfly accumulation) happens in
//! [`F128`](super::numeric::F128) precision; this module only exists to convert at the boundary,
//! so callers never need to know `F128` exists. `f64 -> F128` is exact (`f128`'s mantissa is a
//! strict superset of `f64`'s), and `F128 -> f64` narrows via a plain cast.
//!
//! This intentionally mirrors the shape of `realfft`'s `RealFftPlanner`/`RealToComplex`/
//! `ComplexToReal` (same method names/semantics), so a caller already using `realfft` can swap
//! backends without relearning an API.

use std::fmt;
use std::sync::Arc;

use realfft::num_complex::Complex;

use super::numeric::F128;
use super::vendor::realfft as vendor_rfft;

/// Error returned by a [`F128RealToComplex`] or [`F128ComplexToReal`] transform.
#[derive(Debug)]
pub(crate) struct F128FftError(vendor_rfft::FftError);

impl fmt::Display for F128FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for F128FftError {}

fn to_f128(input: &[f64]) -> Vec<F128> {
    input.iter().map(|&v| F128::from_f64(v)).collect()
}

fn to_f128_complex(input: &[Complex<f64>]) -> Vec<Complex<F128>> {
    input
        .iter()
        .map(|c| Complex::new(F128::from_f64(c.re), F128::from_f64(c.im)))
        .collect()
}

fn write_from_f128(dst: &mut [f64], src: &[F128]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.to_f64();
    }
}

fn write_from_f128_complex(dst: &mut [Complex<f64>], src: &[Complex<F128>]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = Complex::new(s.re.to_f64(), s.im.to_f64());
    }
}

/// Plans `f128`-precision real FFTs. See [`realfft::RealFftPlanner`] for the API this mirrors.
pub(crate) struct F128RealFftPlanner {
    inner: vendor_rfft::RealFftPlanner<F128>,
}

impl F128RealFftPlanner {
    pub(crate) fn new() -> Self {
        Self {
            inner: vendor_rfft::RealFftPlanner::new(),
        }
    }

    pub(crate) fn plan_fft_forward(&mut self, len: usize) -> Arc<F128RealToComplex> {
        Arc::new(F128RealToComplex {
            inner: self.inner.plan_fft_forward(len),
        })
    }

    pub(crate) fn plan_fft_inverse(&mut self, len: usize) -> Arc<F128ComplexToReal> {
        Arc::new(F128ComplexToReal {
            inner: self.inner.plan_fft_inverse(len),
        })
    }
}

/// Forward real-to-complex `f128`-precision FFT for a fixed length. See
/// [`realfft::RealToComplex`] for the API this mirrors.
pub(crate) struct F128RealToComplex {
    inner: Arc<dyn vendor_rfft::RealToComplex<F128>>,
}

impl F128RealToComplex {
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
    /// [`complex_len()`](Self::complex_len)), converting to/from `f128` precision at the
    /// boundary. `input` is left in an unspecified state after the call, matching `realfft`.
    pub(crate) fn process(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<(), F128FftError> {
        let mut f128_in = to_f128(input);
        let mut f128_out = vec![Complex::new(F128::from_f64(0.0), F128::from_f64(0.0)); output.len()];
        self.inner.process(&mut f128_in, &mut f128_out).map_err(F128FftError)?;
        write_from_f128_complex(output, &f128_out);
        Ok(())
    }
}

/// Inverse complex-to-real `f128`-precision FFT for a fixed length. See
/// [`realfft::ComplexToReal`] for the API this mirrors.
pub(crate) struct F128ComplexToReal {
    inner: Arc<dyn vendor_rfft::ComplexToReal<F128>>,
}

impl F128ComplexToReal {
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
    /// [`len()`](Self::len)), converting to/from `f128` precision at the boundary. `input` is
    /// left in an unspecified state after the call, matching `realfft`.
    pub(crate) fn process(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<(), F128FftError> {
        let mut f128_in = to_f128_complex(input);
        let mut f128_out = vec![F128::from_f64(0.0); output.len()];
        self.inner.process(&mut f128_in, &mut f128_out).map_err(F128FftError)?;
        write_from_f128(output, &f128_out);
        Ok(())
    }
}

/// Maps our vendored (but field-for-field identical) `FftError` to the real `realfft` crate's
/// `FftError`, so [`F128RealToComplex`]/[`F128ComplexToReal`] can implement the real
/// `realfft::RealToComplex`/`ComplexToReal` traits below and slot into `core.rs`'s existing
/// `Arc<dyn RealToComplex<T>>` / `Arc<dyn ComplexToReal<T>>` fields unchanged.
fn map_error(e: F128FftError) -> realfft::FftError {
    match e.0 {
        vendor_rfft::FftError::InputBuffer(expected, got) => realfft::FftError::InputBuffer(expected, got),
        vendor_rfft::FftError::OutputBuffer(expected, got) => realfft::FftError::OutputBuffer(expected, got),
        vendor_rfft::FftError::ScratchBuffer(expected, got) => realfft::FftError::ScratchBuffer(expected, got),
        vendor_rfft::FftError::InputValues(first, last) => realfft::FftError::InputValues(first, last),
    }
}

// `get_scratch_len` returns 0 and `process_with_scratch` ignores the caller-provided scratch:
// the `f128` engine's own working buffers are `Vec<Complex<F128>>`, not `Vec<Complex<f64>>`, so
// an `f64`-shaped scratch buffer from a caller can't actually be reused here regardless. This
// only gives up the scratch-reuse optimization `process_with_scratch` normally provides, not
// correctness (see `real.rs` module docs re: this engine's allocation-per-call cost already).

impl realfft::RealToComplex<f64> for F128RealToComplex {
    fn process(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<(), realfft::FftError> {
        F128RealToComplex::process(self, input, output).map_err(map_error)
    }

    fn process_with_scratch(
        &self,
        input: &mut [f64],
        output: &mut [Complex<f64>],
        _scratch: &mut [Complex<f64>],
    ) -> Result<(), realfft::FftError> {
        F128RealToComplex::process(self, input, output).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        F128RealToComplex::len(self)
    }

    fn make_input_vec(&self) -> Vec<f64> {
        F128RealToComplex::make_input_vec(self)
    }

    fn make_output_vec(&self) -> Vec<Complex<f64>> {
        F128RealToComplex::make_output_vec(self)
    }

    fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
        Vec::new()
    }
}

impl realfft::ComplexToReal<f64> for F128ComplexToReal {
    fn process(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<(), realfft::FftError> {
        F128ComplexToReal::process(self, input, output).map_err(map_error)
    }

    fn process_with_scratch(
        &self,
        input: &mut [Complex<f64>],
        output: &mut [f64],
        _scratch: &mut [Complex<f64>],
    ) -> Result<(), realfft::FftError> {
        F128ComplexToReal::process(self, input, output).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        F128ComplexToReal::len(self)
    }

    fn make_input_vec(&self) -> Vec<Complex<f64>> {
        F128ComplexToReal::make_input_vec(self)
    }

    fn make_output_vec(&self) -> Vec<f64> {
        F128ComplexToReal::make_output_vec(self)
    }

    fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
        Vec::new()
    }
}

/// Plans an `f128`-precision forward real FFT, already coerced to the real `realfft` crate's
/// `RealToComplex<f64>` trait object -- this is the shape `core.rs` needs to drop it into its
/// existing (generic-over-`T`) `Arc<dyn realfft::RealToComplex<T>>` field for `T = f64`.
pub(crate) fn plan_fft_forward(len: usize) -> Arc<dyn realfft::RealToComplex<f64>> {
    F128RealFftPlanner::new().plan_fft_forward(len) as Arc<dyn realfft::RealToComplex<f64>>
}

/// Inverse counterpart of [`plan_fft_forward`].
pub(crate) fn plan_fft_inverse(len: usize) -> Arc<dyn realfft::ComplexToReal<f64>> {
    F128RealFftPlanner::new().plan_fft_inverse(len) as Arc<dyn realfft::ComplexToReal<f64>>
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Forward-then-inverse should return (a scaled copy of) the original signal, for both
    /// power-of-two and non-power-of-two lengths -- confirms the vendored mixed-radix/Bluestein
    /// selection logic still does the right thing once its twiddles are backed by `F128`.
    #[test]
    fn round_trip_recovers_signal() {
        for len in [1usize, 2, 3, 4, 5, 7, 8, 12, 17, 32, 100, 101, 257, 1024, 4000] {
            let mut planner = F128RealFftPlanner::new();
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

    /// Forward-then-inverse round trips aren't a great way to see `F128`'s precision benefit: the
    /// forward and inverse rounding errors substantially cancel, and the final result is narrowed
    /// back to `f64` regardless of which engine produced it, so both engines end up pinned near
    /// the same `f64`-representable floor (see the (removed) first version of this test, which
    /// asserted a gap that round-tripping doesn't actually produce).
    ///
    /// The forward *spectrum*, before any such cancellation, is where the benefit actually shows.
    /// This compares both engines' forward spectra at a few bins against a brute-force O(N)
    /// direct DFT computed independently in `F128` arithmetic (i.e. not via any FFT butterfly
    /// structure) -- for `len` this small, that reference is accurate to within a handful of
    /// `f128` ULPs (~1e-34), many orders of magnitude tighter than either FFT engine's own
    /// `f64`-representable output, so it's trustworthy as ground truth here.
    #[test]
    fn forward_spectrum_is_more_accurate_than_f64() {
        let len = 512usize;
        let original: Vec<f64> = (0..len)
            .map(|i| (i as f64 * 0.083).sin() * 0.9 + (i as f64 * 1.7).cos() * 0.1)
            .collect();

        let mut planner = F128RealFftPlanner::new();
        let r2c = planner.plan_fft_forward(len);
        let mut input = original.clone();
        let mut f128_spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut f128_spectrum).unwrap();

        let mut f64_planner = realfft::RealFftPlanner::<f64>::new();
        let f64_r2c = f64_planner.plan_fft_forward(len);
        let mut f64_input = original.clone();
        let mut f64_spectrum = f64_r2c.make_output_vec();
        f64_r2c.process(&mut f64_input, &mut f64_spectrum).unwrap();

        let mut total_f128_error = 0.0f64;
        let mut total_f64_error = 0.0f64;
        for &k in &[0usize, 1, len / 4, len / 2 - 1, len / 2] {
            let reference = direct_dft_bin_f128(&original, k, len);
            let f128_error = (f128_spectrum[k] - reference).norm();
            let f64_error = (f64_spectrum[k] - reference).norm();
            total_f128_error += f128_error;
            total_f64_error += f64_error;
        }

        assert!(
            total_f128_error <= total_f64_error,
            "expected the f128 spectrum (total error {total_f128_error}) to be at least as \
             accurate as plain f64's (total error {total_f64_error})"
        );
        assert!(
            total_f64_error > 0.0,
            "test is vacuous if the f64 engine has no error to beat"
        );
    }

    /// Computes one DFT output bin directly (`O(len)`), in `F128` arithmetic, independent of any
    /// FFT algorithm -- used as a high-precision reference in
    /// [`forward_spectrum_is_more_accurate_than_f64`].
    fn direct_dft_bin_f128(signal: &[f64], bin: usize, len: usize) -> Complex<f64> {
        use crate::f128_fft::numeric::F128;
        use crate::f128_fft::vendor::rustfft::FftDirection;

        let mut sum = Complex::new(F128::from_f64(0.0), F128::from_f64(0.0));
        for (n, &sample) in signal.iter().enumerate() {
            let angle_index = (bin * n) % len;
            let twiddle = F128::twiddle_factor(angle_index, len, FftDirection::Forward);
            let sample = F128::from_f64(sample);
            sum.re = sum.re + sample * twiddle.re;
            sum.im = sum.im + sample * twiddle.im;
        }
        Complex::new(sum.re.to_f64(), sum.im.to_f64())
    }
}
