//! High-precision real FFT engines, compiled by the `high_precision` feature and selected with
//! [`Config::high_precision`](crate::Config::high_precision).
//!
//! `ardftsrc`'s default FFT backend (`realfft`, backed by `rustfft`) computes twiddle factors in
//! plain `f64`. This module provides alternative backends for callers who want to push past that
//! ceiling: they run the same FFT algorithms (mixed-radix, Bluestein's, Rader's -- vendored from
//! `rustfft`, see `vendor` for details and rationale) with every twiddle factor and internal
//! accumulation computed in a wider type:
//!
//! - [`HighPrecision::DoubleDouble`]: double-double (~106-bit mantissa), backed by `twofloat`.
//! - [`HighPrecision::F128`]: IEEE binary128 (113-bit mantissa), backed by `rustc_apfloat`.
//! - [`HighPrecision::F256`]: IEEE binary256 (237-bit mantissa), backed by `f256`.
//!
//! None of them are general replacements: there's no SIMD, and every backend is at least an order
//! of magnitude slower than `f64` per operation (the binary128 and binary256 backends are pure
//! software floats), so they are meant for offline / opt-in use at extreme quality settings, not
//! realtime streaming.

mod numeric;
mod real;
#[allow(unused)]
mod vendor;

use std::sync::Arc;

use crate::HighPrecision;
use numeric::{Dd, F128, F256};

/// Plans a forward real FFT of `len` samples in the selected precision.
pub(crate) fn plan_fft_forward(len: usize, precision: HighPrecision) -> Arc<dyn realfft::RealToComplex<f64>> {
    match precision {
        HighPrecision::DoubleDouble => real::plan_fft_forward::<Dd>(len),
        HighPrecision::F128 => real::plan_fft_forward::<F128>(len),
        HighPrecision::F256 => real::plan_fft_forward::<F256>(len),
    }
}

/// Inverse counterpart of [`plan_fft_forward`].
pub(crate) fn plan_fft_inverse(len: usize, precision: HighPrecision) -> Arc<dyn realfft::ComplexToReal<f64>> {
    match precision {
        HighPrecision::DoubleDouble => real::plan_fft_inverse::<Dd>(len),
        HighPrecision::F128 => real::plan_fft_inverse::<F128>(len),
        HighPrecision::F256 => real::plan_fft_inverse::<F256>(len),
    }
}
