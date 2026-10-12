//! Numeric traits the resampling pipeline is generic over.
//!
//! [`DspFloat`] is the arithmetic the window, extrapolation, decimation, and spectral helpers
//! need; [`Sample`] adds the real FFT that [`CpuCore`](crate::cpu_core::CpuCore) runs. Both are
//! implemented for every `num_traits::Float` type (in practice `f32` and `f64`, which use the SIMD
//! `realfft` engine) and, with the `high_precision` feature, for the wide types in
//! `high_precision`, which do not implement `num_traits::Float`.
//!
//! Filter coefficients (tapers, phase tables, decimation taps) are designed in `f64` and
//! converted with [`DspFloat::cast_f64`], so in a wider type they are exact `f64` values.
//! Rounding a coefficient perturbs the filter's response by a fixed amount; it does not add
//! signal-dependent error.
//!
//! `DspFloat` is `pub` only so it can be a supertrait of the public [`Sample`]; this module is
//! private, so callers can neither name nor implement it.

use std::ops::Neg;
use std::sync::Arc;

use num_traits::{Float, Num, NumCast, ToPrimitive};
use realfft::{ComplexToReal, FftNum, RealFftPlanner, RealToComplex};

/// Arithmetic shared by every stage of the resampling pipeline.
///
/// Method names avoid `num_traits`' `FromPrimitive::from_f64`/`ToPrimitive::to_f64`, which
/// `f32`/`f64` and the wide types also implement.
pub trait DspFloat: Copy + Num + Neg<Output = Self> + PartialOrd {
    /// The nearest representable value to `value` (exact for `f64` and wider types).
    fn cast_f64(value: f64) -> Self;
    /// The nearest representable value to `value`.
    fn cast_usize(value: usize) -> Self;
    /// The gap between `1` and the next larger representable value, as an `f64`.
    fn machine_epsilon() -> f64;
    fn abs(self) -> Self;
    /// The larger operand; a NaN operand is ignored, as with `f64::max`.
    fn max(self, other: Self) -> Self;
    fn is_finite(self) -> bool;
    /// Whether the sign bit is set, including for `-0`.
    fn is_sign_negative(self) -> bool;
}

/// A sample type ardftsrc can resample in.
///
/// Every stage of the resampler runs in the sample type, including both FFTs, so it sets the
/// precision of the whole pipeline. It is implemented for every `num_traits::Float` type that is
/// also a `realfft::FftNum` (in practice `f32` and `f64`, which use a SIMD FFT) and, with the
/// `high_precision` feature, for `DoubleDouble`, `F128`, and `F256`. Convert samples into and
/// out of the wide types with their `From` impls and `to_f64`.
///
/// It cannot be implemented for other types.
pub trait Sample: DspFloat + FftNum {
    /// Plans a forward real FFT of `len` samples whose twiddles and accumulation are computed at
    /// this type's full precision.
    #[doc(hidden)]
    fn plan_fft_forward(len: usize) -> Arc<dyn RealToComplex<Self>>;
    /// Inverse counterpart of [`plan_fft_forward`](Self::plan_fft_forward).
    #[doc(hidden)]
    fn plan_fft_inverse(len: usize) -> Arc<dyn ComplexToReal<Self>>;
}

impl<T: Float> DspFloat for T {
    #[inline]
    fn cast_f64(value: f64) -> Self {
        <T as NumCast>::from(value).unwrap_or_else(T::nan)
    }

    #[inline]
    fn cast_usize(value: usize) -> Self {
        <T as NumCast>::from(value).unwrap_or_else(T::nan)
    }

    #[inline]
    fn machine_epsilon() -> f64 {
        ToPrimitive::to_f64(&T::epsilon()).unwrap_or(f64::NAN)
    }

    #[inline]
    fn abs(self) -> Self {
        Float::abs(self)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        Float::max(self, other)
    }

    #[inline]
    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        Float::is_sign_negative(self)
    }
}

impl<T: Float + FftNum> Sample for T {
    fn plan_fft_forward(len: usize) -> Arc<dyn RealToComplex<Self>> {
        RealFftPlanner::<T>::new().plan_fft_forward(len)
    }

    fn plan_fft_inverse(len: usize) -> Arc<dyn ComplexToReal<Self>> {
        RealFftPlanner::<T>::new().plan_fft_inverse(len)
    }
}
