//! Numeric types used internally by the [`high_precision`](super) engines.
//!
//! None of the backing types (`TwoFloat`, `rustc_apfloat::ieee::Quad`, `f256::f256`) implement
//! the full set of `num_traits` traits (`Zero`, `One`, `Num`, `Signed`, `FromPrimitive`) that
//! `FftNum` and `num_complex::Complex<T>` require, and orphan rules mean we can't add those impls
//! to foreign types from here. Each backend is therefore a local newtype that carries those impls.
//! The per-type leaf operations live in [`HighPrecisionFloat`]; [`impl_high_precision_newtype!`]
//! derives everything else from them, including the vendored engine's `FftNum` impl.

mod double_double;
mod f128;
mod f256;
mod trig;

pub(crate) use double_double::Dd;
pub(crate) use f128::F128;
pub(crate) use f256::F256;

use num_complex::Complex;

use super::vendor::rustfft::{FftDirection, FftNum};

/// Leaf operations each high-precision backend supplies natively.
///
/// Method names are prefixed where they would otherwise collide with `num_traits` methods of the
/// same name (`Signed::abs`, `FromPrimitive::from_f64`).
pub(crate) trait HighPrecisionFloat: FftNum + PartialOrd {
    /// Widens an `f64` without rounding (every backend's mantissa is a superset of `f64`'s).
    fn from_f64_exact(value: f64) -> Self;
    /// Narrows to the nearest `f64` (ties to even).
    fn to_f64(self) -> f64;
    fn hp_abs(self) -> Self;
    fn hp_is_sign_negative(self) -> bool;
    /// Rounds to an integral value (the direction of halfway cases is unspecified).
    fn hp_round(self) -> Self;
    /// Sine and cosine at this type's full precision.
    fn sin_cos(self) -> (Self, Self);
    /// `2*pi` at this type's full precision.
    fn tau() -> Self;
}

/// Builds the unit-magnitude twiddle factor `exp(-2*pi*i*numerator/denominator)` (or its
/// conjugate for the inverse direction), computing the angle and its sine/cosine entirely in
/// `T`'s precision rather than routing through an `f64` intermediate.
pub(crate) fn twiddle_factor<T: HighPrecisionFloat>(
    numerator: usize,
    denominator: usize,
    direction: FftDirection,
) -> Complex<T> {
    debug_assert!(denominator > 0);
    debug_assert!(numerator < (1 << f64::MANTISSA_DIGITS) && denominator < (1 << f64::MANTISSA_DIGITS));
    let turns = T::from_f64_exact(numerator as f64) / T::from_f64_exact(denominator as f64);
    let angle = -(turns * T::tau());
    let (sin, cos) = angle.sin_cos();
    let result = Complex { re: cos, im: sin };
    match direction {
        FftDirection::Forward => result,
        FftDirection::Inverse => result.conj(),
    }
}

/// Implements arithmetic, `num_traits`, `Debug`, and `FftNum` for a single-field newtype whose
/// inner type supports `+ - * / %` and unary `-`, in terms of [`HighPrecisionFloat`].
///
/// `$finish` maps the inner type's binary-operator output back to the inner type (identity for
/// most types; `.value` for `rustc_apfloat`, whose operators return `StatusAnd<T>`).
macro_rules! impl_high_precision_newtype {
    ($ty:ident, $finish:expr) => {
        impl_high_precision_newtype!(@binop $ty, $finish, Add, add, AddAssign, add_assign, +);
        impl_high_precision_newtype!(@binop $ty, $finish, Sub, sub, SubAssign, sub_assign, -);
        impl_high_precision_newtype!(@binop $ty, $finish, Mul, mul, MulAssign, mul_assign, *);
        impl_high_precision_newtype!(@binop $ty, $finish, Div, div, DivAssign, div_assign, /);
        impl_high_precision_newtype!(@binop $ty, $finish, Rem, rem, RemAssign, rem_assign, %);

        impl ::std::ops::Neg for $ty {
            type Output = $ty;
            #[inline]
            fn neg(self) -> $ty {
                $ty(-self.0)
            }
        }

        impl ::num_traits::Zero for $ty {
            #[inline]
            fn zero() -> Self {
                <$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact(0.0)
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == <Self as ::num_traits::Zero>::zero()
            }
        }

        impl ::num_traits::One for $ty {
            #[inline]
            fn one() -> Self {
                <$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact(1.0)
            }
        }

        impl ::num_traits::Num for $ty {
            type FromStrRadixErr = <f64 as ::num_traits::Num>::FromStrRadixErr;
            #[inline]
            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <f64 as ::num_traits::Num>::from_str_radix(str, radix)
                    .map(<$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact)
            }
        }

        impl ::num_traits::Signed for $ty {
            #[inline]
            fn abs(&self) -> Self {
                $crate::high_precision::numeric::HighPrecisionFloat::hp_abs(*self)
            }
            #[inline]
            fn abs_sub(&self, other: &Self) -> Self {
                if *self <= *other { <Self as ::num_traits::Zero>::zero() } else { *self - *other }
            }
            #[inline]
            fn signum(&self) -> Self {
                if self.is_negative() {
                    -<Self as ::num_traits::One>::one()
                } else {
                    <Self as ::num_traits::One>::one()
                }
            }
            #[inline]
            fn is_positive(&self) -> bool {
                !self.is_negative()
            }
            #[inline]
            fn is_negative(&self) -> bool {
                $crate::high_precision::numeric::HighPrecisionFloat::hp_is_sign_negative(*self)
            }
        }

        impl ::num_traits::FromPrimitive for $ty {
            #[inline]
            fn from_i64(n: i64) -> Option<Self> {
                Some(<$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact(n as f64))
            }
            #[inline]
            fn from_u64(n: u64) -> Option<Self> {
                Some(<$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact(n as f64))
            }
            #[inline]
            fn from_f32(n: f32) -> Option<Self> {
                Some(<$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact(n as f64))
            }
            #[inline]
            fn from_f64(n: f64) -> Option<Self> {
                Some(<$ty as $crate::high_precision::numeric::HighPrecisionFloat>::from_f64_exact(n))
            }
        }

        impl ::std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                ::std::fmt::Debug::fmt(
                    &$crate::high_precision::numeric::HighPrecisionFloat::to_f64(*self),
                    f,
                )
            }
        }

        impl $crate::high_precision::vendor::rustfft::FftNum for $ty {
            fn twiddle(
                numerator: usize,
                denominator: usize,
                direction: $crate::high_precision::vendor::rustfft::FftDirection,
            ) -> ::num_complex::Complex<Self> {
                $crate::high_precision::numeric::twiddle_factor(numerator, denominator, direction)
            }
        }
    };

    (@binop $ty:ident, $finish:expr, $trait_:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl ::std::ops::$trait_ for $ty {
            type Output = $ty;
            #[inline]
            fn $method(self, rhs: $ty) -> $ty {
                $ty(($finish)(self.0 $op rhs.0))
            }
        }

        impl ::std::ops::$assign_trait for $ty {
            #[inline]
            fn $assign_method(&mut self, rhs: $ty) {
                *self = *self $op rhs;
            }
        }
    };
}

pub(crate) use impl_high_precision_newtype;

#[cfg(test)]
mod tests {
    use super::*;
    use ::f256::f256;

    /// Converts any backend value to `f256` without rounding by peeling off `f64` limbs. Three
    /// limbs (159 bits) cover every backend's mantissa, and each subtraction is exact because it
    /// removes the nearest `f64` to the remaining value.
    fn to_f256<T: HighPrecisionFloat>(x: T) -> f256 {
        let mut rest = x;
        let mut sum = f256::from(0.0);
        for _ in 0..3 {
            let limb = rest.to_f64();
            sum += f256::from(limb);
            rest = rest - T::from_f64_exact(limb);
        }
        sum
    }

    fn assert_f64_roundtrip_is_exact<T: HighPrecisionFloat>() {
        for value in [0.0, -0.0, 1.0, -1.0, 0.1, 123.456, 1e-300, 1e300, f64::MAX, f64::MIN_POSITIVE] {
            assert_eq!(T::from_f64_exact(value).to_f64(), value);
        }
    }

    fn assert_twiddle_matches_f64<T: HighPrecisionFloat>() {
        for denominator in [4usize, 5, 7, 1024] {
            for numerator in 0..denominator {
                let hp = twiddle_factor::<T>(numerator, denominator, FftDirection::Forward);
                let angle = -2.0 * std::f64::consts::PI * numerator as f64 / denominator as f64;
                assert!((hp.re.to_f64() - angle.cos()).abs() < 1e-14);
                assert!((hp.im.to_f64() - angle.sin()).abs() < 1e-14);
            }
        }
    }

    fn assert_sin_squared_plus_cos_squared_is_one<T: HighPrecisionFloat>(tolerance: f64) {
        for angle in [0.0f64, 0.5, 1.0, 2.5, -6.2, 6.283185307179586, 3.14159265358979] {
            let (s, c) = T::from_f64_exact(angle).sin_cos();
            let residual = (s * s + c * c - T::from_f64_exact(1.0)).to_f64();
            assert!(residual.abs() < tolerance, "sin^2+cos^2 != 1 at {angle}: residual {residual:e}");
        }
    }

    /// A quarter turn lands exactly on a quadrant boundary of every range-reduction scheme.
    fn assert_quarter_turn_is_exact<T: HighPrecisionFloat>(tolerance: f64) {
        let quarter = twiddle_factor::<T>(1, 4, FftDirection::Forward);
        let re = quarter.re.to_f64();
        let im_plus_one = (quarter.im + T::from_f64_exact(1.0)).to_f64();
        assert!(re.abs() < tolerance, "cos(-pi/2) should be ~0, got {re:e}");
        assert!(im_plus_one.abs() < tolerance, "sin(-pi/2) should be ~-1, off by {im_plus_one:e}");
    }

    /// Checks every twiddle on a grid of FFT lengths against `F256`, whose trig comes from an
    /// independent implementation with far more headroom than the backend under test.
    fn assert_twiddles_match_f256<T: HighPrecisionFloat>(tolerance: f64) {
        for denominator in [5usize, 7, 1024, 4096] {
            for numerator in 0..denominator {
                let hp = twiddle_factor::<T>(numerator, denominator, FftDirection::Forward);
                let reference = twiddle_factor::<F256>(numerator, denominator, FftDirection::Forward);
                let re_error = F256(to_f256(hp.re) - reference.re.0).to_f64();
                let im_error = F256(to_f256(hp.im) - reference.im.0).to_f64();
                assert!(
                    re_error.abs() < tolerance && im_error.abs() < tolerance,
                    "twiddle {numerator}/{denominator}: re error {re_error:e}, im error {im_error:e}"
                );
            }
        }
    }

    #[test]
    fn f64_roundtrip_is_exact() {
        assert_f64_roundtrip_is_exact::<Dd>();
        assert_f64_roundtrip_is_exact::<F128>();
        assert_f64_roundtrip_is_exact::<F256>();
    }

    #[test]
    fn twiddle_matches_f64_to_within_f64_precision() {
        assert_twiddle_matches_f64::<Dd>();
        assert_twiddle_matches_f64::<F128>();
        assert_twiddle_matches_f64::<F256>();
    }

    #[test]
    fn sin_squared_plus_cos_squared_is_one() {
        assert_sin_squared_plus_cos_squared_is_one::<Dd>(1e-30);
        assert_sin_squared_plus_cos_squared_is_one::<F128>(1e-32);
        assert_sin_squared_plus_cos_squared_is_one::<F256>(1e-65);
    }

    #[test]
    fn quarter_turn_is_exact() {
        assert_quarter_turn_is_exact::<Dd>(1e-30);
        assert_quarter_turn_is_exact::<F128>(1e-33);
        assert_quarter_turn_is_exact::<F256>(1e-65);
    }

    #[test]
    fn double_double_twiddles_match_f256() {
        assert_twiddles_match_f256::<Dd>(1e-30);
    }

    #[test]
    fn f128_twiddles_match_f256() {
        assert_twiddles_match_f256::<F128>(1e-33);
    }
}
