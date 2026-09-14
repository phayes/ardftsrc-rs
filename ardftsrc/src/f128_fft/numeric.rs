//! `f128` numeric type used internally by the [`f128_fft`](super) engine.
//!
//! `f128` itself can't be used directly as the vendored engine's `T: FftNum` because it doesn't
//! implement the `num_traits` traits (`Zero`, `One`, `Num`, `Signed`, `FromPrimitive`) that
//! `FftNum` and `num_complex::Complex<T>` require, and orphan rules mean we can't add those impls
//! to a foreign (indeed, primitive) type from here. `F128` is a local newtype that exists purely
//! to carry those impls -- every method below is a thin forward onto the wrapped `f128`.
//!
//! std's `f128` ships `sin`/`cos`/`sin_cos`/`sqrt`/`round`, but on this codebase's target
//! platforms those currently return incorrect results (the intrinsics they're backed by aren't
//! implemented correctly yet). `sin_cos` and the rounding used by its range reduction are
//! therefore hand-rolled below from primitive arithmetic (`+ - * /` and comparisons, which are
//! correct) rather than calling std's versions. `sqrt` is never needed here: the vendored engine
//! only calls it on fixed `f32`/`f64` internals, never on the generic numeric type.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use num_complex::Complex;
use num_traits::{FromPrimitive, Num, One, Signed, Zero};

use super::vendor::rustfft::FftDirection;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct F128(pub(crate) f128);

impl F128 {
    #[inline]
    pub(crate) fn from_f64(value: f64) -> Self {
        F128(value as f128)
    }

    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        self.0 as f64
    }

    /// Builds the unit-magnitude twiddle factor `exp(-2*pi*i*numerator/denominator)` (or its
    /// conjugate for the inverse direction), computing the angle and its sine/cosine entirely in
    /// `f128` precision rather than routing through an `f64` intermediate.
    pub(crate) fn twiddle_factor(numerator: usize, denominator: usize, direction: FftDirection) -> Complex<F128> {
        debug_assert!(denominator > 0);
        let turns = F128::from_f64(numerator as f64) / F128::from_f64(denominator as f64);
        let angle = -turns.0 * trig::TAU;
        let (sin, cos) = trig::sin_cos(angle);
        let result = Complex {
            re: F128(cos),
            im: F128(sin),
        };
        match direction {
            FftDirection::Forward => result,
            FftDirection::Inverse => result.conj(),
        }
    }
}

impl fmt::Debug for F128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&(self.0 as f64), f)
    }
}

macro_rules! forward_binop {
    ($trait_:ident, $method:ident) => {
        impl $trait_ for F128 {
            type Output = F128;
            #[inline]
            fn $method(self, rhs: F128) -> F128 {
                F128($trait_::$method(self.0, rhs.0))
            }
        }
    };
}

forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);
forward_binop!(Div, div);
forward_binop!(Rem, rem);

macro_rules! forward_assign_op {
    ($trait_:ident, $method:ident) => {
        impl $trait_ for F128 {
            #[inline]
            fn $method(&mut self, rhs: F128) {
                $trait_::$method(&mut self.0, rhs.0)
            }
        }
    };
}

forward_assign_op!(AddAssign, add_assign);
forward_assign_op!(SubAssign, sub_assign);
forward_assign_op!(MulAssign, mul_assign);
forward_assign_op!(DivAssign, div_assign);
forward_assign_op!(RemAssign, rem_assign);

impl Neg for F128 {
    type Output = F128;
    #[inline]
    fn neg(self) -> F128 {
        F128(-self.0)
    }
}

impl Zero for F128 {
    #[inline]
    fn zero() -> Self {
        F128(0.0)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for F128 {
    #[inline]
    fn one() -> Self {
        F128(1.0)
    }
}

impl Num for F128 {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <f64 as Num>::from_str_radix(str, radix).map(F128::from_f64)
    }
}

impl Signed for F128 {
    #[inline]
    fn abs(&self) -> Self {
        F128(self.0.abs())
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other { F128::zero() } else { *self - *other }
    }
    #[inline]
    fn signum(&self) -> Self {
        F128(self.0.signum())
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.0.is_sign_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.0.is_sign_negative()
    }
}

impl FromPrimitive for F128 {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        Some(F128::from_f64(n as f64))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        Some(F128::from_f64(n as f64))
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        Some(F128::from_f64(n as f64))
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        Some(F128::from_f64(n))
    }
}

/// Hand-rolled `f128` sine/cosine.
///
/// std's `f128::sin`/`cos`/`sin_cos`/`round` currently return incorrect results on this
/// codebase's target platforms, so this builds `sin_cos` from scratch out of the primitive
/// operations that *do* work: `+ - * /`, comparisons, and casts.
///
/// Range reduction follows the same Cody-Waite shape used by double-double and other
/// extended-precision libraries: reduce to `[-pi/4, pi/4]` by subtracting the nearest multiple of
/// `pi/2`, then evaluate a polynomial in that restricted range and use the quadrant to combine
/// sin/cos. `round_nearest` substitutes for std's broken `f128::round` via the standard
/// "magic number" trick (adding and subtracting 2^112 forces the FPU's own correctly-rounded
/// addition to round the value to the nearest integer at that binade), which only relies on `+`
/// and `-`.
///
/// The polynomial itself is a plain Taylor series rather than a minimax fit: on `[-pi/4, pi/4]`
/// the Taylor remainder is already below `1e-40` by the 16th term (the factorial in the
/// denominator dominates), well past `f128`'s ~1e-34 ulp, so no externally-sourced coefficient
/// table is needed.
mod trig {
    pub(super) const TAU: f128 = 6.28318530717958647692528676655900576839433879875021164194989_f128;
    const FRAC_PI_2: f128 = 1.57079632679489661923132169163975144209858469968755291048747_f128;
    const FRAC_PI_4: f128 = 0.785398163397448309615660845819875721049292349843776455243736_f128;

    /// 2^112: one past `f128`'s 112 explicit mantissa bits, used for magic-number rounding.
    const TWO_POW_112: f128 = 5192296858534827628530496329220096.0_f128;

    fn round_nearest(x: f128) -> f128 {
        if x.abs() >= TWO_POW_112 {
            return x; // already integral at this magnitude
        }
        if x >= 0.0 {
            (x + TWO_POW_112) - TWO_POW_112
        } else {
            (x - TWO_POW_112) + TWO_POW_112
        }
    }

    // Coefficient of x^(2k+1) in the Taylor series of sin(x), for k = 1..=16 (the leading x^1
    // term is applied separately in `restricted_sin`).
    const SIN_COEFFS: [f128; 16] = [
        -0.166666666666666666666666666666666666666666666666666666666667_f128,
        0.00833333333333333333333333333333333333333333333333333333333333_f128,
        -0.000198412698412698412698412698412698412698412698412698412698413_f128,
        0.00000275573192239858906525573192239858906525573192239858906525573_f128,
        -0.0000000250521083854417187750521083854417187750521083854417187750521_f128,
        0.000000000160590438368216145993923771701549479327257105034882812660590_f128,
        -0.000000000000764716373181981647590113198578807044415510023975632441240907_f128,
        0.00000000000000281145725434552076319894558301032001623349273520453103397392_f128,
        -0.00000000000000000822063524662432971695598123687228074922073899182611413442667_f128,
        0.0000000000000000000195729410633912612308475743735054303552874737900621765105397_f128,
        -0.0000000000000000000000386817017063068403771691193152281232317934264625734713647030_f128,
        0.0000000000000000000000000644695028438447339619485321920468720529890441042891189411716_f128,
        -0.0000000000000000000000000000918368986379554614842571683647391339786168719434317933634923_f128,
        0.000000000000000000000000000000113099628864477169315587645769383169924405014708659844043710_f128,
        -0.000000000000000000000000000000000121612504155351794962997468569229214972478510439419187143774_f128,
        0.000000000000000000000000000000000000115163356207719502805868814932982211148180407613086351461907_f128,
    ];

    // Coefficient of x^(2k) in the Taylor series of cos(x), for k = 1..=16 (the leading constant
    // 1 term is applied separately in `restricted_cos`).
    const COS_COEFFS: [f128; 16] = [
        -0.5_f128,
        0.0416666666666666666666666666666666666666666666666666666666667_f128,
        -0.00138888888888888888888888888888888888888888888888888888888889_f128,
        0.0000248015873015873015873015873015873015873015873015873015873016_f128,
        -0.000000275573192239858906525573192239858906525573192239858906525573_f128,
        0.00000000208767569878680989792100903212014323125434236545347656458768_f128,
        -0.0000000000114707455977297247138516979786821056662326503596344866186136_f128,
        0.0000000000000477947733238738529743820749111754402759693764984770275775567_f128,
        -0.000000000000000156192069685862264622163643500573334235194040844696168554107_f128,
        0.000000000000000000411031762331216485847799061843614037461036949591305706721334_f128,
        -0.000000000000000000000889679139245057328674889744250246834331248808639189841388168_f128,
        0.00000000000000000000000161173757109611834904871330480117180132472610260722797352929_f128,
        -0.00000000000000000000000000247959626322479746007494354584795661742265554247265842081429_f128,
        0.00000000000000000000000000000327988923706983791015204172731211192780774542655113547726758_f128,
        -0.00000000000000000000000000000000376998762881590564385292152564610566414683382362199480145699_f128,
        0.00000000000000000000000000000000000380039075485474359259367089278841296788995345123184959824293_f128,
    ];

    fn restricted_sin(x: f128) -> f128 {
        let x2 = x * x;
        let mut acc = SIN_COEFFS[SIN_COEFFS.len() - 1];
        for &coeff in SIN_COEFFS[..SIN_COEFFS.len() - 1].iter().rev() {
            acc = acc * x2 + coeff;
        }
        x * (1.0_f128 + x2 * acc)
    }

    fn restricted_cos(x: f128) -> f128 {
        let x2 = x * x;
        let mut acc = COS_COEFFS[COS_COEFFS.len() - 1];
        for &coeff in COS_COEFFS[..COS_COEFFS.len() - 1].iter().rev() {
            acc = acc * x2 + coeff;
        }
        1.0_f128 + x2 * acc
    }

    /// Reduces `value` to `[-pi/4, pi/4]` plus a quadrant in `0..4`.
    fn quadrant(value: f128) -> (f128, i32) {
        if value.abs() < FRAC_PI_4 {
            (value, 0)
        } else {
            let quotient = round_nearest(value / FRAC_PI_2);
            let remainder = value - quotient * FRAC_PI_2;
            (remainder, (quotient as i64).rem_euclid(4) as i32)
        }
    }

    pub(super) fn sin_cos(x: f128) -> (f128, f128) {
        let (r, quadrant) = quadrant(x);
        let s = restricted_sin(r);
        let c = restricted_cos(r);
        match quadrant {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_roundtrip_is_exact() {
        for value in [0.0, 1.0, -1.0, 0.1, 123.456, 1e-300, 1e300] {
            assert_eq!(F128::from_f64(value).to_f64(), value);
        }
    }

    #[test]
    fn twiddle_matches_f64_to_within_f64_precision() {
        for denominator in [4usize, 5, 7, 1024] {
            for numerator in 0..denominator {
                let dd = F128::twiddle_factor(numerator, denominator, FftDirection::Forward);
                let f64_result: Complex<f64> = {
                    let angle = -2.0 * std::f64::consts::PI * numerator as f64 / denominator as f64;
                    Complex::new(angle.cos(), angle.sin())
                };
                assert!((dd.re.to_f64() - f64_result.re).abs() < 1e-14);
                assert!((dd.im.to_f64() - f64_result.im).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn sin_cos_matches_f64_to_within_f64_precision() {
        for angle in [
            0.0f64,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            -0.5,
            -2.5,
            -6.2,
            6.283185307179586,
            -0.0000001,
        ] {
            let (s, c) = trig::sin_cos(angle as f128);
            assert!((s as f64 - angle.sin()).abs() < 1e-15, "sin({angle}) mismatch");
            assert!((c as f64 - angle.cos()).abs() < 1e-15, "cos({angle}) mismatch");
        }
    }

    #[test]
    fn sin_squared_plus_cos_squared_is_one() {
        for angle in [0.0f64, 0.5, 1.0, 2.5, -6.2, 6.283185307179586, 3.14159265358979] {
            let (s, c) = trig::sin_cos(angle as f128);
            let identity = s * s + c * c - 1.0_f128;
            assert!(
                (identity as f64).abs() < 1e-30,
                "sin^2+cos^2 != 1 at {angle}: residual {identity:?}"
            );
        }
    }

    #[test]
    fn round_nearest_is_exact_at_quadrant_boundaries() {
        // Regression case: std's f128::round() silently returns 0 on this codebase's target
        // platforms, which would otherwise corrupt range reduction exactly here.
        let half_pi = trig::sin_cos(1.57079632679489661923132169163975144209858469968755291048747_f128);
        assert!(
            (half_pi.0 as f64 - 1.0).abs() < 1e-33,
            "sin(pi/2) should be ~1, got {:?}",
            half_pi.0
        );
        assert!(
            (half_pi.1 as f64).abs() < 1e-33,
            "cos(pi/2) should be ~0, got {:?}",
            half_pi.1
        );
    }
}
