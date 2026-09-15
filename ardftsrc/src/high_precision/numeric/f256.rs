//! IEEE binary256 (~237-bit) backend, backed by the `f256` crate.

use f256::f256;

use super::{HighPrecisionFloat, impl_high_precision_newtype};

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct F256(pub(crate) f256);

impl HighPrecisionFloat for F256 {
    #[inline]
    fn from_f64_exact(value: f64) -> Self {
        F256(f256::from(value))
    }

    #[inline]
    fn to_f64(self) -> f64 {
        nearest_f64(self.0)
    }

    #[inline]
    fn hp_abs(self) -> Self {
        F256(self.0.abs())
    }

    #[inline]
    fn hp_is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    #[inline]
    fn hp_round(self) -> Self {
        F256(self.0.round())
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (F256(sin), F256(cos))
    }

    #[inline]
    fn tau() -> Self {
        F256(::f256::consts::TAU)
    }
}

impl_high_precision_newtype!(F256, |value| value);

/// Width of the binary256 exponent field.
const EXP_BITS: u32 = 19;
/// Fraction bits stored in the high `u128` of `f256::to_bits` (below the sign and exponent).
const HI_FRACTION_BITS: u32 = 128 - 1 - EXP_BITS;
const EXP_MAX: u128 = (1 << EXP_BITS) - 1;
const EXP_BIAS: i64 = (1 << (EXP_BITS - 1)) - 1;

/// Rounds a binary256 value to the nearest `f64` (ties to even), including overflow to infinity
/// and gradual underflow into subnormals. `f256` has no conversion in this direction.
fn nearest_f64(x: f256) -> f64 {
    let (hi, lo) = x.to_bits();
    let sign = ((hi >> 127) as u64) << 63;
    let biased_exp = (hi >> HI_FRACTION_BITS) & EXP_MAX;
    let fraction_hi = hi & ((1 << HI_FRACTION_BITS) - 1);

    if biased_exp == EXP_MAX {
        return if fraction_hi == 0 && lo == 0 {
            f64::from_bits(sign | f64::INFINITY.to_bits())
        } else {
            f64::NAN
        };
    }
    // binary256 zeros and subnormals are all below half the smallest f64 subnormal.
    if biased_exp == 0 {
        return f64::from_bits(sign);
    }

    let exp = biased_exp as i64 - EXP_BIAS;
    if exp > f64::MAX_EXP as i64 - 1 {
        return f64::from_bits(sign | f64::INFINITY.to_bits());
    }
    // Anything below 2^-1076 is under half the smallest f64 subnormal (2^-1074).
    if exp < -1076 {
        return f64::from_bits(sign);
    }

    // Top 64 significand bits (explicit leading one at bit 63); the rest only matters as sticky.
    let significand_hi = (1 << HI_FRACTION_BITS) | fraction_hi;
    let discarded_bits = HI_FRACTION_BITS + 1 - 64;
    let top = (significand_hi >> discarded_bits) as u64;
    let sticky = significand_hi & ((1 << discarded_bits) - 1) != 0 || lo != 0;

    // Normal results keep 53 bits; subnormal results lose one more bit per binade below 2^-1022.
    const MIN_NORMAL_EXP: i64 = -1022;
    let (biased_out, shift) = if exp >= MIN_NORMAL_EXP {
        ((exp + 1023) as u64, 11u32)
    } else {
        (0, 11 + (MIN_NORMAL_EXP - exp) as u32)
    };

    // Work in a u128 window so shifts of up to 65 bits stay in range.
    let window = (top as u128) << 62;
    let shift = shift + 62;
    let mut mantissa = (window >> shift) as u64;
    let remainder = window & ((1 << shift) - 1);
    let half = 1u128 << (shift - 1);
    if remainder > half || (remainder == half && (sticky || mantissa & 1 == 1)) {
        mantissa += 1;
    }

    if biased_out == 0 {
        // Subnormal; a carry into bit 52 correctly becomes the smallest normal exponent.
        return f64::from_bits(sign | mantissa);
    }
    let (mantissa, biased_out) = if mantissa == 1 << 53 {
        (mantissa >> 1, biased_out + 1)
    } else {
        (mantissa, biased_out)
    };
    if biased_out >= 0x7FF {
        return f64::from_bits(sign | f64::INFINITY.to_bits());
    }
    f64::from_bits(sign | (biased_out << 52) | (mantissa & ((1 << 52) - 1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(value: f64) -> f256 {
        f256::from(value)
    }

    #[test]
    fn roundtrips_special_and_subnormal_values() {
        for value in [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            2.2250738585072009e-308,
            1.0 + f64::EPSILON,
        ] {
            assert_eq!(nearest_f64(f(value)).to_bits(), value.to_bits(), "{value:e}");
        }
        assert!(nearest_f64(f256::NAN).is_nan());
    }

    #[test]
    fn rounds_ties_to_even() {
        let half_ulp_of_one = f(f64::EPSILON / 2.0);
        // 1 + ulp/2 ties to the even neighbour 1.
        assert_eq!(nearest_f64(f(1.0) + half_ulp_of_one), 1.0);
        // (1 + ulp) + ulp/2 ties to the even neighbour 1 + 2ulp.
        assert_eq!(nearest_f64(f(1.0 + f64::EPSILON) + half_ulp_of_one), 1.0 + 2.0 * f64::EPSILON);
        // Just above the tie rounds up; the excess is far below f64's reach, so only sticky sees it.
        let tiny = f(f64::EPSILON) * f(2f64.powi(-100));
        assert_eq!(nearest_f64(f(1.0) + half_ulp_of_one + tiny), 1.0 + f64::EPSILON);
        assert_eq!(nearest_f64(f(1.0) + half_ulp_of_one - tiny), 1.0);
    }

    #[test]
    fn overflows_and_underflows() {
        assert_eq!(nearest_f64(f(f64::MAX) * f(2.0)), f64::INFINITY);
        assert_eq!(nearest_f64(-f(f64::MAX) * f(2.0)), f64::NEG_INFINITY);
        // Just below the overflow threshold (MAX + ulp/2) rounds back down to MAX. MAX's
        // significand is ~2, so half its ulp is MAX * EPSILON / 4.
        let half_ulp_of_max = f(f64::MAX) * f(f64::EPSILON / 4.0) * f(0.999);
        assert_eq!(nearest_f64(f(f64::MAX) + half_ulp_of_max), f64::MAX);
        let above_half_ulp_of_max = f(f64::MAX) * f(f64::EPSILON / 4.0) * f(1.001);
        assert_eq!(nearest_f64(f(f64::MAX) + above_half_ulp_of_max), f64::INFINITY);
        assert_eq!(nearest_f64(f(f64::MAX) + half_ulp_of_max), f64::MAX);
        // Half the smallest subnormal ties to zero; slightly more rounds up to it.
        let min_subnormal = f(5e-324);
        assert_eq!(nearest_f64(min_subnormal / f(2.0)).to_bits(), 0.0f64.to_bits());
        assert_eq!(nearest_f64(min_subnormal * f(0.75)), 5e-324);
        assert_eq!(nearest_f64(min_subnormal * f(0.25)), 0.0);
        // Rounding the largest subnormal up carries into the smallest normal.
        let largest_subnormal = f(2.2250738585072009e-308);
        assert_eq!(nearest_f64(largest_subnormal + min_subnormal * f(0.75)), f64::MIN_POSITIVE);
    }
}
