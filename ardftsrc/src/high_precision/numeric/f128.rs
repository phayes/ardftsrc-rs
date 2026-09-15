//! IEEE binary128 (~113-bit) backend, backed by `rustc_apfloat`'s correctly rounded soft float.
//!
//! `rustc_apfloat` provides arithmetic, rounding, and conversions but no trigonometry, so
//! `sin_cos` is built from those primitives with [`trig`](super::trig).

use std::sync::LazyLock;

use rustc_apfloat::ieee::{Double, Quad};
use rustc_apfloat::{Float, FloatConvert, Round, StatusAnd};

use super::trig::{self, Coefficients};
use super::{HighPrecisionFloat, impl_high_precision_newtype};

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct F128(pub(crate) Quad);

impl HighPrecisionFloat for F128 {
    #[inline]
    fn from_f64_exact(value: f64) -> Self {
        let double = Double::from_bits(value.to_bits() as u128);
        F128(FloatConvert::<Quad>::convert(double, &mut false).value)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        let double: Double = FloatConvert::<Double>::convert(self.0, &mut false).value;
        f64::from_bits(double.to_bits() as u64)
    }

    #[inline]
    fn hp_abs(self) -> Self {
        F128(self.0.abs())
    }

    #[inline]
    fn hp_is_sign_negative(self) -> bool {
        self.0.is_negative()
    }

    #[inline]
    fn hp_round(self) -> Self {
        F128(self.0.round_to_integral(Round::NearestTiesToEven).value)
    }

    fn sin_cos(self) -> (Self, Self) {
        static COEFFICIENTS: LazyLock<Coefficients<F128>> = LazyLock::new(Coefficients::new);
        trig::sin_cos(self, &COEFFICIENTS)
    }

    fn tau() -> Self {
        static TAU: LazyLock<Quad> = LazyLock::new(|| {
            "6.28318530717958647692528676655900576839433879875021164194989"
                .parse()
                .expect("valid float literal")
        });
        F128(*TAU)
    }
}

impl_high_precision_newtype!(F128, |status: StatusAnd<Quad>| status.value);
