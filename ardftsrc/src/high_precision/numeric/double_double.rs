//! Double-double (~106-bit) backend, backed by `twofloat`.
//!
//! `twofloat`'s arithmetic is used directly, but its own `sin_cos` evaluates a 7-term polynomial
//! that is only accurate to roughly 1e-16..1e-21 -- far short of double-double's ~1e-32 -- so
//! `sin_cos` is built from that arithmetic with [`trig`](super::trig) instead.

use std::sync::LazyLock;

use twofloat::TwoFloat;

use super::trig::{self, Coefficients};
use super::{HighPrecisionFloat, impl_high_precision_newtype};

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct Dd(pub(crate) TwoFloat);

impl HighPrecisionFloat for Dd {
    #[inline]
    fn from_f64_exact(value: f64) -> Self {
        Dd(TwoFloat::from(value))
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self.0)
    }

    #[inline]
    fn hp_abs(self) -> Self {
        Dd(self.0.abs())
    }

    #[inline]
    fn hp_is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    #[inline]
    fn hp_round(self) -> Self {
        Dd(self.0.round())
    }

    fn sin_cos(self) -> (Self, Self) {
        static COEFFICIENTS: LazyLock<Coefficients<Dd>> = LazyLock::new(Coefficients::new);
        trig::sin_cos(self, &COEFFICIENTS)
    }

    #[inline]
    fn tau() -> Self {
        Dd(twofloat::consts::TAU)
    }
}

impl_high_precision_newtype!(Dd, |value| value);
