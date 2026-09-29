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

impl_high_precision_newtype!(Dd, |value| value, div = div);

/// Double-double division by Joldes et al. (2017) Algorithm 17 (relative error below `15u^2`).
///
/// `twofloat`'s own `TwoFloat / TwoFloat` (Algorithm 18) forms the reciprocal residual
/// `1 - y.hi * (1 / y.hi)` with a plain `f64` multiply rather than an FMA, so the residual
/// usually rounds to zero and the quotient is only `f64`-accurate (e.g. `1 / 3` has a zero low
/// word). This version relies only on `twofloat`'s `TwoFloat * f64` (Algorithm 9), which is exact
/// to double-double precision.
fn div(x: TwoFloat, y: TwoFloat) -> TwoFloat {
    let th = x.hi() / y.hi();
    let r = y * th;
    let pi_h = x.hi() - r.hi();
    let delta_l = x.lo() - r.lo();
    let delta = pi_h + delta_l;
    let tl = delta / y.hi();
    TwoFloat::new_add(th, tl)
}
