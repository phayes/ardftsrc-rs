//! Vendored from the `primal-check` crate (MIT OR Apache-2.0): see `f128_fft::vendor` module docs
//! for why -- it's a tiny crate providing just Miller-Rabin primality testing (used by Rader's
//! algorithm), not worth taking on as an external dependency.

pub(crate) use is_prime::miller_rabin;
#[allow(unused_imports)]
pub(crate) use perfect_power::{as_perfect_power, as_prime_power};

mod perfect_power;
mod is_prime;
