//! Quad-precision real FFT engine, compiled by the `f128` feature and selected with
//! [`Config::f128`](crate::Config::f128) or
//! [`Config::with_f128(true)`](crate::Config::with_f128).
//!
//! `ardftsrc`'s default FFT backend (`realfft`, backed by `rustfft`) computes twiddle factors in
//! plain `f64`. This module is an alternative backend for callers who want to push past that ceiling:
//! it runs the same FFT algorithms (mixed-radix, Bluestein's, Rader's -- vendored from `rustfft`, see `vendor` for
//! details and rationale) but with every twiddle factor and internal accumulation computed in
//! `f128` (113-bit mantissa, vs. 53 for `f64`).
//!
//! It is not a general replacement: there's no SIMD, and `f128` arithmetic costs roughly one to
//! two orders of magnitude more than `f64` per operation, so this is meant for offline / opt-in
//! use at extreme quality settings, not realtime streaming.

mod numeric;
mod real;
#[allow(unused)]
mod vendor;

// With all this vendoring, these are the only functions we actually need to expose.
pub(crate) use real::{plan_fft_forward, plan_fft_inverse};
