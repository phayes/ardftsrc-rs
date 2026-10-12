//! High-precision sample types and real FFT engines, compiled by the `high_precision` feature.
//!
//! `ardftsrc` runs the whole resampling pipeline in the resampler's sample type. With `f32` and
//! `f64`, its FFTs use `realfft` (backed by `rustfft`), which computes twiddle factors in plain
//! `f64`. This module provides wider sample types for callers who want to push past that
//! ceiling. Their FFTs run the same algorithms (mixed-radix, Bluestein's, Rader's -- vendored
//! from `rustfft`, see `vendor` for details and rationale) with every twiddle factor and
//! internal accumulation computed in the wide type:
//!
//! - [`DoubleDouble`]: double-double (~106-bit mantissa), backed by `twofloat`.
//! - [`F128`]: IEEE binary128 (113-bit mantissa), backed by `rustc_apfloat`.
//! - [`F256`]: IEEE binary256 (237-bit mantissa), backed by `f256`.
//!
//! None of them are general replacements: there's no SIMD, and every type is at least an order
//! of magnitude slower than `f64` per operation (the binary128 and binary256 types are pure
//! software floats), so they are meant for offline / opt-in use at extreme quality settings, not
//! realtime streaming.

mod numeric;
mod real;
#[allow(unused)]
mod vendor;

pub use numeric::{DoubleDouble, F128, F256};
