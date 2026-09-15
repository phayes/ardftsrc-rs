//! Vendored from the `transpose` crate 0.2.3 (MIT OR Apache-2.0): see `high_precision::vendor` module docs
//! for why. rustfft uses this for cache-friendly out-of-place matrix transposes in mixed-radix
//! and Good–Thomas. Upstream also ships an in-place path (`transpose_inplace`); it is dropped
//! here because nothing in this engine calls it.
//!
//! Upstream is `#![no_std]`; that attribute is crate-root-only so it's dropped here, but nothing
//! below actually needs `std` (only `core`). `unsafe_op_in_unsafe_fn` is allowed for the same
//! reason as the vendored `rustfft` subtree: upstream predates edition 2024's tightening of that
//! lint.
#![allow(unsafe_op_in_unsafe_fn)]

mod out_of_place;
pub(crate) use out_of_place::transpose;
