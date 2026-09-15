//! Scalar-only FFT engine vendored and trimmed from `rustfft` 6.4.1 (MIT OR Apache-2.0). See
//! `high_precision::vendor` module docs for why this is vendored rather than depended on.
//!
//! ## What's kept vs. dropped from upstream
//!
//! Kept essentially as-is: the mixed-radix / mixed-radix-small / Good-Thomas / Rader's /
//! Bluestein's algorithm selection and implementations (`plan.rs`, `algorithm/*.rs`), which
//! already handle arbitrary (non-power-of-two) FFT sizes correctly and are the hard part to get
//! right. Dropped: the AVX/SSE/NEON/WASM SIMD backends and the `FftPlanner` dispatch wrapper
//! around them (`FftPlannerScalar` is used directly instead) -- high-precision arithmetic isn't
//! a hardware-vectorizable primitive, so there's nothing for those to accelerate here. Also dropped:
//! upstream's own `#[cfg(test)]` unit tests, which depend on upstream's internal `test_utils`
//! module (needs `rand`) that wasn't worth vendoring just for that -- correctness here is instead
//! covered by `high_precision::real`'s tests, which exercise this engine through the real-FFT layer.
//!
//! Two other changes from upstream, both isolated and mechanical:
//! - `common::FftNum` gained a `twiddle()` method in place of the original blanket impl; concrete
//!   impls are provided for `f32`/`f64` (matching upstream's behavior exactly, kept only so this
//!   vendored engine can be cross-checked against real `rustfft` output at matching precision) and
//!   for each high-precision type in `high_precision::numeric` (at that type's full precision).
//!   See `common.rs` for the full rationale.
//! - The two hardcoded `sqrt(1/2)` butterfly constants in `algorithm/butterflies.rs` now go
//!   through `compute_twiddle` (`cos(pi/4) == sqrt(1/2)`) instead of `T::from_f64`, for the same
//!   reason.
//!
//! `unsafe_op_in_unsafe_fn` is allowed crate-wide for this subtree: upstream predates edition
//! 2024's tightening of that lint (unsafe fn bodies are no longer an implicit unsafe block), and
//! retrofitting explicit `unsafe {}` blocks throughout vendored code isn't worth the diff noise
//! for code we want to stay a close, diffable match to upstream.
#![allow(unsafe_op_in_unsafe_fn)]

#[macro_use]
pub(crate) mod common;

pub(crate) mod algorithm;
pub(crate) mod array_utils;
pub(crate) mod fft_cache;
pub(crate) mod fft_helper;
pub(crate) mod math_utils;
pub(crate) mod plan;
pub(crate) mod twiddles;

pub(crate) use common::FftNum;
pub(crate) use num_complex::Complex;
pub(crate) use plan::FftPlannerScalar;

use num_traits::Zero;

/// A trait that allows FFT algorithms to report their expected input/output size.
pub(crate) trait Length {
    fn len(&self) -> usize;
}

/// Represents a FFT direction, IE a forward FFT or an inverse FFT.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum FftDirection {
    Forward,
    Inverse,
}

impl FftDirection {
    #[inline]
    #[allow(dead_code)]
    pub fn opposite_direction(&self) -> FftDirection {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward,
        }
    }
}

impl std::fmt::Display for FftDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward => f.write_str("Forward"),
            Self::Inverse => f.write_str("Inverse"),
        }
    }
}

/// A trait that allows FFT algorithms to report whether they compute forward or inverse FFTs.
pub(crate) trait Direction {
    fn fft_direction(&self) -> FftDirection;
}

/// Trait for algorithms that compute FFTs. See `rustfft::Fft` for full documentation of the
/// contract each method must uphold -- unchanged from upstream here.
pub(crate) trait Fft<T: FftNum>: Length + Direction + Sync + Send {
    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.get_inplace_scratch_len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    );

    fn get_inplace_scratch_len(&self) -> usize;
    fn get_outofplace_scratch_len(&self) -> usize;
    fn get_immutable_scratch_len(&self) -> usize;
}
