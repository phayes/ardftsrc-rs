#![doc = include_str!("../README.md")]

mod config;
mod core;
mod error;
mod interleaved_resampler;
mod lpc;
mod planar_resampler;
mod planar_vecs;
mod realtime;

pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use interleaved_resampler::InterleavedResampler;
pub use planar_resampler::PlanarResampler;
pub use planar_vecs::PlanarVecs;
pub use realtime::RealtimeResampler;

#[cfg(test)]
mod test_utils;

// feature: rodio
#[cfg(feature = "rodio")]
mod rodio;

#[cfg(feature = "rodio")]
pub use rodio::RodioResampler;

// feature: audioadapter
#[cfg(feature = "audioadapter")]
mod adapter_resampler;

#[cfg(feature = "audioadapter")]
pub use adapter_resampler::AdapterResampler;

// Re-export audioadapter for convenience.
#[cfg(feature = "audioadapter")]
pub use audioadapter;

#[track_caller]
pub(crate) fn panic_msg(msg: &str) -> ! {
    panic!(
        "ardftsrc: {}. This is a bug in the ardfsrc crate. Please file a bug report at https://github.com/phayes/ardftsrc-rs/issues",
        msg
    );
}

/// Like [`panic_msg`], but appends an underlying [`std::error::Error`] after `context` (via [`Display`](std::fmt::Display)).
#[track_caller]
pub(crate) fn panic_err(context: &str, err: impl std::error::Error) -> ! {
    panic!(
        "ardftsrc: {}: {}. This is a bug in the ardfsrc crate. Please file a bug report at https://github.com/phayes/ardftsrc-rs/issues",
        context, err
    );
}
