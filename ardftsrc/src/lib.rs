#![doc = include_str!("../../README.md")]

mod interleaved_resampler;
mod planar_resampler;
mod config;
mod core;
mod error;
mod lpc;
mod planar_vecs;
mod streaming;

pub use interleaved_resampler::InterleavedResampler;
pub use planar_resampler::PlanarResampler;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use planar_vecs::PlanarVecs;
pub use streaming::StreamingResampler;

#[cfg(test)]
mod test_utils;

// Audioadapter

#[cfg(feature = "audioadapter")]
mod adapter_resampler;

#[cfg(feature = "audioadapter")]
pub use adapter_resampler::AdapterResampler;

// Re-export audioadapter for convenience.
#[cfg(feature = "audioadapter")]
pub use audioadapter;
