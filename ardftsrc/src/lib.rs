#![doc = include_str!("../../README.md")]

mod adapter_resampler;
mod interleaved_resampler;
mod planar_resampler;
mod config;
mod core;
mod error;
mod lpc;
mod planar_vecs;
mod streaming;
#[cfg(test)]
mod test_utils;

pub use adapter_resampler::AdapterResampler;
pub use interleaved_resampler::InterleavedResampler;
pub use planar_resampler::PlanarResampler;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use planar_vecs::PlanarVecs;
pub use streaming::StreamingResampler;

// Re-export audioadapter and audioadapter_buffers for convenience.
pub use audioadapter;
pub use audioadapter_buffers;
