#![doc = include_str!("../README.md")]

mod config;
mod core;
mod error;
mod interleaved_resampler;
mod lpc;
mod offthread;
mod planar_resampler;
mod planar_vecs;
mod realtime;

pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use interleaved_resampler::InterleavedResampler;
pub use planar_resampler::PlanarResampler;
pub use planar_vecs::PlanarVecs;
pub use realtime::{StreamingConfig, RealtimeResampler};


#[cfg(feature = "rodio")]
pub mod rodio;

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
