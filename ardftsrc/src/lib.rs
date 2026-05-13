#![doc = include_str!("../README.md")]

mod config;
mod core;
mod error;
mod interleaved_resampler;
mod lpc;
mod planar_resampler;
mod planar_vecs;

pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use interleaved_resampler::InterleavedResampler;
pub use planar_resampler::PlanarResampler;
pub use planar_vecs::PlanarVecs;

#[cfg(test)]
mod test_utils;

// feature: realtime
#[cfg(feature = "realtime")]
mod realtime;

//#[cfg(feature = "realtime")]
//mod offthread;

#[cfg(feature = "realtime")]
pub use realtime::RealtimeResampler;

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
