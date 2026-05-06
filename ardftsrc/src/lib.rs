#![doc = include_str!("../../README.md")]

mod batch;
mod chunk_adapter;
mod chunk_interleaved;
mod chunk_planar;
mod config;
mod core;
mod error;
mod lpc;
mod planar_vecs;
mod streaming;
#[cfg(test)]
mod test_utils;

pub use batch::BatchResampler;
pub use chunk_adapter::ChunkAdapterResampler;
pub use chunk_interleaved::ChunkInterleavedResampler;
pub use chunk_planar::ChunkPlanarResampler;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use planar_vecs::PlanarVecs;
pub use streaming::StreamingResampler;

// Re-export audioadapter and audioadapter_buffers for convenience.
pub use audioadapter;
pub use audioadapter_buffers;
