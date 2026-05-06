#![doc = include_str!("../../README.md")]

mod batch;
mod chunk_adapter;
mod config;
mod core;
mod error;
mod lpc;
mod chunk_interleaved;
mod chunk_planar;
mod streaming;
#[cfg(test)]
mod test_utils;
mod vec_of_vecs;

pub use batch::BatchResampler;
pub use chunk_adapter::ChunkAdapterResampler;
pub use chunk_interleaved::ChunkInterleavedResampler;
pub use chunk_planar::ChunkPlanarResampler;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use streaming::StreamingResampler;
pub use vec_of_vecs::PlanarVecs;

// Re-export audioadapter and audioadapter_buffers for convenience.
pub use audioadapter;
pub use audioadapter_buffers;
