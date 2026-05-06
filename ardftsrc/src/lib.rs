#![doc = include_str!("../../README.md")]

mod batch;
mod chunk;
mod config;
mod core;
mod error;
mod lpc;
mod streaming;
mod vec_of_vecs;

pub use vec_of_vecs::PlanarVecs;
pub use batch::{BatchResampler};
pub use chunk::ChunkResampler;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use streaming::StreamingResampler;

// Re-export audioadapter and audioadapter_buffers for convenience.
pub use audioadapter;
pub use audioadapter_buffers;