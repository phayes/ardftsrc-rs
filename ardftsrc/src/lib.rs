#![doc = include_str!("../../README.md")]

mod batch;
mod chunk;
mod config;
mod core;
mod error;
mod lpc;
mod streaming;

pub use batch::{BatchResampler, SequentialVecOfVecs};
pub use chunk::ChunkResampler;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;
pub use streaming::StreamingResampler;

// Re-export audioadapter and audioadapter_buffers for convenience.
pub use audioadapter;
pub use audioadapter_buffers;