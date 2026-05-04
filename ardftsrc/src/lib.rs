mod ardftsrc;
mod ardftsrc_core;
mod config;
mod error;
mod lpc;

use audioadapter::Adapter;

pub use ardftsrc::Ardftsrc;
pub use config::{Config, DerivedConfig, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;

pub(crate) use ardftsrc_core::ArdftsrcCore;

pub fn adapter_to_interleaved<'a, T: 'a>(adapter: &dyn Adapter<'a, T>) -> Vec<T> {
    let channels = adapter.channels();
    let frames = adapter.frames();
    let mut interleaved = Vec::with_capacity(channels * frames);
    for frame in 0..frames {
        for channel in 0..channels {
            let sample = adapter
                .read_sample(channel, frame)
                .expect("adapter bounds checked by frame/channel loops");
            interleaved.push(sample);
        }
    }
    interleaved
}
