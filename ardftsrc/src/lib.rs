mod ardftsrc;
mod ardftsrc_core;
mod config;
mod error;
mod lpc;

use audioadapter::Adapter;

pub use ardftsrc::Ardftsrc;
pub use config::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
pub use error::Error;

pub(crate) use ardftsrc_core::ArdftsrcCore;
pub(crate) use config::DerivedConfig;

/// Utility function that copies an adapter into a new interleaved `Vec`.
///
/// Samples are copied frame-by-frame with [`Adapter::copy_from_frame_to_slice`]. The returned
/// vector is sized for `adapter.frames() * adapter.channels()` and truncated if the adapter reports
/// fewer copied samples.
#[must_use]
pub fn adapter_to_interleaved_vec<'a, T>(adapter: &dyn Adapter<'a, T>) -> Vec<T>
where
    T: Clone + Default + 'a,
{
    let mut output = vec![T::default(); adapter.frames() * adapter.channels()];
    let written = adapter_to_interleaved_slice(adapter, &mut output)
        .expect("adapter_to_interleaved_vec allocates enough output samples");
    output.truncate(written);
    output
}

/// Utility function that copies an adapter into an interleaved slice.
///
/// Returns the number of samples copied, or an error if `output` is too small.
pub fn adapter_to_interleaved_slice<'a, T: 'a>(adapter: &dyn Adapter<'a, T>, output: &mut [T]) -> Result<usize, Error> {
    let channels = adapter.channels();
    let required = channels * adapter.frames();
    if output.len() < required {
        return Err(Error::InsufficientOutputBuffer {
            expected: required,
            actual: output.len(),
        });
    }

    let mut written = 0;
    for frame_idx in 0..adapter.frames() {
        written += adapter.copy_from_frame_to_slice(frame_idx, 0, &mut output[written..]);
    }

    Ok(written)
}
