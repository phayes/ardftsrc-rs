use audio_core::Sample;
use audioadapter_buffers::direct::InterleavedSlice;
use num_traits::Float;
use realfft::FftNum;

use crate::{AdapterResampler, Error};

pub(crate) fn assert_no_nans<T>(samples: &[T], label: &str)
where
    T: Float,
{
    if let Some((idx, _)) = samples.iter().copied().enumerate().find(|(_, s)| s.is_nan()) {
        panic!("{label} contains NaN at index {idx}");
    }
}

pub(crate) fn process_all_samples<T>(resampler: &mut AdapterResampler<T>, input: &[T]) -> Result<Vec<T>, Error>
where
    T: Float + FftNum + Sample + Send + Sync,
{
    let channels = resampler.config().channels;
    let input_adapter =
        InterleavedSlice::new(input, channels, input.len() / channels).map_err(|_| Error::MalformedInputLength {
            channels,
            samples: input.len(),
        })?;
    let output = resampler.process_all(&input_adapter)?.interleave();
    assert_no_nans(&output, "test_utils::process_all_samples output");
    Ok(output)
}
