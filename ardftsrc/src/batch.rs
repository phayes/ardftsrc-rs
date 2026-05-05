use num_traits::Float;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use realfft::FftNum;

use crate::{ChunkResampler, Config, Error};
use audio_core::Sample;
use audioadapter::{Adapter, AdapterMut};

#[derive(Clone, Debug, PartialEq)]
pub struct SequentialVecOfVecs<T> {
    buf: Vec<Vec<T>>,
    frames: usize,
}

pub struct SequentialVecOfVecsIter<'a, T> {
    inner: std::slice::Iter<'a, Vec<T>>,
}

pub struct SequentialVecOfVecsIterMut<'a, T> {
    inner: std::slice::IterMut<'a, Vec<T>>,
}

impl<T> SequentialVecOfVecs<T> {
    pub fn new(buf: Vec<Vec<T>>) -> Result<Self, Error> {
        let frames = buf.first().map_or(0, Vec::len);
        if let Some(channel) = buf.iter().find(|channel| channel.len() != frames) {
            return Err(Error::WrongFrameCount {
                expected: frames,
                actual: channel.len(),
            });
        }

        Ok(Self { buf, frames })
    }

    #[must_use]
    #[inline]
    pub fn channels(&self) -> usize {
        self.buf.len()
    }

    #[must_use]
    #[inline]
    pub fn frames(&self) -> usize {
        self.frames
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() || self.frames == 0
    }

    #[must_use]
    #[inline]
    pub fn get_channel(&self, index: usize) -> Option<&[T]> {
        self.buf.get(index).map(Vec::as_slice)
    }

    #[must_use]
    #[inline]
    pub fn get_channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.buf.get_mut(index).map(Vec::as_mut_slice)
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[Vec<T>] {
        &self.buf
    }

    #[must_use]
    #[inline]
    pub fn into_inner(self) -> Vec<Vec<T>> {
        self.buf
    }

    #[must_use]
    #[inline]
    pub fn pop_channel(&mut self) -> Option<Vec<T>> {
        let channel = self.buf.pop();
        if self.buf.is_empty() {
            self.frames = 0;
        }
        channel
    }
}

impl<'a, T> Iterator for SequentialVecOfVecsIter<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Vec::as_slice)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for SequentialVecOfVecsIter<'_, T> {}

impl<'a, T> Iterator for SequentialVecOfVecsIterMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Vec::as_mut_slice)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for SequentialVecOfVecsIterMut<'_, T> {}

impl<T> From<SequentialVecOfVecs<T>> for Vec<Vec<T>> {
    #[inline]
    fn from(value: SequentialVecOfVecs<T>) -> Self {
        value.into_inner()
    }
}

impl<T> IntoIterator for SequentialVecOfVecs<T> {
    type Item = Vec<T>;
    type IntoIter = std::vec::IntoIter<Vec<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a SequentialVecOfVecs<T> {
    type Item = &'a [T];
    type IntoIter = SequentialVecOfVecsIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SequentialVecOfVecsIter { inner: self.buf.iter() }
    }
}

impl<'a, T> IntoIterator for &'a mut SequentialVecOfVecs<T> {
    type Item = &'a mut [T];
    type IntoIter = SequentialVecOfVecsIterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SequentialVecOfVecsIterMut {
            inner: self.buf.iter_mut(),
        }
    }
}

unsafe impl<'a, T> Adapter<'a, T> for SequentialVecOfVecs<T>
where
    T: Clone + 'a,
{
    #[inline]
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> T {
        unsafe { self.buf.get_unchecked(channel).get_unchecked(frame).clone() }
    }

    #[inline]
    fn channels(&self) -> usize {
        self.channels()
    }

    #[inline]
    fn frames(&self) -> usize {
        self.frames()
    }

    fn copy_from_channel_to_slice(&self, channel: usize, skip: usize, slice: &mut [T]) -> usize {
        if channel >= self.channels() || skip >= self.frames {
            return 0;
        }

        let frames_to_write = (self.frames - skip).min(slice.len());
        slice[..frames_to_write].clone_from_slice(&self.buf[channel][skip..skip + frames_to_write]);
        frames_to_write
    }
}

unsafe impl<'a, T> AdapterMut<'a, T> for SequentialVecOfVecs<T>
where
    T: Clone + 'a,
{
    unsafe fn write_sample_unchecked(&mut self, channel: usize, frame: usize, value: &T) -> bool {
        unsafe {
            *self.buf.get_unchecked_mut(channel).get_unchecked_mut(frame) = value.clone();
        }
        false
    }

    fn copy_from_slice_to_channel(&mut self, channel: usize, skip: usize, slice: &[T]) -> (usize, usize) {
        if channel >= self.channels() || skip >= self.frames {
            return (0, 0);
        }

        let frames_to_read = (self.frames - skip).min(slice.len());
        self.buf[channel][skip..skip + frames_to_read].clone_from_slice(&slice[..frames_to_read]);
        (frames_to_read, 0)
    }
}

pub struct BatchResampler<T = f64>
where
    T: Float + FftNum + Sample,
{
    inner: ChunkResampler<T>,
}

impl<T> BatchResampler<T>
where
    T: Float + FftNum + Sample,
{
    /// Constructs a batch resampler from `config`.
    pub fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            inner: ChunkResampler::new(config)?,
        })
    }

    /// Process multiple independent tracks.
    ///
    /// Each input slice is treated as its own stream with no inter-track context. See
    /// `batch_gapless()` for gapless processing of multiple tracks.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch<'a>(&self, inputs: &[&dyn Adapter<'a, T>]) -> Result<Vec<SequentialVecOfVecs<T>>, Error>
    where
        T: Send + Sync,
    {
        let prepared_inputs = Self::prepare_generic_adapter_inputs(self.inner.config(), inputs)?;
        self.batch_planar(prepared_inputs)
    }

    /// Process multiple tracks as one gapless sequence.
    ///
    /// Adjacent inputs are treated as tracks from the same album or other back-to-back
    /// material. Each track is returned separately, but the previous track's tail and next
    /// track's head are used as edge context to improve gapless playback.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch_gapless<'a>(&self, inputs: &[&dyn Adapter<'a, T>]) -> Result<Vec<SequentialVecOfVecs<T>>, Error>
    where
        T: Send + Sync,
    {
        let prepared_inputs = Self::prepare_generic_adapter_inputs(self.inner.config(), inputs)?;
        self.batch_planar_gapless(prepared_inputs)
    }

    /// Process multiple independent tracks. This is a planar specialization of `batch()`.
    ///
    /// Each input slice is treated as its own stream with no inter-track context. See
    /// `batch_gapless()` for gapless processing of multiple tracks.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch_planar(&self, inputs: Vec<SequentialVecOfVecs<T>>) -> Result<Vec<SequentialVecOfVecs<T>>, Error>
    where
        T: Send + Sync,
    {
        let config = self.inner.config().clone();

        #[cfg(feature = "rayon")]
        {
            inputs
                .into_par_iter()
                .map(|input| {
                    let channel_outputs = Self::process_track(config.clone(), input.as_slice(), None, None)?;
                    SequentialVecOfVecs::new(channel_outputs)
                })
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .into_iter()
                .map(|input| {
                    let channel_outputs = Self::process_track(config.clone(), input.as_slice(), None, None)?;
                    SequentialVecOfVecs::new(channel_outputs)
                })
                .collect()
        }
    }

    /// Process multiple tracks as one gapless sequence. This is a planar specialization of `batch_gapless()`.
    ///
    /// Adjacent inputs are treated as tracks from the same album or other back-to-back
    /// material. Each track is returned separately, but the previous track's tail and next
    /// track's head are used as edge context to improve gapless playback.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch_planar_gapless(
        &self,
        inputs: Vec<SequentialVecOfVecs<T>>,
    ) -> Result<Vec<SequentialVecOfVecs<T>>, Error>
    where
        T: Send + Sync,
    {
        let config = self.inner.config().clone();
        let context_chunk_size = self.inner.input_chunk_size() / config.channels;

        for input in &inputs {
            Self::validate_track(&config, input.as_slice())?;
        }

        #[cfg(feature = "rayon")]
        {
            inputs
                .par_iter()
                .enumerate()
                .map(|(track_idx, input)| {
                    let pre = track_idx
                        .checked_sub(1)
                        .map(|idx| Self::track_tail_context(inputs[idx].as_slice(), context_chunk_size));
                    let post = inputs
                        .get(track_idx + 1)
                        .map(|input| Self::track_head_context(input.as_slice(), context_chunk_size));
                    let channel_outputs = Self::process_track(config.clone(), input.as_slice(), pre, post)?;
                    SequentialVecOfVecs::new(channel_outputs)
                })
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .iter()
                .enumerate()
                .map(|(track_idx, input)| {
                    let pre = track_idx
                        .checked_sub(1)
                        .map(|idx| Self::track_tail_context(inputs[idx].as_slice(), context_chunk_size));
                    let post = inputs
                        .get(track_idx + 1)
                        .map(|input| Self::track_head_context(input.as_slice(), context_chunk_size));
                    let channel_outputs = Self::process_track(config.clone(), input.as_slice(), pre, post)?;
                    SequentialVecOfVecs::new(channel_outputs)
                })
                .collect()
        }
    }

    fn prepare_generic_adapter_inputs<'a>(
        config: &Config,
        inputs: &[&dyn Adapter<'a, T>],
    ) -> Result<Vec<SequentialVecOfVecs<T>>, Error> {
        inputs
            .iter()
            .map(|input| {
                if input.channels() != config.channels {
                    return Err(Error::WrongChannelCount {
                        expected: config.channels,
                        actual: input.channels(),
                    });
                }

                let frames = input.frames();
                let mut per_channel = Vec::with_capacity(config.channels);
                for channel_idx in 0..config.channels {
                    let mut channel = vec![T::zero(); frames];
                    let copied = input.copy_from_channel_to_slice(channel_idx, 0, &mut channel);
                    if copied != frames {
                        return Err(Error::WrongFrameCount {
                            expected: frames,
                            actual: copied,
                        });
                    }
                    per_channel.push(channel);
                }
                SequentialVecOfVecs::new(per_channel)
            })
            .collect()
    }

    fn process_track(
        config: Config,
        channel_inputs: &[Vec<T>],
        pre: Option<Vec<Vec<T>>>,
        post: Option<Vec<Vec<T>>>,
    ) -> Result<Vec<Vec<T>>, Error>
    where
        T: Send + Sync,
    {
        Self::validate_track(&config, channel_inputs)?;

        let mut resampler = ChunkResampler::new(config)?;

        if let Some(pre) = pre {
            for (core, channel_pre) in resampler.cores.iter_mut().zip(pre) {
                core.pre(channel_pre);
            }
        }

        if let Some(post) = post {
            for (core, channel_post) in resampler.cores.iter_mut().zip(post) {
                core.post(channel_post);
            }
        }

        Self::process_cores(&mut resampler, channel_inputs)
    }

    fn track_tail_context(channel_inputs: &[Vec<T>], context_chunk_size: usize) -> Vec<Vec<T>> {
        channel_inputs
            .iter()
            .map(|channel| {
                let start = channel.len().saturating_sub(context_chunk_size);
                channel[start..].to_vec()
            })
            .collect()
    }

    fn track_head_context(channel_inputs: &[Vec<T>], context_chunk_size: usize) -> Vec<Vec<T>> {
        channel_inputs
            .iter()
            .map(|channel| {
                let end = channel.len().min(context_chunk_size);
                channel[..end].to_vec()
            })
            .collect()
    }

    fn validate_track(config: &Config, channel_inputs: &[Vec<T>]) -> Result<(), Error> {
        if channel_inputs.len() != config.channels {
            return Err(Error::WrongChannelCount {
                expected: config.channels,
                actual: channel_inputs.len(),
            });
        }

        let frames = channel_inputs.first().map_or(0, Vec::len);
        if let Some(channel) = channel_inputs.iter().find(|channel| channel.len() != frames) {
            return Err(Error::WrongFrameCount {
                expected: frames,
                actual: channel.len(),
            });
        }

        Ok(())
    }

    fn process_cores(resampler: &mut ChunkResampler<T>, channel_inputs: &[Vec<T>]) -> Result<Vec<Vec<T>>, Error>
    where
        T: Send + Sync,
    {
        #[cfg(feature = "rayon")]
        {
            resampler
                .cores
                .par_iter_mut()
                .zip(channel_inputs.par_iter())
                .map(|(core, channel_input)| core.process_all(channel_input))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            resampler
                .cores
                .iter_mut()
                .zip(channel_inputs.iter())
                .map(|(core, channel_input)| core.process_all(channel_input))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaperType;
    use audioadapter_buffers::direct::InterleavedSlice;

    fn mono_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config {
            input_sample_rate,
            output_sample_rate,
            channels: 1,
            quality: 64,
            bandwidth: 0.95,
            taper_type: TaperType::Cosine(3.45),
        }
    }

    #[test]
    fn batch_planar_gapless_matches_manual_pre_post() {
        let config = mono_config(44_100, 48_000);
        let batch = BatchResampler::new(config.clone()).unwrap();
        let context_chunk_size = ChunkResampler::<f32>::new(config.clone()).unwrap().input_chunk_size();
        let track_frames = context_chunk_size * 2 + 17;

        let tracks: Vec<SequentialVecOfVecs<f32>> = (0..3)
            .map(|track_idx| {
                let channel = (0..track_frames)
                    .map(|frame| {
                        let continuous_frame = track_idx * track_frames + frame;
                        (continuous_frame as f32 * 0.017).sin() * 0.25
                    })
                    .collect();
                SequentialVecOfVecs::new(vec![channel]).unwrap()
            })
            .collect();

        let outputs = batch.batch_planar_gapless(tracks.clone()).unwrap();

        for (track_idx, (input, output)) in tracks.iter().zip(outputs.iter()).enumerate() {
            let mut resampler = ChunkResampler::new(config.clone()).unwrap();
            if let Some(previous) = track_idx.checked_sub(1).map(|idx| tracks[idx].get_channel(0).unwrap()) {
                resampler
                    .pre(&previous[previous.len().saturating_sub(context_chunk_size)..])
                    .unwrap();
            }
            if let Some(next) = tracks.get(track_idx + 1).map(|track| track.get_channel(0).unwrap()) {
                resampler.post(&next[..next.len().min(context_chunk_size)]).unwrap();
            }

            let expected = resampler.process_all(input.get_channel(0).unwrap()).unwrap();
            assert_eq!(output.get_channel(0).unwrap().len(), expected.len());
            for (actual, expected) in output.get_channel(0).unwrap().iter().zip(expected.iter()) {
                assert!((*actual - *expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn batch_gapless_matches_planar_gapless() {
        let config = mono_config(44_100, 48_000);
        let batch = BatchResampler::new(config).unwrap();
        let context_chunk_size = batch.inner.input_chunk_size();
        let track_frames = context_chunk_size * 2 + 17;

        let tracks: Vec<SequentialVecOfVecs<f32>> = (0..3)
            .map(|track_idx| {
                let channel = (0..track_frames)
                    .map(|frame| {
                        let continuous_frame = track_idx * track_frames + frame;
                        (continuous_frame as f32 * 0.017).sin() * 0.25
                    })
                    .collect();
                SequentialVecOfVecs::new(vec![channel]).unwrap()
            })
            .collect();
        let adapter_inputs: Vec<&dyn Adapter<'_, f32>> =
            tracks.iter().map(|track| track as &dyn Adapter<'_, f32>).collect();

        let adapter_outputs = batch.batch_gapless(&adapter_inputs).unwrap();
        let planar_outputs = batch.batch_planar_gapless(tracks.clone()).unwrap();

        assert_eq!(adapter_outputs.len(), planar_outputs.len());
        for (adapter_output, planar_output) in adapter_outputs.iter().zip(planar_outputs.iter()) {
            assert_eq!(adapter_output, planar_output);
        }
    }

    #[test]
    fn batch_test() {
        let config = mono_config(44_100, 48_000);
        let batch_driver = BatchResampler::new(config.clone()).unwrap();
        let chunk = batch_driver.inner.input_chunk_size();
        let tracks: Vec<Vec<f32>> = vec![
            (0..(chunk + 17))
                .map(|frame| (frame as f32 * 0.009).sin() * 0.2)
                .collect(),
            (0..(chunk * 2 + 5))
                .map(|frame| (frame as f32 * 0.012).cos() * 0.15)
                .collect(),
            (0..(chunk / 2 + 11))
                .map(|frame| (frame as f32 * 0.021).sin() * 0.25)
                .collect(),
        ];
        let input_refs: Vec<&[f32]> = tracks.iter().map(Vec::as_slice).collect();
        let input_adapters = input_refs
            .iter()
            .map(|input| InterleavedSlice::new(input, 1, input.len()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let input_adapter_refs = input_adapters
            .iter()
            .map(|input| input as &dyn Adapter<'_, f32>)
            .collect::<Vec<_>>();

        let expected: Vec<Vec<f32>> = tracks
            .iter()
            .map(|track| {
                let mut resampler = ChunkResampler::new(config.clone()).unwrap();
                resampler.process_all(track).unwrap()
            })
            .collect();
        let actual = batch_driver.batch(&input_adapter_refs).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_track, expected_track) in actual.iter().zip(expected.iter()) {
            let actual_interleaved = crate::adapter_to_interleaved_vec(actual_track);
            assert_eq!(actual_interleaved.len(), expected_track.len());
            for (left, right) in actual_interleaved.iter().zip(expected_track.iter()) {
                assert!((*left - *right).abs() < 1e-5);
            }
        }
    }
}
