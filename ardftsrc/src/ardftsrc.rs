use num_traits::Float;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use realfft::FftNum;

use crate::{ArdftsrcCore, Config, DerivedConfig, Error};
use audio_core::Sample;
use audioadapter::{Adapter, AdapterMut};
use audioadapter_buffers::direct::InterleavedSlice;
use std::collections::VecDeque;

pub struct Ardftsrc<T = f32>
where
    T: Float + FftNum + Sample,
{
    config: Config,
    derived: DerivedConfig<T>,
    cores: Vec<ArdftsrcCore<T>>,

    // Staging areas.
    // TODO: This can be removed by using Core::mut_input_ref() and writing samples directly
    input_staging: Vec<Vec<T>>,

    // Samples API state, interleaved.
    sample_pending_input: VecDeque<T>,
    samples_pending_output: VecDeque<T>,
    samples_input_chunk_buffer: Vec<T>,
    samples_output_chunk_buffer: Vec<T>,
    samples_finalized: bool,
}

impl<T> Ardftsrc<T>
where
    T: Float + FftNum + Sample,
{
    /// Constructs a resampler from `config`.
    ///
    /// Returns a ready-to-use resampler instance, or an error if `config` is invalid or derived
    /// FFT geometry cannot be prepared.
    pub fn new(config: Config) -> Result<Self, Error> {
        let derived = config.derive_config::<T>()?;
        let cores = (0..config.channels)
            .map(|_| ArdftsrcCore::new(derived.clone()))
            .collect();

        let input_chunk_size = derived.input_chunk_frames * config.channels;
        let output_chunk_size = derived.output_chunk_frames * config.channels;

        let input_staging: Vec<Vec<T>> = vec![vec![T::zero(); derived.input_chunk_frames]; config.channels];

        Ok(Self {
            config,
            derived,
            cores,
            input_staging,
            sample_pending_input: VecDeque::with_capacity(input_chunk_size * 2), // Slightly oversized to avoid reallocations.
            samples_pending_output: VecDeque::with_capacity(output_chunk_size * 2), // Slightly oversized to avoid reallocations.
            samples_input_chunk_buffer: Vec::with_capacity(input_chunk_size),
            samples_output_chunk_buffer: vec![T::zero(); output_chunk_size],
            samples_finalized: false,
        })
    }

    /// Returns the configuration this instance was built with.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the total number of interleaved input samples processed.
    pub fn input_sample_processed(&self) -> usize {
        self.cores.iter().map(ArdftsrcCore::input_sample_processed).sum()
    }

    /// Returns the total number of interleaved output samples processed.
    pub fn output_sample_processed(&self) -> usize {
        self.cores.iter().map(ArdftsrcCore::output_sample_processed).sum()
    }

    /// Returns the required `input` length (interleaved samples) for each `process_chunk()` call.
    ///
    /// Use this to allocate/read fixed-size streaming input buffers.
    #[must_use]
    #[inline]
    pub fn input_chunk_size(&self) -> usize {
        self.derived.input_chunk_frames * self.config.channels
    }

    /// Returns the recommended per-call `output` capacity in interleaved samples.
    ///
    /// For chunked streaming, size output slices passed to `process_chunk()` and
    /// `process_chunk_final()` to at least this value.
    #[must_use]
    pub fn output_chunk_size(&self) -> usize {
        self.derived.output_chunk_frames * self.config.channels
    }

    /// Returns algorithmic latency to trim/flush, or zero in same-rate passthrough mode.
    #[must_use]
    pub fn output_delay_frames(&self) -> usize {
        if self.is_passthrough() {
            0
        } else {
            self.derived.output_offset
        }
    }

    /// Returns the expected output length for a given input length.
    ///
    /// `input_size` can be expressed in either frames or samples; the
    /// returned value uses the same unit.
    #[must_use]
    pub fn expected_output_size(&self, input_size: usize) -> usize {
        (input_size * self.config.output_sample_rate).div_ceil(self.config.input_sample_rate)
    }

    /// Resamples a complete interleaved input buffer and returns all output samples.
    ///
    /// This is a convenience wrapper around the streaming API.
    ///
    /// When the `rayon` feature is enabled, each channel is processed in parallel.
    pub fn process_all(&mut self, input: &[T]) -> Result<Vec<T>, Error>
    where
        T: Send + Sync,
    {
        if !input.len().is_multiple_of(self.config.channels) {
            return Err(Error::MalformedInputLength {
                channels: self.config.channels,
                samples: input.len(),
            });
        }

        if self.config.channels == 1 {
            return self.cores[0].process_all(input);
        }

        let channel_inputs = self.deinterleave_context(input);
        let mut channel_outputs: Vec<Vec<T>> = (0..self.config.channels).map(|_| Vec::new()).collect();

        #[cfg(feature = "rayon")]
        {
            self.cores
                .par_iter_mut()
                .zip(channel_inputs.par_iter())
                .zip(channel_outputs.par_iter_mut())
                .try_for_each(|((core, channel_input), channel_output)| -> Result<(), Error> {
                    *channel_output = core.process_all(channel_input)?;
                    Ok(())
                })?;
        }

        #[cfg(not(feature = "rayon"))]
        {
            for ((core, channel_input), channel_output) in self
                .cores
                .iter_mut()
                .zip(channel_inputs.iter())
                .zip(channel_outputs.iter_mut())
            {
                *channel_output = core.process_all(channel_input)?;
            }
        }

        let written_per_channel = channel_outputs.first().map_or(0, Vec::len);
        let mut output = vec![T::zero(); written_per_channel * self.config.channels];
        for frame_idx in 0..written_per_channel {
            for channel_idx in 0..self.config.channels {
                output[frame_idx * self.config.channels + channel_idx] = channel_outputs[channel_idx][frame_idx];
            }
        }

        Ok(output)
    }

    /// Process multiple independent tracks.
    ///
    /// Each input slice is treated as its own stream with no inter-track context. See
    /// `batch_gapless()` for gapless processing of multiple tracks.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch(&self, inputs: &[&[T]]) -> Result<Vec<Vec<T>>, Error>
    where
        T: Send + Sync,
    {
        let config = self.config.clone();

        #[cfg(feature = "rayon")]
        {
            inputs
                .par_iter()
                .map(|input| {
                    let mut resampler = Ardftsrc::new(config.clone())?;
                    resampler.process_all(input)
                })
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .iter()
                .map(|input| {
                    let mut resampler = Ardftsrc::new(config.clone())?;
                    resampler.process_all(input)
                })
                .collect()
        }
    }

    /// Process multiple tracks in parallel while preserving adjacent-track context.
    ///
    /// Use this when you want to resample an entire album or track collection that is
    /// meant to be played back-to-back with no gaps.
    ///
    /// For each track:
    /// - `pre` is filled from the tail of the previous track when one exists.
    /// - `post` is filled from the head of the next track when one exists.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch_gapless(&self, inputs: &[&[T]]) -> Result<Vec<Vec<T>>, Error>
    where
        T: Send + Sync,
    {
        let config = self.config.clone();
        let context_samples = self.input_chunk_size();
        let channels = self.config.channels;

        #[cfg(feature = "rayon")]
        {
            inputs
                .par_iter()
                .enumerate()
                .map(|(idx, input)| {
                    let mut resampler = Ardftsrc::new(config.clone())?;

                    if idx > 0 {
                        let pre = Self::batch_context_tail(inputs[idx - 1], context_samples, channels);
                        if !pre.is_empty() {
                            resampler.pre(pre)?;
                        }
                    }

                    if idx + 1 < inputs.len() {
                        let post = Self::batch_context_head(inputs[idx + 1], context_samples, channels);
                        if !post.is_empty() {
                            resampler.post(post)?;
                        }
                    }

                    resampler.process_all(input)
                })
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .iter()
                .enumerate()
                .map(|(idx, input)| {
                    let mut resampler = Ardftsrc::new(config.clone())?;

                    if idx > 0 {
                        let pre = Self::batch_context_tail(inputs[idx - 1], context_samples, channels);
                        if !pre.is_empty() {
                            resampler.pre(pre)?;
                        }
                    }

                    if idx + 1 < inputs.len() {
                        let post = Self::batch_context_head(inputs[idx + 1], context_samples, channels);
                        if !post.is_empty() {
                            resampler.post(post)?;
                        }
                    }

                    resampler.process_all(input)
                })
                .collect()
        }
    }

    /// Resets internal streaming state so the next input is treated as a new, independent stream.
    ///
    /// Call this between unrelated audio inputs (for example, between files) when reusing the
    /// same resampler instance, so edge/history state from one input cannot bleed into the next.
    pub fn reset(&mut self) {
        for core in &mut self.cores {
            core.reset();
        }
        self.sample_pending_input.clear();
        self.samples_pending_output.clear();
        self.samples_finalized = false;
    }

    /// Accepts interleaved streaming samples of any length.
    ///
    /// Input is internally buffered and converted into fixed-size core chunks. This method does
    /// not return produced output directly; call `read_samples()` to drain available samples.
    pub fn write_samples(&mut self, input: &[T]) -> Result<(), Error> {
        // A write after sample finalization starts a new independent stream.
        if self.samples_finalized {
            self.reset();
        }

        self.sample_pending_input.extend(input.iter().copied());

        self.process_pending_samples(false);

        Ok(())
    }

    // Process all pending input into the output queue.
    fn process_pending_samples(&mut self, finalize: bool) {
        while self.sample_pending_input.len() >= self.input_chunk_size() {
            // We have enough input to process a chunk, drain the input buffer into the input chunk buffer.
            self.samples_input_chunk_buffer.clear();
            self.samples_input_chunk_buffer
                .extend(self.sample_pending_input.drain(..self.input_chunk_size()));

            let input_adapter = InterleavedSlice::new(
                &self.samples_input_chunk_buffer,
                self.config.channels,
                self.derived.input_chunk_frames,
            )
            .expect("ardftsrc: Invalid input chunk size. This is a bug in the ardftsrc crate.");
            let mut output_adapter = InterleavedSlice::new_mut(
                &mut self.samples_output_chunk_buffer,
                self.config.channels,
                self.derived.output_chunk_frames,
            )
            .expect("ardftsrc: Invalid output chunk size. This is a bug in the ardftsrc crate.");

            let _samples_written = self
                .process_chunk_inner(&input_adapter, &mut output_adapter, false)
                .expect("ardftsrc: Invalid chunk size. This is a bug in the ardftsrc crate.");
            debug_assert_eq!(_samples_written, self.output_chunk_size());

            self.samples_pending_output.extend(self.samples_output_chunk_buffer.iter().copied());
        }

        // Process the final chunk (mabye undersized) and finalize the entire stream
        if finalize && !self.sample_pending_input.is_empty() {
            // Remaining samples must form complete frames
            let remaining_samples = self.sample_pending_input.len();
            debug_assert!(remaining_samples % self.config.channels == 0);

            let remaining_frames = remaining_samples / self.config.channels;

            // Move remaining input into chunk buffer
            self.samples_input_chunk_buffer.clear();
            self.samples_input_chunk_buffer
                .extend(self.sample_pending_input.drain(..));

            let input_adapter =
                InterleavedSlice::new(&self.samples_input_chunk_buffer, self.config.channels, remaining_frames)
                    .expect("ardftsrc: Invalid final input chunk size.");

            // Output buffer should already be sized for max possible output frames.
            let mut output_adapter = InterleavedSlice::new_mut(
                &mut self.samples_output_chunk_buffer,
                self.config.channels,
                self.derived.output_chunk_frames,
            )
            .expect("ardftsrc: Invalid output chunk size.");

            let samples_written = self.process_chunk_inner(&input_adapter, &mut output_adapter, true).expect("ardftsrc: Invalid chunk size. This is a bug in the ardftsrc crate.");

            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));

            // Call finalize() and write the final tail output.
            let mut output_adapter = InterleavedSlice::new_mut(
                &mut self.samples_output_chunk_buffer,
                self.config.channels,
                self.derived.output_chunk_frames,
            )
            .expect("ardftsrc: Invalid output chunk size.");

            let samples_written = self
                .finalize(&mut output_adapter)
                .expect("ardftsrc: Invalid output chunk size. This is a bug in the ardftsrc crate.");
            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.into_iter().take(samples_written));
        }
    }

    /// Reads up to `output.len()` interleaved samples from internally buffered output.
    ///
    /// Returns the number of samples copied into `output`.
    pub fn read_samples(&mut self, output: &mut [T]) -> usize {
        let drain_count = output.len().min(self.samples_pending_output.len());

        for (dst, src) in output[..drain_count]
            .iter_mut()
            .zip(self.samples_pending_output.drain(..drain_count))
        {
            *dst = src;
        }

        drain_count
    }

    /// Marks the sample-stream as finalized and flushes delayed output into pending samples.
    ///
    /// After this call, no new input should be written for the current stream. Keep calling
    /// `read_samples()` until it returns zero to drain all finalized output.
    ///
    /// If your streaming pipeline does not need delayed tail output at end-of-stream, call
    /// `reset()` directly instead of `finalize_samples()`. This is specifically for abrupt
    /// switching cases (for example, skipping to another track) where you intentionally discard
    /// the previous track tail. For normal track endings where tail output is desired, use
    /// `finalize_samples()`.
    ///
    /// For multi-channel streams, callers must provide a complete interleaved frame via write_samples()
    /// (a multiple of `channels`) before finalizing. If a dangling partial frame remains buffered,
    /// this method returns `Error::MalformedInputLength`.
    pub fn finalize_samples(&mut self) -> Result<(), Error> {
        if self.samples_finalized {
            return Err(Error::StreamAlreadyFinalized);
        }
        self.process_pending_samples(true);
        self.samples_finalized = true;
        Ok(())
    }

    /// Processes a streaming chunk. Channels are interleaved in the input buffer.
    ///
    /// `input` must contain exactly `input_buffer_size()` samples (all channels, interleaved),
    /// and `output` should provide at least `output_buffer_size()` capacity.
    ///
    /// The method returns the actual sample count written for this chunk, which may be smaller
    /// (for example while startup delay is being trimmed).
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn process_chunk<'a>(
        &mut self,
        input: &dyn Adapter<'a, T>,
        output: &mut dyn AdapterMut<'a, T>,
    ) -> Result<usize, Error> {
        // Output buffer must be at least the size of the output chunk (but can be larger).
        self.ensure_output_buffer_size(output)?;
        self.ensure_input_buffer_size(input, false)?;

        // Process the chunk.
        self.process_chunk_inner(input, output, false)
    }

    /// Processes the final chunk, which may be shorter than the regular chunk size.
    ///
    /// Call this exactly once at end-of-stream for the trailing partial chunk (it may be empty if
    /// input length is an exact multiple of `input_buffer_size()`). After this call, no further
    /// chunk-processing calls should be made; call `finalize()` to drain remaining delayed tail.
    /// Size `output` to at least `output_buffer_size()`.
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn process_chunk_final<'a>(
        &mut self,
        input: &dyn Adapter<'a, T>,
        output: &mut dyn AdapterMut<'a, T>,
    ) -> Result<usize, Error> {
        // Output buffer must be at least the size of the output chunk (but can be larger).
        self.ensure_output_buffer_size(output)?;
        self.ensure_input_buffer_size(input, true)?;

        // Process the chunk.
        self.process_chunk_inner(input, output, true)
    }

    #[inline]
    fn process_chunk_inner<'a>(
        &mut self,
        input: &dyn Adapter<'a, T>,
        output: &mut dyn AdapterMut<'a, T>,
        is_final: bool,
    ) -> Result<usize, Error> {
        let mut total_written = 0;

        // For each channel, process the chunk in the core
        for channel_idx in 0..self.config.channels {
            let core = &mut self.cores[channel_idx];
            let core_input = &mut self.input_staging[channel_idx][..input.frames()];
            input.copy_from_channel_to_slice(channel_idx, 0, core_input);
            let core_output = core.process_chunk(core_input, is_final)?;

            let (out_written, _) = output.copy_from_slice_to_channel(channel_idx, 0, core_output);
            if out_written != core_output.len() {
                return Err(Error::WrongFrameCount {
                    expected: core_output.len(),
                    actual: out_written,
                });
            }

            total_written += out_written;
        }

        Ok(total_written)
    }

    /// Emits delayed tail samples, then resets stream state.
    ///
    /// This flushes any remaining delayed samples that were held back by the chunked
    /// processing pipeline. It is the terminal step of a stream and should be called once per
    /// stream. If `process_chunk_final()` was not called, this treats the last accepted full chunk
    /// as terminal input.
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn finalize<'a>(&mut self, output: &mut dyn AdapterMut<'a, T>) -> Result<usize, Error> {
        // Ensure the output buffer is large enough.
        self.ensure_output_buffer_size(output)?;

        let mut total_written = 0;
        for (channel_idx, core) in self.cores.iter_mut().enumerate() {
            let finalize_output = core.finalize()?;
            let (out_written, _) = output.copy_from_slice_to_channel(channel_idx, 0, finalize_output);
            if out_written != finalize_output.len() {
                return Err(Error::WrongFrameCount {
                    expected: finalize_output.len(),
                    actual: out_written,
                });
            }
            total_written += out_written;
        }

        Ok(total_written)
    }

    /// Validates channel alignment and chunk length for an `input` slice.
    ///
    /// Non-final calls must provide exactly `input_buffer_size()` samples. Final calls may provide
    /// fewer samples but never more than `input_buffer_size()`.
    #[inline]
    fn ensure_input_buffer_size<'a>(&self, input: &dyn Adapter<'a, T>, is_final: bool) -> Result<(), Error> {
        if self.config.channels != input.channels() {
            return Err(Error::WrongChannelCount {
                expected: self.config.channels,
                actual: input.channels(),
            });
        }

        if (!is_final && input.frames() != self.derived.input_chunk_frames)
            || (is_final && input.frames() > self.derived.input_chunk_frames)
        {
            return Err(Error::WrongFrameCount {
                expected: self.derived.input_chunk_frames,
                actual: input.frames(),
            });
        }

        Ok(())
    }

    /// Verifies `output` has capacity for at least one produced chunk.
    ///
    /// Callers should allocate at least `output_buffer_size()` samples before processing.
    #[inline]
    fn ensure_output_buffer_size<'a>(&self, output: &dyn AdapterMut<'a, T>) -> Result<(), Error> {
        let expected = self.output_chunk_size();
        if output.channels() * output.frames() < expected {
            return Err(Error::InsufficientOutputBuffer {
                expected,
                actual: output.channels() * output.frames(),
            });
        }
        Ok(())
    }

    /// Returns up to `max_samples` aligned head samples for interleaved context.
    fn batch_context_head(input: &[T], max_samples: usize, channels: usize) -> &[T] {
        let aligned_input_len = input.len() - (input.len() % channels);
        let mut take = aligned_input_len.min(max_samples);
        take -= take % channels;
        &input[..take]
    }

    /// Returns up to `max_samples` aligned tail samples for interleaved context.
    fn batch_context_tail(input: &[T], max_samples: usize, channels: usize) -> &[T] {
        let aligned_input_len = input.len() - (input.len() % channels);
        let mut take = aligned_input_len.min(max_samples);
        take -= take % channels;
        let start = aligned_input_len - take;
        &input[start..aligned_input_len]
    }

    /// Returns true when rates match and FFT processing can be bypassed losslessly.
    fn is_passthrough(&self) -> bool {
        self.config.input_sample_rate == self.config.output_sample_rate
    }

    // Copy the interleaved input samples into the input staging buffers (one input buffer per channel).
    fn stage_input_channels(&mut self, input: &[T]) {
        let num_frames = input.len() / self.config.channels;

        if self.config.channels == 1 {
            self.input_staging[0][..num_frames].copy_from_slice(&input[..num_frames]);
            return;
        }

        for channel_input in &mut self.input_staging {
            channel_input[..num_frames].fill(T::zero());
        }

        for (frame_idx, frame) in input.chunks_exact(self.config.channels).enumerate() {
            for (channel_idx, sample) in frame.iter().enumerate() {
                self.input_staging[channel_idx][frame_idx] = *sample;
            }
        }
    }

    /// Sets previous-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the last chunk of the previous track.
    ///
    /// The buffer must use the same channel interleaving as stream input.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the end of the previous track.
    /// - Query chunk size with `input_buffer_size()`.
    ///
    /// Shorter buffers are still valid: any missing start context falls back to LPC
    /// extrapolation.
    pub fn pre(&mut self, pre: &[T]) -> Result<(), Error> {
        match self.normalize_context(pre)? {
            None => {
                for core in &mut self.cores {
                    core.pre(Vec::new());
                }
            }
            Some(pre) => {
                let per_channel = self.deinterleave_context(pre);
                for (core, samples) in self.cores.iter_mut().zip(per_channel) {
                    core.pre(samples);
                }
            }
        }
        Ok(())
    }

    /// Sets next-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the first chunk of the next track.
    ///
    /// You may call this at any time while the current stream is still active. It must be called
    /// before "process_chunk_final(...)" (for chunk API) or "finalize_samples()" (for samples API).
    ///
    /// This is useful for live gapless handoff: while track A is streaming, once track B is known you
    /// can call `post(...)` on track A with B's head samples so A's stop-edge uses real next-track
    /// context.
    ///
    /// The buffer must use the same channel interleaving as stream input.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the start of the next track.
    /// - Query chunk size with `input_buffer_size()`.
    ///
    /// Shorter buffers are still valid: any missing stop context falls back to LPC
    /// extrapolation.
    pub fn post(&mut self, post: &[T]) -> Result<(), Error> {
        match self.normalize_context(post)? {
            None => {
                for core in &mut self.cores {
                    core.post(Vec::new());
                }
            }
            Some(post) => {
                let per_channel = self.deinterleave_context(post);
                for (core, samples) in self.cores.iter_mut().zip(per_channel) {
                    core.post(samples);
                }
            }
        }
        Ok(())
    }

    fn normalize_context<'a>(&self, context: &'a [T]) -> Result<Option<&'a [T]>, Error> {
        if context.is_empty() {
            return Ok(None);
        }
        if !context.len().is_multiple_of(self.config.channels) {
            return Err(Error::MalformedInputLength {
                channels: self.config.channels,
                samples: context.len(),
            });
        }
        Ok(Some(context))
    }

    fn deinterleave_context(&self, context: &[T]) -> Vec<Vec<T>> {
        let frames = context.len() / self.config.channels;
        if self.config.channels == 1 {
            return vec![context[..frames].to_vec()];
        }

        let mut channels = vec![Vec::with_capacity(frames); self.config.channels];
        for frame in context.chunks_exact(self.config.channels) {
            for (channel_idx, sample) in frame.iter().enumerate() {
                channels[channel_idx].push(*sample);
            }
        }
        channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArdftsrcCore, TaperType};
    use dasp_signal::Signal;

    fn input_chunk_frames(resampler: &Ardftsrc<f32>) -> usize {
        resampler.input_chunk_size() / resampler.config().channels
    }

    fn output_chunk_frames(resampler: &Ardftsrc<f32>) -> usize {
        resampler.output_chunk_size() / resampler.config().channels
    }

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

    fn stereo_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config {
            channels: 2,
            ..mono_config(input_sample_rate, output_sample_rate)
        }
    }

    fn resample_stream_with_context(
        config: Config,
        input: &[f32],
        pre: Option<&[f32]>,
        post: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut resampler = Ardftsrc::new(config).unwrap();
        if let Some(pre) = pre {
            resampler.pre(pre).unwrap();
        }
        if let Some(post) = post {
            resampler.post(post).unwrap();
        }

        let input_buffer_size = resampler.input_chunk_size();
        let full_chunks = input.len() / input_buffer_size;
        let has_partial = !input.len().is_multiple_of(input_buffer_size);
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut output = vec![0.0; output_blocks * resampler.output_chunk_size()];
        let mut written = 0;
        let mut offset = 0;
        while offset + input_buffer_size <= input.len() {
            written += resampler
                .process_chunk(&input[offset..offset + input_buffer_size], &mut output[written..])
                .unwrap();
            offset += input_buffer_size;
        }
        written += resampler
            .process_chunk_final(&input[offset..], &mut output[written..])
            .unwrap();
        written += resampler.finalize(&mut output[written..]).unwrap();
        output.truncate(written);
        output
    }

    fn resample_stream_with_sample_api(
        config: Config,
        input: &[f32],
        write_block_size: usize,
        read_block_size: usize,
    ) -> Vec<f32> {
        let mut resampler = Ardftsrc::new(config).unwrap();
        let mut output = Vec::new();
        let mut read_buffer = vec![0.0; read_block_size.max(1)];
        let channels = resampler.config().channels;
        let mut write_step = write_block_size.max(1);
        write_step -= write_step % channels;
        if write_step == 0 {
            write_step = channels;
        }

        let mut offset = 0;
        while offset < input.len() {
            let end = (offset + write_step).min(input.len());
            resampler.write_samples(&input[offset..end]).unwrap();
            offset = end;

            loop {
                let written = resampler.read_samples(&mut read_buffer);
                if written == 0 {
                    break;
                }
                output.extend_from_slice(&read_buffer[..written]);
            }
        }

        resampler.finalize_samples().unwrap();

        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&read_buffer[..written]);
        }

        output
    }

    fn max_abs_error_with_index(actual: &[f32], expected: &[f32]) -> (f32, usize) {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs_error = 0.0f32;
        let mut max_abs_error_idx = 0;
        for (idx, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs_error = (left - right).abs();
            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_abs_error_idx = idx;
            }
        }
        (max_abs_error, max_abs_error_idx)
    }

    fn run_core_process_all(core: &mut ArdftsrcCore<f32>, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::new();
        let mut chunk_output = vec![0.0; core.output_buffer_size()];
        let mut offset = 0;
        let input_chunk = core.input_buffer_size();

        while offset + input_chunk <= input.len() {
            let written = core
                .process_chunk(&input[offset..offset + input_chunk], &mut chunk_output, false)
                .unwrap();
            output.extend_from_slice(&chunk_output[..written]);
            offset += input_chunk;
        }

        let final_input = &input[offset..];
        let written = core.process_chunk(final_input, &mut chunk_output, true).unwrap();
        output.extend_from_slice(&chunk_output[..written]);

        let written = core.finalize(&mut chunk_output).unwrap();
        output.extend_from_slice(&chunk_output[..written]);

        output
    }

    fn deinterleave_channel(samples: &[f32], channels: usize, channel: usize) -> Vec<f32> {
        samples.chunks_exact(channels).map(|frame| frame[channel]).collect()
    }

    #[test]
    fn silence_stays_silent() {
        let mut resampler = Ardftsrc::new(mono_config(48_000, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];

        let output = resampler.process_all(&input).unwrap();

        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn same_rate_passthrough_preserves_samples() {
        let mut resampler = Ardftsrc::new(mono_config(48_000, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&resampler) * 2 + 7)
            .map(|frame| (frame as f32 * 0.013).cos())
            .collect();

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(output, input);
        assert_eq!(resampler.output_delay_frames(), 0);
    }

    #[test]
    fn impulse_output_is_finite() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let mut input = vec![0.0; input_chunk_frames(&resampler)];
        input[0] = 1.0;

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(output.len(), resampler.expected_output_size(input.len()));
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn first_chunk_lpc_start_edge_is_finite() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&resampler))
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let written = resampler.process_chunk(&input, &mut output).unwrap();

        assert_eq!(
            written,
            output_chunk_frames(&resampler) - resampler.output_delay_frames()
        );
        assert!(output[..written].iter().all(|sample| sample.is_finite()));
        assert!(output[..written].iter().any(|sample| sample.abs() > 1e-6));
    }

    #[test]
    fn short_final_chunk_lpc_stop_edge_is_finite_and_bounded() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input_frames = input_chunk_frames(&resampler) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.02).sin() * 0.1)
            .collect();

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(output.len(), resampler.expected_output_size(input_frames));
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().all(|sample| sample.abs() < 1.0));
    }

    #[test]
    fn expected_output_size_matches_frame_count_conversion() {
        let resampler = Ardftsrc::<f32>::new(stereo_config(44_100, 48_000)).unwrap();
        let input_samples = resampler.input_chunk_size() * 2 + 14;
        let input_frames = input_samples / resampler.config().channels;

        let expected_from_frames = resampler.expected_output_size(input_frames) * resampler.config().channels;
        let expected_from_samples = resampler.expected_output_size(input_samples);
        assert_eq!(expected_from_samples, expected_from_frames);
    }

    #[test]
    fn expected_output_size_accepts_non_interleaved_length() {
        let resampler = Ardftsrc::<f32>::new(stereo_config(44_100, 48_000)).unwrap();
        let expected = (3usize * 48_000).div_ceil(44_100);
        assert_eq!(resampler.expected_output_size(3), expected);
    }

    #[test]
    fn stereo_wrapper_matches_channel_core_outputs() {
        let config = stereo_config(44_100, 48_000);
        let mut wrapper = Ardftsrc::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&wrapper) * 2 + 37;
        let input: Vec<f32> = (0..input_frames)
            .flat_map(|frame| {
                let t = frame as f32;
                [(t * 0.01).sin() * 0.25, (t * 0.017).cos() * 0.2]
            })
            .collect();

        let context_frames = input_chunk_frames(&wrapper) / 2;
        let pre: Vec<f32> = (0..context_frames)
            .flat_map(|frame| {
                let t = frame as f32;
                [(t * 0.03).sin() * 0.1, (t * 0.027).cos() * 0.08]
            })
            .collect();
        let post: Vec<f32> = (0..context_frames)
            .flat_map(|frame| {
                let t = frame as f32;
                [(t * 0.021).sin() * 0.09, (t * 0.012).cos() * 0.07]
            })
            .collect();

        wrapper.pre(&pre).unwrap();
        wrapper.post(&post).unwrap();
        let wrapped_output = wrapper.process_all(&input).unwrap();

        let mono_config = Config {
            channels: 1,
            ..config.clone()
        };
        let derived = mono_config.derive_config::<f32>().unwrap();
        let mut left_core = ArdftsrcCore::<f32>::new(derived.clone());
        let mut right_core = ArdftsrcCore::<f32>::new(derived);

        left_core.pre(deinterleave_channel(&pre, 2, 0));
        right_core.pre(deinterleave_channel(&pre, 2, 1));
        left_core.post(deinterleave_channel(&post, 2, 0));
        right_core.post(deinterleave_channel(&post, 2, 1));

        let left_output = run_core_process_all(&mut left_core, &deinterleave_channel(&input, 2, 0));
        let right_output = run_core_process_all(&mut right_core, &deinterleave_channel(&input, 2, 1));

        assert_eq!(left_output.len(), right_output.len());
        assert_eq!(wrapped_output.len(), left_output.len() * 2);

        for frame_idx in 0..left_output.len() {
            let left = wrapped_output[frame_idx * 2];
            let right = wrapped_output[frame_idx * 2 + 1];
            assert!((left - left_output[frame_idx]).abs() < 1e-6);
            assert!((right - right_output[frame_idx]).abs() < 1e-6);
        }
    }

    #[test]
    fn pre_and_post_reject_non_interleaved_length() {
        let mut resampler = Ardftsrc::<f32>::new(stereo_config(44_100, 48_000)).unwrap();

        assert!(matches!(
            resampler.pre(&[0.0; 3]),
            Err(Error::MalformedInputLength {
                channels: 2,
                samples: 3
            })
        ));
        assert!(matches!(
            resampler.post(&[0.0; 3]),
            Err(Error::MalformedInputLength {
                channels: 2,
                samples: 3
            })
        ));
    }

    #[test]
    fn pre_context_changes_first_chunk_output() {
        let mut with_zero_pre = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let mut with_one_pre = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&with_zero_pre))
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();
        let mut out_zero = vec![0.0; output_chunk_frames(&with_zero_pre)];
        let mut out_one = vec![0.0; output_chunk_frames(&with_one_pre)];
        let pre_zero = vec![0.0; input_chunk_frames(&with_zero_pre)];
        let pre_one = vec![1.0; input_chunk_frames(&with_one_pre)];
        with_zero_pre.pre(&pre_zero).unwrap();
        with_one_pre.pre(&pre_one).unwrap();

        let written_zero = with_zero_pre.process_chunk(&input, &mut out_zero).unwrap();
        let written_one = with_one_pre.process_chunk(&input, &mut out_one).unwrap();

        assert_eq!(written_zero, written_one);
        assert!(
            out_zero[..written_zero]
                .iter()
                .zip(out_one[..written_one].iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    #[test]
    fn post_context_changes_flush_output() {
        let mut with_zero_post = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let mut with_one_post = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&with_zero_post))
            .map(|frame| (frame as f32 * 0.015).cos() * 0.125)
            .collect();
        let mut chunk_out = vec![0.0; output_chunk_frames(&with_zero_post)];
        let mut flush_zero = vec![0.0; output_chunk_frames(&with_zero_post)];
        let mut flush_one = vec![0.0; output_chunk_frames(&with_one_post)];
        let post_zero = vec![0.0; input_chunk_frames(&with_zero_post)];
        let post_one = vec![1.0; input_chunk_frames(&with_one_post)];
        with_zero_post.post(&post_zero).unwrap();
        with_one_post.post(&post_one).unwrap();

        with_zero_post.process_chunk(&input, &mut chunk_out).unwrap();
        with_one_post.process_chunk(&input, &mut chunk_out).unwrap();
        let written_zero = with_zero_post.finalize(&mut flush_zero).unwrap();
        let written_one = with_one_post.finalize(&mut flush_one).unwrap();

        assert_eq!(written_zero, written_one);
        assert!(
            flush_zero[..written_zero]
                .iter()
                .zip(flush_one[..written_one].iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    #[test]
    fn flush_after_full_chunk_uses_lpc_tail_edge() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&resampler))
            .map(|frame| (frame as f32 * 0.015).cos() * 0.125)
            .collect();
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let first_written = resampler.process_chunk(&input, &mut output).unwrap();
        let flush_written = resampler.finalize(&mut output).unwrap();

        assert_eq!(first_written + flush_written, output_chunk_frames(&resampler));
        assert!(output[..flush_written].iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn streaming_and_offline_paths_match() {
        let config = mono_config(44_100, 48_000);
        let mut offline = Ardftsrc::new(config.clone()).unwrap();
        let mut streaming = Ardftsrc::new(config).unwrap();
        let input_frames = input_chunk_frames(&streaming) * 2 + input_chunk_frames(&streaming) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();

        let offline_output = offline.process_all(&input).unwrap();

        let full_chunks = input.len() / input_chunk_frames(&streaming);
        let has_partial = !input.len().is_multiple_of(input_chunk_frames(&streaming));
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut streaming_output = vec![0.0; output_blocks * output_chunk_frames(&streaming)];
        let mut written = 0;
        let chunk_len = input_chunk_frames(&streaming);
        let mut offset = 0;
        while offset + chunk_len <= input.len() {
            written += streaming
                .process_chunk(&input[offset..offset + chunk_len], &mut streaming_output[written..])
                .unwrap();
            offset += chunk_len;
        }
        written += streaming
            .process_chunk_final(&input[offset..], &mut streaming_output[written..])
            .unwrap();
        written += streaming.finalize(&mut streaming_output[written..]).unwrap();
        streaming_output.truncate(written);
        streaming_output.truncate(offline_output.len());

        assert_eq!(offline_output.len(), streaming_output.len());
        for (left, right) in offline_output.iter().zip(streaming_output) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn split_resampling_with_pre_post_matches_full_stream() {
        let mut config = mono_config(44_100, 48_000);
        config.taper_type = TaperType::Cosine(1.55);
        let mut full_resampler = Ardftsrc::new(config.clone()).unwrap();
        let context_len = full_resampler.input_chunk_size();
        let split_frames = 4 * config.input_sample_rate;
        let total_frames = split_frames * 3;
        let output_sample_rate = config.output_sample_rate;

        let input: Vec<f32> = dasp_signal::rate(config.input_sample_rate as f64)
            .const_hz(440.0)
            .sine()
            .take(total_frames)
            .map(|sample| sample as f32 * 0.25)
            .collect();

        let full_output = full_resampler.process_all(&input).unwrap();

        let split0 = &input[..split_frames];
        let split1 = &input[split_frames..split_frames * 2];
        let split2 = &input[split_frames * 2..];

        let split0_post = &split1[..context_len];
        let split1_pre = &split0[split0.len() - context_len..];
        let split1_post = &split2[..context_len];
        let split2_pre = &split1[split1.len() - context_len..];

        let out0 = resample_stream_with_context(config.clone(), split0, None, Some(split0_post));
        let out1 = resample_stream_with_context(config.clone(), split1, Some(split1_pre), Some(split1_post));
        let out2 = resample_stream_with_context(config, split2, Some(split2_pre), None);

        let mut stitched = Vec::with_capacity(out0.len() + out1.len() + out2.len());
        stitched.extend_from_slice(&out0);
        stitched.extend_from_slice(&out1);
        stitched.extend_from_slice(&out2);

        let tolerance = 5e-7f32;
        assert_eq!(stitched.len(), full_output.len());
        let (max_abs_error, max_abs_error_idx) = max_abs_error_with_index(&stitched, &full_output);
        let output_seam_1 = out0.len();
        let output_seam_2 = out0.len() + out1.len();
        let nearest_seam_distance = max_abs_error_idx
            .abs_diff(output_seam_1)
            .min(max_abs_error_idx.abs_diff(output_seam_2));
        let max_error_time_seconds = max_abs_error_idx as f64 / output_sample_rate as f64;
        eprintln!(
            "max_abs_error={max_abs_error} at sample={max_abs_error_idx} (~{max_error_time_seconds:.6}s), seam_1={output_seam_1}, seam_2={output_seam_2}, nearest_seam_distance={nearest_seam_distance}"
        );
        assert!(
            max_abs_error < tolerance,
            "max_abs_error={max_abs_error}, tolerance={tolerance}"
        );
    }

    #[test]
    fn split_resampling_is_worse_without_pre_post() {
        let config = mono_config(44_100, 48_000);
        let mut full_resampler = Ardftsrc::new(config.clone()).unwrap();
        let context_len = full_resampler.input_chunk_size();
        let split_frames = 4 * config.input_sample_rate;
        let total_frames = split_frames * 3;
        let output_sample_rate = config.output_sample_rate;

        let input: Vec<f32> = dasp_signal::rate(config.input_sample_rate as f64)
            .const_hz(440.0)
            .sine()
            .take(total_frames)
            .map(|sample| sample as f32 * 0.25)
            .collect();

        let full_output = full_resampler.process_all(&input).unwrap();

        let split0 = &input[..split_frames];
        let split1 = &input[split_frames..split_frames * 2];
        let split2 = &input[split_frames * 2..];

        let split0_post = &split1[..context_len];
        let split1_pre = &split0[split0.len() - context_len..];
        let split1_post = &split2[..context_len];
        let split2_pre = &split1[split1.len() - context_len..];

        let with_out0 = resample_stream_with_context(config.clone(), split0, None, Some(split0_post));
        let with_out1 = resample_stream_with_context(config.clone(), split1, Some(split1_pre), Some(split1_post));
        let with_out2 = resample_stream_with_context(config.clone(), split2, Some(split2_pre), None);

        let without_out0 = resample_stream_with_context(config.clone(), split0, None, None);
        let without_out1 = resample_stream_with_context(config.clone(), split1, None, None);
        let without_out2 = resample_stream_with_context(config, split2, None, None);

        let mut with_context = Vec::with_capacity(with_out0.len() + with_out1.len() + with_out2.len());
        with_context.extend_from_slice(&with_out0);
        with_context.extend_from_slice(&with_out1);
        with_context.extend_from_slice(&with_out2);

        let mut without_context = Vec::with_capacity(without_out0.len() + without_out1.len() + without_out2.len());
        without_context.extend_from_slice(&without_out0);
        without_context.extend_from_slice(&without_out1);
        without_context.extend_from_slice(&without_out2);

        let (with_max_abs_error, with_max_abs_error_idx) = max_abs_error_with_index(&with_context, &full_output);
        let (without_max_abs_error, without_max_abs_error_idx) =
            max_abs_error_with_index(&without_context, &full_output);

        eprintln!(
            "with_context max_abs_error={with_max_abs_error} at sample={with_max_abs_error_idx} (~{:.6}s)",
            with_max_abs_error_idx as f64 / output_sample_rate as f64
        );
        eprintln!(
            "without_context max_abs_error={without_max_abs_error} at sample={without_max_abs_error_idx} (~{:.6}s)",
            without_max_abs_error_idx as f64 / output_sample_rate as f64
        );

        let min_ratio = 100.0f32;
        let ratio = without_max_abs_error / with_max_abs_error;
        assert!(
            ratio >= min_ratio,
            "expected without_context/with_context ratio >= {min_ratio}, got {ratio} (without={without_max_abs_error}, with={with_max_abs_error})"
        );
    }

    #[test]
    fn process_all_resets_state_between_calls() {
        let config = mono_config(44_100, 48_000);
        let mut reused = Ardftsrc::new(config.clone()).unwrap();
        let mut reference = Ardftsrc::new(config).unwrap();
        let first_input: Vec<f32> = (0..(input_chunk_frames(&reused) * 2 + 11))
            .map(|frame| (frame as f32 * 0.009).sin() * 0.2)
            .collect();
        let second_input: Vec<f32> = (0..(input_chunk_frames(&reused) + 7))
            .map(|frame| (frame as f32 * 0.013).cos() * 0.15)
            .collect();

        let first_reused = reused.process_all(&first_input).unwrap();
        let first_reference = reference.process_all(&first_input).unwrap();
        assert_eq!(first_reused.len(), first_reference.len());
        for (left, right) in first_reused.iter().zip(first_reference.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }

        let second_reused = reused.process_all(&second_input).unwrap();
        let second_reference = reference.process_all(&second_input).unwrap();
        assert_eq!(second_reused.len(), second_reference.len());
        for (left, right) in second_reused.iter().zip(second_reference.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn stereo_channels_are_processed_independently() {
        let mut resampler = Ardftsrc::new(stereo_config(44_100, 48_000)).unwrap();
        let mut input = vec![0.0; resampler.input_chunk_size()];
        input[0] = 1.0;

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(
            output.len(),
            resampler.expected_output_size(input_chunk_frames(&resampler)) * 2
        );
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.chunks_exact(2).any(|frame| frame[0].abs() > 1e-6));
        assert!(output.chunks_exact(2).all(|frame| frame[1].abs() < 1e-6));
    }

    #[test]
    fn stereo_streaming_and_offline_paths_match() {
        let config = stereo_config(44_100, 48_000);
        let mut offline = Ardftsrc::new(config.clone()).unwrap();
        let mut streaming = Ardftsrc::new(config).unwrap();
        let input_frames = input_chunk_frames(&streaming) * 2 + input_chunk_frames(&streaming) / 3;
        let mut input = Vec::with_capacity(input_frames * 2);
        for frame in 0..input_frames {
            input.push((frame as f32 * 0.01).sin() * 0.25);
            input.push((frame as f32 * 0.017).cos() * 0.125);
        }

        let offline_output = offline.process_all(&input).unwrap();

        let input_buffer_size = streaming.input_chunk_size();
        let full_chunks = input.len() / input_buffer_size;
        let has_partial = !input.len().is_multiple_of(input_buffer_size);
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut streaming_output = vec![0.0; output_blocks * streaming.output_chunk_size()];
        let mut written = 0;
        let mut offset = 0;
        while offset + input_buffer_size <= input.len() {
            written += streaming
                .process_chunk(
                    &input[offset..offset + input_buffer_size],
                    &mut streaming_output[written..],
                )
                .unwrap();
            offset += input_buffer_size;
        }
        written += streaming
            .process_chunk_final(&input[offset..], &mut streaming_output[written..])
            .unwrap();
        written += streaming.finalize(&mut streaming_output[written..]).unwrap();
        streaming_output.truncate(written);
        streaming_output.truncate(offline_output.len());

        assert_eq!(offline_output.len(), streaming_output.len());
        for (left, right) in offline_output.iter().zip(streaming_output) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn rejects_wrong_streaming_chunk_size() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler) - 1];

        assert!(resampler.process_chunk(&input, &mut []).is_err());
    }

    #[test]
    fn too_small_output_does_not_advance_stream_state() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut too_small = vec![0.0; output_chunk_frames(&resampler) - 1];

        assert!(matches!(
            resampler.process_chunk(&input, &mut too_small),
            Err(Error::InsufficientOutputBuffer { .. })
        ));
        assert_eq!(resampler.input_sample_processed(), 0);

        let mut output = vec![0.0; output_chunk_frames(&resampler)];
        let expected = output_chunk_frames(&resampler) - resampler.output_delay_frames();
        assert_eq!(resampler.process_chunk(&input, &mut output).unwrap(), expected);
        assert_eq!(resampler.input_sample_processed(), input.len());
    }

    #[test]
    fn input_sample_count_tracks_full_and_partial_stream_input() {
        let mut resampler = Ardftsrc::new(stereo_config(44_100, 48_000)).unwrap();
        let mut output = vec![0.0; resampler.output_chunk_size()];
        let full = vec![0.0; resampler.input_chunk_size()];
        let partial = vec![0.0; 10];

        assert_eq!(resampler.input_sample_processed(), 0);
        let _ = resampler.process_chunk(&full, &mut output).unwrap();
        assert_eq!(resampler.input_sample_processed(), full.len());
        let _ = resampler.process_chunk_final(&partial, &mut output).unwrap();
        assert_eq!(resampler.input_sample_processed(), full.len() + partial.len());
    }

    #[test]
    fn too_small_finish_output_does_not_mark_flushed() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut first_output = vec![0.0; output_chunk_frames(&resampler)];
        resampler.process_chunk(&input, &mut first_output).unwrap();

        let mut too_small = vec![0.0; resampler.output_delay_frames() - 1];

        assert!(matches!(
            resampler.finalize(&mut too_small),
            Err(Error::InsufficientOutputBuffer { .. })
        ));

        let mut output = vec![0.0; output_chunk_frames(&resampler)];
        assert_eq!(
            resampler.finalize(&mut output).unwrap(),
            resampler.output_delay_frames()
        );
    }

    #[test]
    fn first_chunk_is_delay_trimmed_and_flushes_tail() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let first_written = resampler.process_chunk(&input, &mut output).unwrap();
        let flush_written = resampler.finalize(&mut output).unwrap();

        assert_eq!(first_written + flush_written, output_chunk_frames(&resampler));
        assert_eq!(
            first_written,
            output_chunk_frames(&resampler) - resampler.output_delay_frames()
        );
        assert_eq!(flush_written, resampler.output_delay_frames());
    }

    #[test]
    fn finalize_without_explicit_final_chunk_caps_to_expected_total() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let input_samples = input.len() * 2;
        let expected_total = resampler.expected_output_size(input_samples);
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let first_written = resampler.process_chunk(&input, &mut output).unwrap();
        let second_written = resampler.process_chunk(&input, &mut output).unwrap();
        let flush_written = resampler.finalize(&mut output).unwrap();

        assert_eq!(first_written + second_written + flush_written, expected_total);
    }

    #[test]
    fn empty_finish_does_not_emit_extra_block() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        resampler.process_chunk(&input, &mut output).unwrap();
        assert_eq!(resampler.process_chunk_final(&[], &mut output).unwrap(), 0);
        assert_eq!(
            resampler.finalize(&mut output).unwrap(),
            resampler.output_delay_frames()
        );
    }

    #[test]
    fn empty_stream_flushes_no_samples() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        assert_eq!(resampler.process_chunk_final(&[], &mut output).unwrap(), 0);
        assert_eq!(resampler.finalize(&mut output).unwrap(), 0);
    }

    #[test]
    fn second_finish_after_reset_returns_zero_without_input() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();

        let mut output = vec![0.0; output_chunk_frames(&resampler)];
        assert_eq!(resampler.finalize(&mut output).unwrap(), 0);
        assert_eq!(resampler.finalize(&mut output).unwrap(), 0);
    }

    #[test]
    fn write_samples_accepts_non_channel_aligned_input() {
        let mut resampler = Ardftsrc::new(stereo_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; 3];
        assert!(resampler.write_samples(&input).is_ok());
    }

    #[test]
    fn finalize_samples_rejects_dangling_partial_frame() {
        let mut resampler = Ardftsrc::new(stereo_config(44_100, 48_000)).unwrap();
        resampler.write_samples(&[0.0]).unwrap();
        assert!(matches!(
            resampler.finalize_samples(),
            Err(Error::MalformedInputLength {
                channels: 2,
                samples: 1
            })
        ));
    }

    #[test]
    fn write_samples_and_read_samples_match_chunk_output() {
        let config = mono_config(44_100, 48_000);
        let mut chunk_resampler = Ardftsrc::new(config.clone()).unwrap();
        let mut sample_resampler = Ardftsrc::new(config).unwrap();
        let input: Vec<f32> = (0..chunk_resampler.input_chunk_size())
            .map(|frame| (frame as f32 * 0.013).sin() * 0.3)
            .collect();

        let split = input.len() / 2;
        sample_resampler.write_samples(&input[..split]).unwrap();
        sample_resampler.write_samples(&input[split..]).unwrap();

        let mut sample_output = vec![0.0; sample_resampler.output_chunk_size()];
        let sample_written = sample_resampler.read_samples(&mut sample_output);

        let mut chunk_output = vec![0.0; chunk_resampler.output_chunk_size()];
        let chunk_written = chunk_resampler.process_chunk(&input, &mut chunk_output).unwrap();

        assert_eq!(sample_written, chunk_written);
        for (left, right) in sample_output[..sample_written]
            .iter()
            .zip(chunk_output[..chunk_written].iter())
        {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn sample_api_finalize_matches_process_all_total_output() {
        let config = mono_config(44_100, 48_000);
        let mut offline = Ardftsrc::new(config.clone()).unwrap();
        let driver = Ardftsrc::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&driver) * 2 + input_chunk_frames(&driver) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.008).sin() * 0.25)
            .collect();

        let expected = offline.process_all(&input).unwrap();
        let actual = resample_stream_with_sample_api(config, &input, 7, 11);

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn sample_api_stereo_matches_process_all_total_output() {
        let config = stereo_config(44_100, 48_000);
        let mut offline = Ardftsrc::new(config.clone()).unwrap();
        let driver = Ardftsrc::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&driver) * 2 + 17;
        let mut input = Vec::with_capacity(input_frames * 2);
        for frame in 0..input_frames {
            input.push((frame as f32 * 0.01).sin() * 0.25);
            input.push((frame as f32 * 0.017).cos() * 0.2);
        }

        let expected = offline.process_all(&input).unwrap();
        let actual = resample_stream_with_sample_api(config, &input, 9, 13);

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn finalize_samples_read_until_zero_drains_stream() {
        let config = mono_config(44_100, 48_000);
        let mut offline = Ardftsrc::new(config.clone()).unwrap();
        let mut stream = Ardftsrc::new(config).unwrap();
        let input_frames = input_chunk_frames(&stream) + input_chunk_frames(&stream) / 4;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.011).sin() * 0.2)
            .collect();

        let expected = offline.process_all(&input).unwrap();

        stream.write_samples(&input).unwrap();
        stream.finalize_samples().unwrap();

        let mut actual = Vec::new();
        let mut read_buffer = vec![0.0; 5];
        loop {
            let written = stream.read_samples(&mut read_buffer);
            if written == 0 {
                break;
            }
            actual.extend_from_slice(&read_buffer[..written]);
        }

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn write_samples_after_finalize_samples_starts_new_stream() {
        let mut stream = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let chunk = stream.input_chunk_size();
        let first = vec![0.1f32; chunk];
        let second = vec![0.2f32; chunk];

        stream.write_samples(&first).unwrap();
        stream.finalize_samples().unwrap();

        // Start a new stream; this should reset core history and sample counters.
        stream.write_samples(&second).unwrap();

        assert_eq!(stream.input_sample_processed(), second.len());
    }

    #[test]
    fn batch_test() {
        let config = mono_config(44_100, 48_000);
        let driver = Ardftsrc::new(config.clone()).unwrap();
        let chunk = input_chunk_frames(&driver);
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

        let expected: Vec<Vec<f32>> = tracks
            .iter()
            .map(|track| {
                let mut resampler = Ardftsrc::new(config.clone()).unwrap();
                resampler.process_all(track).unwrap()
            })
            .collect();
        let actual = driver.batch(&input_refs).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_track, expected_track) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_track.len(), expected_track.len());
            for (left, right) in actual_track.iter().zip(expected_track.iter()) {
                assert!((*left - *right).abs() < 1e-5);
            }
        }
    }
}
