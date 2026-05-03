use num_traits::Float;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use realfft::FftNum;

use crate::{ArdftsrcCore, Config, DerivedConfig, Error};

pub struct Ardftsrc<T = f32>
where
    T: Float + FftNum,
{
    config: Config,
    derived: DerivedConfig<T>,
    cores: Vec<ArdftsrcCore<T>>,
    input_staging: Vec<Vec<T>>,
    output_staging: Vec<Vec<T>>,
}

impl<T> Ardftsrc<T>
where
    T: Float + FftNum,
{
    /// Constructs a resampler from `config`.
    ///
    /// Returns a ready-to-use resampler instance, or an error if `config` is invalid or derived
    /// FFT geometry cannot be prepared.
    pub fn new(config: Config) -> Result<Self, Error> {
        let derived = config.derive_config::<T>()?;
        let cores = (0..config.channels)
            .map(|_| ArdftsrcCore::new(derived.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let input_staging = vec![vec![T::zero(); derived.input_chunk_frames]; config.channels];
        let output_staging = vec![vec![T::zero(); derived.output_chunk_frames]; config.channels];

        Ok(Self {
            config,
            derived,
            cores,
            input_staging,
            output_staging,
        })
    }

    /// Returns the configuration this instance was built with.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the total number of interleaved input samples processed.
    pub fn input_sample_count(&self) -> usize {
        self.cores.iter().map(ArdftsrcCore::input_sample_count).sum()
    }

    /// Returns the required non-final streaming chunk length in frames per channel.
    pub fn input_chunk_frames(&self) -> usize {
        self.derived.input_chunk_frames
    }

    /// Returns the required `input` length (interleaved samples) for each `process_chunk()` call.
    ///
    /// Use this to allocate/read fixed-size streaming input buffers.
    pub fn input_buffer_size(&self) -> usize {
        self.derived.input_chunk_frames * self.config.channels
    }

    /// Returns produced frame count for each non-passthrough transform block before trimming.
    pub fn output_chunk_frames(&self) -> usize {
        self.derived.output_chunk_frames
    }

    /// Returns the recommended per-call `output` capacity in interleaved samples.
    ///
    /// For chunked streaming, size output slices passed to `process_chunk()` and
    /// `process_chunk_final()` to at least this value.
    pub fn output_buffer_size(&self) -> usize {
        self.derived.output_chunk_frames * self.config.channels
    }

    /// Returns algorithmic latency to trim/flush, or zero in same-rate passthrough mode.
    pub fn output_delay_frames(&self) -> usize {
        if self.is_passthrough() {
            0
        } else {
            self.derived.output_offset
        }
    }

    /// Sets previous-track context.
    ///
    /// The buffer must use the same channel interleaving as stream input.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the last chunk of the previous track.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the end of the previous track.
    /// - Query chunk size with `input_chunk_frames()` (frames per channel), or
    ///   `input_buffer_size()` (interleaved samples).
    ///
    /// Shorter buffers are still valid: any missing start context falls back to LPC
    /// extrapolation.
    pub fn pre(&mut self, pre: Vec<T>) -> Result<(), Error> {
        match self.normalize_context(pre)? {
            None => {
                for core in &mut self.cores {
                    core.pre(Vec::new())?;
                }
            }
            Some(pre) => {
                let per_channel = self.deinterleave_context(&pre);
                for (core, samples) in self.cores.iter_mut().zip(per_channel.into_iter()) {
                    core.pre(samples)?;
                }
            }
        }
        Ok(())
    }

    /// Sets next-track context.
    ///
    /// The buffer must use the same channel interleaving as stream input.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the first chunk of the next track.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the start of the next track.
    /// - Query chunk size with `input_chunk_frames()` (frames per channel), or
    ///   `input_buffer_size()` (interleaved samples).
    ///
    /// Shorter buffers are still valid: any missing stop context falls back to LPC
    /// extrapolation.
    pub fn post(&mut self, post: Vec<T>) -> Result<(), Error> {
        match self.normalize_context(post)? {
            None => {
                for core in &mut self.cores {
                    core.post(Vec::new())?;
                }
            }
            Some(post) => {
                let per_channel = self.deinterleave_context(&post);
                for (core, samples) in self.cores.iter_mut().zip(per_channel.into_iter()) {
                    core.post(samples)?;
                }
            }
        }
        Ok(())
    }

    /// Output frames for a complete input length.
    ///
    /// Returns the ceil-rounded number of output frames expected for `input_frames`.
    pub fn output_frame_count(&self, input_frames: usize) -> usize {
        (input_frames * self.config.output_sample_rate).div_ceil(self.config.input_sample_rate)
    }

    /// Output samples needed for a complete interleaved input length.
    ///
    /// This validates that `input_samples` is divisible by channel count, then returns the number
    /// of output samples needed. This can be used to size the output buffer for the entire input
    /// stream.
    ///
    /// Returns `Error::MalformedInputLength` when `input_samples` is not channel-aligned.
    pub fn output_sample_count(&self, input_samples: usize) -> Result<usize, Error> {
        if !input_samples.is_multiple_of(self.config.channels) {
            return Err(Error::MalformedInputLength {
                channels: self.config.channels,
                samples: input_samples,
            });
        }

        let input_frames = input_samples / self.config.channels;
        Ok(self.output_frame_count(input_frames) * self.config.channels)
    }

    /// Resamples a complete interleaved input buffer and returns all output samples.
    ///
    /// This is a convenience wrapper around the streaming API.
    pub fn process_all(&mut self, input: &[T]) -> Result<Vec<T>, Error> {
        let expected_samples = self.output_sample_count(input.len())?;

        let mut output = Vec::with_capacity(expected_samples);
        let mut offset = 0;
        let channels = self.config.channels;
        let input_buffer_size = self.input_buffer_size();
        let output_buffer_size = self.output_buffer_size();
        let mut chunk_output_buffer = vec![T::zero(); output_buffer_size];

        while offset + input_buffer_size <= input.len() {
            let chunk = &input[offset..offset + input_buffer_size];
            let frames = chunk.len() / channels;
            self.stage_input_channels(chunk);

            #[cfg(feature = "rayon")]
            let written = {
                let written_per_channel = self
                    .cores
                    .par_iter_mut()
                    .zip(self.input_staging.par_iter())
                    .zip(self.output_staging.par_iter_mut())
                    .map(|((core, core_input), core_output)| {
                        core.process_chunk(&core_input[..frames], core_output, false)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                written_per_channel.into_iter().sum()
            };

            #[cfg(not(feature = "rayon"))]
            let written = {
                let mut total_written = 0;
                for channel_idx in 0..channels {
                    let core = &mut self.cores[channel_idx];
                    let core_input = &self.input_staging[channel_idx][..frames];
                    let core_output = &mut self.output_staging[channel_idx];
                    total_written += core.process_chunk(core_input, core_output, false)?;
                }
                total_written
            };

            if channels == 1 {
                chunk_output_buffer[..written].copy_from_slice(&self.output_staging[0][..written]);
            } else {
                let written_per_channel = written / channels;
                for frame_idx in 0..written_per_channel {
                    for channel_idx in 0..channels {
                        chunk_output_buffer[frame_idx * channels + channel_idx] =
                            self.output_staging[channel_idx][frame_idx];
                    }
                }
            }

            output.extend_from_slice(&chunk_output_buffer[..written]);
            offset += input_buffer_size;
        }

        let final_chunk = &input[offset..];
        let final_frames = final_chunk.len() / channels;
        self.stage_input_channels(final_chunk);

        #[cfg(feature = "rayon")]
        let written = {
            let written_per_channel = self
                .cores
                .par_iter_mut()
                .zip(self.input_staging.par_iter())
                .zip(self.output_staging.par_iter_mut())
                .map(|((core, core_input), core_output)| {
                    core.process_chunk(&core_input[..final_frames], core_output, true)
                })
                .collect::<Result<Vec<_>, _>>()?;

            written_per_channel.into_iter().sum()
        };

        #[cfg(not(feature = "rayon"))]
        let written = {
            let mut total_written = 0;
            for channel_idx in 0..channels {
                let core = &mut self.cores[channel_idx];
                let core_input = &self.input_staging[channel_idx][..final_frames];
                let core_output = &mut self.output_staging[channel_idx];
                total_written += core.process_chunk(core_input, core_output, true)?;
            }
            total_written
        };

        if channels == 1 {
            chunk_output_buffer[..written].copy_from_slice(&self.output_staging[0][..written]);
        } else {
            let written_per_channel = written / channels;
            for frame_idx in 0..written_per_channel {
                for channel_idx in 0..channels {
                    chunk_output_buffer[frame_idx * channels + channel_idx] =
                        self.output_staging[channel_idx][frame_idx];
                }
            }
        }

        output.extend_from_slice(&chunk_output_buffer[..written]);

        let written = self.finalize(&mut chunk_output_buffer)?;
        output.extend_from_slice(&chunk_output_buffer[..written]);

        Ok(output)
    }

    /// Process multiple independent tracks in parallel.
    ///
    /// Each input slice is treated as its own stream with no inter-track context.
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
    /// meant to be played back to back with no gaps.
    ///
    /// For each track:
    /// - `pre` is filled from the tail of the previous track when one exists.
    /// - `post` is filled from the head of the next track when one exists.
    pub fn batch_gapless(&self, inputs: &[&[T]]) -> Result<Vec<Vec<T>>, Error>
    where
        T: Send + Sync,
    {
        let config = self.config.clone();
        let context_samples = self.input_buffer_size();
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
    pub fn process_chunk(&mut self, input: &[T], output: &mut [T]) -> Result<usize, Error> {
        // Output buffer must be at least the size of the output chunk (but can be larger).
        self.ensure_output_buffer_size(output)?;
        self.ensure_input_buffer_size(input, false)?;

        // Process the chunk.
        self.process_chunk_inner(input, output, false)
    }

    /// Processes the final chunk, which may be shorter than the regular chunk size.
    ///
    /// Call this exactly once at end-of-stream for the trailing partial chunk (it may be empty if
    /// input length is an exact multiple of `input_chunk_frames()`). After this call, no further
    /// chunk-processing calls should be made; call `finalize()` to drain remaining delayed tail.
    /// Size `output` to at least `output_buffer_size()`.
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn process_chunk_final(&mut self, input: &[T], output: &mut [T]) -> Result<usize, Error> {
        // Output buffer must be at least the size of the output chunk (but can be larger).
        self.ensure_output_buffer_size(output)?;
        self.ensure_input_buffer_size(input, true)?;

        // Process the chunk.
        self.process_chunk_inner(input, output, true)
    }

    fn process_chunk_inner(&mut self, input: &[T], output: &mut [T], is_final: bool) -> Result<usize, Error> {
        let frames = input.len() / self.config.channels;

        // Deinterleave the input samples into the input staging buffers (one input buffer per channel).
        self.stage_input_channels(input);

        let mut total_written = 0;

        // For each channel, process the chunk in the core
        for channel_idx in 0..self.config.channels {
            let core = &mut self.cores[channel_idx];
            let core_input = &self.input_staging[channel_idx][..frames];
            let core_output = &mut self.output_staging[channel_idx];
            total_written += core.process_chunk(core_input, core_output, is_final)?;
        }

        // If there is only one channel, copy the output directly to the output buffer.
        if self.config.channels == 1 {
            output[..total_written].copy_from_slice(&self.output_staging[0][..total_written]);
            return Ok(total_written);
        }

        // Re-interleave the output samples into the output buffer.
        let written_per_channel = total_written / self.config.channels;
        for frame_idx in 0..written_per_channel {
            for channel_idx in 0..self.config.channels {
                output[frame_idx * self.config.channels + channel_idx] = self.output_staging[channel_idx][frame_idx];
            }
        }

        // Return total written
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
    pub fn finalize(&mut self, output: &mut [T]) -> Result<usize, Error> {
        // Ensure the output buffer is large enough.
        self.ensure_output_buffer_size(output)?;

        let mut total_written = 0;

        // Finalize each core, capturing the output.
        for channel_idx in 0..self.config.channels {
            let core = &mut self.cores[channel_idx];
            let core_output = &mut self.output_staging[channel_idx];
            total_written += core.finalize(core_output)?;
        }

        // If there is only one channel, copy the output directly to the output buffer.
        if self.config.channels == 1 {
            output[..total_written].copy_from_slice(&self.output_staging[0][..total_written]);
            return Ok(total_written);
        }

        // Re-interleave the output samples into the output buffer.
        let written_per_channel = total_written / self.config.channels;
        for frame_idx in 0..written_per_channel {
            for channel_idx in 0..self.config.channels {
                output[frame_idx * self.config.channels + channel_idx] = self.output_staging[channel_idx][frame_idx];
            }
        }

        // Return total written
        Ok(total_written)
    }

    /// Validates channel alignment and chunk length for an `input` slice.
    ///
    /// Non-final calls must provide exactly `input_buffer_size()` samples. Final calls may provide
    /// fewer samples but never more than `input_buffer_size()`.
    fn ensure_input_buffer_size(&self, input: &[T], is_final: bool) -> Result<(), Error> {
        if !input.len().is_multiple_of(self.config.channels) {
            return Err(Error::MalformedInputLength {
                channels: self.config.channels,
                samples: input.len(),
            });
        }

        let frames = input.len() / self.config.channels;
        let expected_frames = self.derived.input_chunk_frames;

        // Non-final chunks must match the fixed chunk size exactly. Final chunks may be shorter.
        if (!is_final && frames != expected_frames) || (is_final && frames > expected_frames) {
            return Err(Error::WrongChunkLength {
                expected: self.input_buffer_size(),
                actual: input.len(),
            });
        }

        Ok(())
    }

    /// Verifies `output` has capacity for at least one produced chunk.
    ///
    /// Callers should allocate at least `output_buffer_size()` samples before processing.
    fn ensure_output_buffer_size(&self, output: &[T]) -> Result<(), Error> {
        let expected = self.output_buffer_size();
        if output.len() < expected {
            return Err(Error::InsufficientOutputBuffer {
                expected,
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn normalize_context(&self, context: Vec<T>) -> Result<Option<Vec<T>>, Error> {
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

    /// Returns up to `max_samples` aligned head samples for interleaved context.
    fn batch_context_head(input: &[T], max_samples: usize, channels: usize) -> Vec<T> {
        let aligned_input_len = input.len() - (input.len() % channels);
        let mut take = aligned_input_len.min(max_samples);
        take -= take % channels;
        input[..take].to_vec()
    }

    /// Returns up to `max_samples` aligned tail samples for interleaved context.
    fn batch_context_tail(input: &[T], max_samples: usize, channels: usize) -> Vec<T> {
        let aligned_input_len = input.len() - (input.len() % channels);
        let mut take = aligned_input_len.min(max_samples);
        take -= take % channels;
        let start = aligned_input_len - take;
        input[start..aligned_input_len].to_vec()
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

    /// Computes ceil-rounded output frames for rational-rate conversion sizing.
    fn output_frame_count(input_frames: usize, input_rate: usize, output_rate: usize) -> usize {
        (input_frames * output_rate).div_ceil(input_rate)
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
            resampler.pre(pre.to_vec()).unwrap();
        }
        if let Some(post) = post {
            resampler.post(post.to_vec()).unwrap();
        }

        let input_buffer_size = resampler.input_buffer_size();
        let full_chunks = input.len() / input_buffer_size;
        let has_partial = !input.len().is_multiple_of(input_buffer_size);
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut output = vec![0.0; output_blocks * resampler.output_buffer_size()];
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
        let input = vec![0.0; resampler.input_chunk_frames()];

        let output = resampler.process_all(&input).unwrap();

        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn same_rate_passthrough_preserves_samples() {
        let mut resampler = Ardftsrc::new(mono_config(48_000, 48_000)).unwrap();
        let input: Vec<f32> = (0..resampler.input_chunk_frames() * 2 + 7)
            .map(|frame| (frame as f32 * 0.013).cos())
            .collect();

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(output, input);
        assert_eq!(resampler.output_delay_frames(), 0);
    }

    #[test]
    fn impulse_output_is_finite() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let mut input = vec![0.0; resampler.input_chunk_frames()];
        input[0] = 1.0;

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(output.len(), output_frame_count(input.len(), 44_100, 48_000));
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn first_chunk_lpc_start_edge_is_finite() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..resampler.input_chunk_frames())
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();
        let mut output = vec![0.0; resampler.output_chunk_frames()];

        let written = resampler.process_chunk(&input, &mut output).unwrap();

        assert_eq!(
            written,
            resampler.output_chunk_frames() - resampler.output_delay_frames()
        );
        assert!(output[..written].iter().all(|sample| sample.is_finite()));
        assert!(output[..written].iter().any(|sample| sample.abs() > 1e-6));
    }

    #[test]
    fn short_final_chunk_lpc_stop_edge_is_finite_and_bounded() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input_frames = resampler.input_chunk_frames() / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.02).sin() * 0.1)
            .collect();

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(output.len(), output_frame_count(input_frames, 44_100, 48_000));
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().all(|sample| sample.abs() < 1.0));
    }

    #[test]
    fn output_sample_count_matches_frame_count_conversion() {
        let resampler = Ardftsrc::<f32>::new(stereo_config(44_100, 48_000)).unwrap();
        let input_samples = resampler.input_buffer_size() * 2 + 14;
        let input_frames = input_samples / resampler.config().channels;

        let expected = resampler.output_frame_count(input_frames) * resampler.config().channels;
        assert_eq!(resampler.output_sample_count(input_samples).unwrap(), expected);
    }

    #[test]
    fn output_sample_count_rejects_non_interleaved_length() {
        let resampler = Ardftsrc::<f32>::new(stereo_config(44_100, 48_000)).unwrap();

        assert!(matches!(
            resampler.output_sample_count(3),
            Err(Error::MalformedInputLength {
                channels: 2,
                samples: 3
            })
        ));
    }

    #[test]
    fn stereo_wrapper_matches_channel_core_outputs() {
        let config = stereo_config(44_100, 48_000);
        let mut wrapper = Ardftsrc::<f32>::new(config.clone()).unwrap();
        let input_frames = wrapper.input_chunk_frames() * 2 + 37;
        let input: Vec<f32> = (0..input_frames)
            .flat_map(|frame| {
                let t = frame as f32;
                [(t * 0.01).sin() * 0.25, (t * 0.017).cos() * 0.2]
            })
            .collect();

        let context_frames = wrapper.input_chunk_frames() / 2;
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

        wrapper.pre(pre.clone()).unwrap();
        wrapper.post(post.clone()).unwrap();
        let wrapped_output = wrapper.process_all(&input).unwrap();

        let mono_config = Config {
            channels: 1,
            ..config.clone()
        };
        let derived = mono_config.derive_config::<f32>().unwrap();
        let mut left_core = ArdftsrcCore::<f32>::new(derived.clone()).unwrap();
        let mut right_core = ArdftsrcCore::<f32>::new(derived).unwrap();

        left_core.pre(deinterleave_channel(&pre, 2, 0)).unwrap();
        right_core.pre(deinterleave_channel(&pre, 2, 1)).unwrap();
        left_core.post(deinterleave_channel(&post, 2, 0)).unwrap();
        right_core.post(deinterleave_channel(&post, 2, 1)).unwrap();

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
            resampler.pre(vec![0.0; 3]),
            Err(Error::MalformedInputLength {
                channels: 2,
                samples: 3
            })
        ));
        assert!(matches!(
            resampler.post(vec![0.0; 3]),
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
        let input: Vec<f32> = (0..with_zero_pre.input_chunk_frames())
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();
        let mut out_zero = vec![0.0; with_zero_pre.output_chunk_frames()];
        let mut out_one = vec![0.0; with_one_pre.output_chunk_frames()];
        with_zero_pre
            .pre(vec![0.0; with_zero_pre.input_chunk_frames()])
            .unwrap();
        with_one_pre.pre(vec![1.0; with_one_pre.input_chunk_frames()]).unwrap();

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
        let input: Vec<f32> = (0..with_zero_post.input_chunk_frames())
            .map(|frame| (frame as f32 * 0.015).cos() * 0.125)
            .collect();
        let mut chunk_out = vec![0.0; with_zero_post.output_chunk_frames()];
        let mut flush_zero = vec![0.0; with_zero_post.output_chunk_frames()];
        let mut flush_one = vec![0.0; with_one_post.output_chunk_frames()];
        with_zero_post
            .post(vec![0.0; with_zero_post.input_chunk_frames()])
            .unwrap();
        with_one_post
            .post(vec![1.0; with_one_post.input_chunk_frames()])
            .unwrap();

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
        let input: Vec<f32> = (0..resampler.input_chunk_frames())
            .map(|frame| (frame as f32 * 0.015).cos() * 0.125)
            .collect();
        let mut output = vec![0.0; resampler.output_chunk_frames()];

        let first_written = resampler.process_chunk(&input, &mut output).unwrap();
        let flush_written = resampler.finalize(&mut output).unwrap();

        assert_eq!(first_written + flush_written, resampler.output_chunk_frames());
        assert!(output[..flush_written].iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn streaming_and_offline_paths_match() {
        let config = mono_config(44_100, 48_000);
        let mut offline = Ardftsrc::new(config.clone()).unwrap();
        let mut streaming = Ardftsrc::new(config).unwrap();
        let input_frames = streaming.input_chunk_frames() * 2 + streaming.input_chunk_frames() / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();

        let offline_output = offline.process_all(&input).unwrap();

        let full_chunks = input.len() / streaming.input_chunk_frames();
        let has_partial = !input.len().is_multiple_of(streaming.input_chunk_frames());
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut streaming_output = vec![0.0; output_blocks * streaming.output_chunk_frames()];
        let mut written = 0;
        let chunk_len = streaming.input_chunk_frames();
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
        let context_len = full_resampler.input_buffer_size();
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
        let context_len = full_resampler.input_buffer_size();
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
        let first_input: Vec<f32> = (0..(reused.input_chunk_frames() * 2 + 11))
            .map(|frame| (frame as f32 * 0.009).sin() * 0.2)
            .collect();
        let second_input: Vec<f32> = (0..(reused.input_chunk_frames() + 7))
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
        let mut input = vec![0.0; resampler.input_buffer_size()];
        input[0] = 1.0;

        let output = resampler.process_all(&input).unwrap();

        assert_eq!(
            output.len(),
            output_frame_count(resampler.input_chunk_frames(), 44_100, 48_000) * 2
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
        let input_frames = streaming.input_chunk_frames() * 2 + streaming.input_chunk_frames() / 3;
        let mut input = Vec::with_capacity(input_frames * 2);
        for frame in 0..input_frames {
            input.push((frame as f32 * 0.01).sin() * 0.25);
            input.push((frame as f32 * 0.017).cos() * 0.125);
        }

        let offline_output = offline.process_all(&input).unwrap();

        let input_buffer_size = streaming.input_buffer_size();
        let full_chunks = input.len() / input_buffer_size;
        let has_partial = !input.len().is_multiple_of(input_buffer_size);
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut streaming_output = vec![0.0; output_blocks * streaming.output_buffer_size()];
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
        let input = vec![0.0; resampler.input_chunk_frames() - 1];

        assert!(resampler.process_chunk(&input, &mut []).is_err());
    }

    #[test]
    fn too_small_output_does_not_advance_stream_state() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; resampler.input_chunk_frames()];
        let mut too_small = vec![0.0; resampler.output_chunk_frames() - 1];

        assert!(matches!(
            resampler.process_chunk(&input, &mut too_small),
            Err(Error::InsufficientOutputBuffer { .. })
        ));
        assert_eq!(resampler.input_sample_count(), 0);

        let mut output = vec![0.0; resampler.output_chunk_frames()];
        let expected = resampler.output_chunk_frames() - resampler.output_delay_frames();
        assert_eq!(resampler.process_chunk(&input, &mut output).unwrap(), expected);
        assert_eq!(resampler.input_sample_count(), input.len());
    }

    #[test]
    fn input_sample_count_tracks_full_and_partial_stream_input() {
        let mut resampler = Ardftsrc::new(stereo_config(44_100, 48_000)).unwrap();
        let mut output = vec![0.0; resampler.output_buffer_size()];
        let full = vec![0.0; resampler.input_buffer_size()];
        let partial = vec![0.0; 10];

        assert_eq!(resampler.input_sample_count(), 0);
        let _ = resampler.process_chunk(&full, &mut output).unwrap();
        assert_eq!(resampler.input_sample_count(), full.len());
        let _ = resampler.process_chunk_final(&partial, &mut output).unwrap();
        assert_eq!(resampler.input_sample_count(), full.len() + partial.len());
    }

    #[test]
    fn too_small_finish_output_does_not_mark_flushed() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; resampler.input_chunk_frames()];
        let mut first_output = vec![0.0; resampler.output_chunk_frames()];
        resampler.process_chunk(&input, &mut first_output).unwrap();

        let mut too_small = vec![0.0; resampler.output_delay_frames() - 1];

        assert!(matches!(
            resampler.finalize(&mut too_small),
            Err(Error::InsufficientOutputBuffer { .. })
        ));

        let mut output = vec![0.0; resampler.output_chunk_frames()];
        assert_eq!(
            resampler.finalize(&mut output).unwrap(),
            resampler.output_delay_frames()
        );
    }

    #[test]
    fn first_chunk_is_delay_trimmed_and_flushes_tail() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; resampler.input_chunk_frames()];
        let mut output = vec![0.0; resampler.output_chunk_frames()];

        let first_written = resampler.process_chunk(&input, &mut output).unwrap();
        let flush_written = resampler.finalize(&mut output).unwrap();

        assert_eq!(first_written + flush_written, resampler.output_chunk_frames());
        assert_eq!(
            first_written,
            resampler.output_chunk_frames() - resampler.output_delay_frames()
        );
        assert_eq!(flush_written, resampler.output_delay_frames());
    }

    #[test]
    fn finalize_without_explicit_final_chunk_caps_to_expected_total() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; resampler.input_chunk_frames()];
        let input_samples = input.len() * 2;
        let expected_total = resampler.output_sample_count(input_samples).unwrap();
        let mut output = vec![0.0; resampler.output_chunk_frames()];

        let first_written = resampler.process_chunk(&input, &mut output).unwrap();
        let second_written = resampler.process_chunk(&input, &mut output).unwrap();
        let flush_written = resampler.finalize(&mut output).unwrap();

        assert_eq!(first_written + second_written + flush_written, expected_total);
    }

    #[test]
    fn empty_finish_does_not_emit_extra_block() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; resampler.input_chunk_frames()];
        let mut output = vec![0.0; resampler.output_chunk_frames()];

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
        let mut output = vec![0.0; resampler.output_chunk_frames()];

        assert_eq!(resampler.process_chunk_final(&[], &mut output).unwrap(), 0);
        assert_eq!(resampler.finalize(&mut output).unwrap(), 0);
    }

    #[test]
    fn second_finish_after_reset_returns_zero_without_input() {
        let mut resampler = Ardftsrc::new(mono_config(44_100, 48_000)).unwrap();

        let mut output = vec![0.0; resampler.output_chunk_frames()];
        assert_eq!(resampler.finalize(&mut output).unwrap(), 0);
        assert_eq!(resampler.finalize(&mut output).unwrap(), 0);
    }

    #[test]
    fn batch_test() {
        let config = mono_config(44_100, 48_000);
        let driver = Ardftsrc::new(config.clone()).unwrap();
        let chunk = driver.input_chunk_frames();
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
