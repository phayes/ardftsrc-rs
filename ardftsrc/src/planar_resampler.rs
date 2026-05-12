use num_traits::Float;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use realfft::FftNum;

use crate::{Config, Error, PlanarVecs, config::DerivedConfig, core::ArdftsrcCore};

pub struct PlanarResampler<T = f64>
where
    T: Float + FftNum,
{
    config: Config,
    derived: DerivedConfig<T>,
    pub(crate) cores: Vec<ArdftsrcCore<T>>,
}

impl<T> PlanarResampler<T>
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
            .collect();

        Ok(Self { config, derived, cores })
    }

    /// Returns the configuration this instance was built with.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the total number of interleaved input samples processed.
    #[inline]
    pub fn input_sample_processed(&self) -> usize {
        self.cores.iter().map(ArdftsrcCore::input_sample_processed).sum()
    }

    /// Returns the total number of interleaved output samples processed.
    #[inline]
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
    #[inline]
    pub fn output_chunk_size(&self) -> usize {
        self.derived.output_chunk_frames * self.config.channels
    }

    /// Returns algorithmic latency to trim/flush, or zero in same-rate passthrough mode.
    #[must_use]
    #[inline]
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
    #[inline]
    pub fn expected_output_size(&self, input_size: usize) -> usize {
        (input_size * self.config.output_sample_rate).div_ceil(self.config.input_sample_rate)
    }

    /// Resamples a complete interleaved input buffer and returns all output samples.
    ///
    /// This is a convenience wrapper around the streaming API.
    ///
    /// When the `rayon` feature is enabled, each channel is processed in parallel.
    pub fn process_all<'a>(&mut self, input: &[&[T]]) -> Result<PlanarVecs<T>, Error>
    where
        T: Send + Sync,
    {
        // If it's passthrough, just return the input
        if self.is_passthrough() {
            let output: Vec<Vec<T>> = input.iter().map(|channel| channel.to_vec()).collect();
            return PlanarVecs::new(output);
        }

        #[cfg(feature = "rayon")]
        let output: Vec<Vec<T>> = self
            .cores
            .par_iter_mut()
            .zip(input.par_iter())
            .map(|(core, channel)| core.process_all(channel))
            .collect::<Result<_, _>>()?;

        #[cfg(not(feature = "rayon"))]
        let output: Vec<Vec<T>> = {
            let mut output = Vec::with_capacity(self.config.channels);
            for (core, channel) in self.cores.iter_mut().zip(input.iter()) {
                output.push(core.process_all(channel)?);
            }
            output
        };

        PlanarVecs::new(output)
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

    /// Processes an interleaved chunk.
    ///
    /// `input` must contain exactly `input_buffer_size()` samples (all channels, interleaved),
    /// and `output` should provide at least `output_buffer_size()` capacity.
    ///
    /// The method returns the actual sample count written for this chunk, which may be smaller
    /// (for example while startup delay is being trimmed).
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn process_chunk<'a>(&mut self, input: &[&[T]], output: &mut [&mut [T]]) -> Result<usize, Error> {
        // Output buffer must be at least the size of the output chunk (but can be larger).
        self.ensure_input_buffer_shape(input, false)?;
        self.ensure_output_buffer_shape(output)?;

        if self.is_passthrough() {
            for (channel_input, channel_output) in input.iter().zip(output.iter_mut()) {
                channel_output[..channel_input.len()].copy_from_slice(channel_input);
            }
            return Ok(input[0].len());
        }

        // Process the chunk.
        let total_written = Self::process_chunk_inner(&mut self.cores, input, output, false)?;

        Ok(total_written)
    }

    /// Processes the final chunk, which may be shorter than the regular chunk size.
    ///
    /// Call this exactly once at end-of-stream for the trailing partial chunk (it may be empty if
    /// input length is an exact multiple of `input_buffer_size()`). After this call, no further
    /// chunk-processing calls should be made; call `finalize()` to drain remaining delayed tail.
    /// Size `output` to at least `output_buffer_size()`.
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn process_chunk_final<'a>(&mut self, input: &[&[T]], output: &mut [&mut [T]]) -> Result<usize, Error> {
        // Output buffer must be at least the size of the output chunk (but can be larger).
        self.ensure_input_buffer_shape(input, true)?;
        self.ensure_output_buffer_shape(output)?;

        if self.is_passthrough() {
            for (channel_input, channel_output) in input.iter().zip(output.iter_mut()) {
                channel_output[..channel_input.len()].copy_from_slice(channel_input);
            }
            return Ok(input[0].len());
        }

        // Process the chunk.
        let total_written = Self::process_chunk_inner(&mut self.cores, input, output, true)?;

        Ok(total_written)
    }

    #[inline]
    fn process_chunk_inner(
        cores: &mut [ArdftsrcCore<T>],
        input: &[&[T]],
        output: &mut [&mut [T]],
        is_final: bool,
    ) -> Result<usize, Error> {
        let mut total_written = 0;

        for (channel_idx, (channel_input, channel_output)) in input.iter().zip(output.iter_mut()).enumerate() {
            let core_output = cores[channel_idx].process_chunk(channel_input.as_ref(), is_final)?;

            debug_assert!(channel_output.len() >= core_output.len());

            channel_output[..core_output.len()].copy_from_slice(&core_output);

            total_written += core_output.len();
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
    /// Returns the number of samples written to the output buffer. Divide by the channel count to get the number of samples written per channel.
    pub fn finalize<'a>(&mut self, output: &mut [&mut [T]]) -> Result<usize, Error> {
        // Ensure the output buffer is large enough.
        self.ensure_output_buffer_shape(output)?;

        if self.is_passthrough() {
            return Ok(0);
        }

        // For each channel, process the chunk in the core.
        let mut total_written = 0;
        for (channel_idx, core) in self.cores.iter_mut().enumerate() {
            let core_output = core.finalize()?;
            output[channel_idx][..core_output.len()].copy_from_slice(&core_output);
            total_written += core_output.len();
        }

        Ok(total_written)
    }

    /// Validates channel alignment and chunk length for an `input` slice.
    ///
    /// Non-final calls must provide exactly `input_buffer_size()` samples. Final calls may provide
    /// fewer samples but never more than `input_buffer_size()`.
    #[inline]
    fn ensure_input_buffer_shape<'a>(&self, input: &[&[T]], is_final: bool) -> Result<(), Error> {
        if input.len() != self.config.channels {
            return Err(Error::WrongChannelCount {
                expected: self.config.channels,
                actual: input.len(),
            });
        }

        if !is_final {
            if let Some(channel) = input
                .iter()
                .find(|channel| channel.len() != self.derived.input_chunk_frames)
            {
                return Err(Error::WrongFrameCount {
                    expected: self.derived.input_chunk_frames,
                    actual: channel.len(),
                });
            }
        } else if let Some(channel) = input
            .iter()
            .find(|channel| channel.len() > self.derived.input_chunk_frames)
        {
            return Err(Error::WrongFrameCount {
                expected: self.derived.input_chunk_frames,
                actual: channel.len(),
            });
        }

        Ok(())
    }

    /// Verifies `output` has capacity for at least one produced chunk.
    ///
    /// Callers should allocate at least `output_buffer_size()` samples before processing.
    #[inline]
    fn ensure_output_buffer_shape<'a>(&self, output: &[&mut [T]]) -> Result<(), Error> {
        if output.len() != self.config.channels {
            return Err(Error::WrongChannelCount {
                expected: self.config.channels,
                actual: output.len(),
            });
        }

        if let Some(channel) = output
            .iter()
            .find(|channel| channel.len() < self.derived.output_chunk_frames)
        {
            return Err(Error::InsufficientOutputBuffer {
                expected: self.derived.output_chunk_frames,
                actual: channel.len(),
            });
        }

        Ok(())
    }

    /// Returns true when rates match and no FFT-domain processing has been requested.
    #[inline]
    fn is_passthrough(&self) -> bool {
        self.config.input_sample_rate == self.config.output_sample_rate
            && (self.config.phase == 0.0 || self.config.phase_intensity == 0.0)
    }

    /// Sets previous-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the last chunk of the previous track.
    ///
    /// The adapter must report the same channel count as this resampler.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the end of the previous track.
    /// - Query chunk size with `input_buffer_size()`.
    ///
    /// Shorter buffers are still valid: any missing start context falls back to LPC
    /// extrapolation. Longer buffers are truncated to fit.
    pub fn pre<'a>(&mut self, pre: Vec<Vec<T>>) -> Result<(), Error> {
        if pre.len() != self.config.channels {
            return Err(Error::WrongChannelCount {
                expected: self.config.channels,
                actual: pre.len(),
            });
        }

        for (core, samples) in self.cores.iter_mut().zip(pre.into_iter()) {
            let mut samples = samples;
            if samples.len() > self.derived.input_chunk_frames {
                samples = samples.split_off(samples.len() - self.derived.input_chunk_frames);
            }
            core.pre(samples);
        }
        Ok(())
    }

    /// Sets next-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the first chunk of the next track.
    ///
    /// You may call this at any time while the current stream is still active. It must be called
    /// before "process_chunk_final(...)".
    ///
    /// This is useful for live gapless handoff: while track A is streaming, once track B is known you
    /// can call `post(...)` on track A with B's head samples so A's stop-edge uses real next-track
    /// context.
    ///
    /// The adapter must report the same channel count as this resampler.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the start of the next track.
    /// - Query chunk size with `input_buffer_size()`.
    ///
    /// Shorter buffers are still valid: any missing stop context falls back to LPC
    /// extrapolation.
    pub fn post<'a>(&mut self, post: Vec<Vec<T>>) -> Result<(), Error> {
        if post.len() != self.config.channels {
            return Err(Error::WrongChannelCount {
                expected: self.config.channels,
                actual: post.len(),
            });
        }

        for (core, samples) in self.cores.iter_mut().zip(post.into_iter()) {
            let mut samples = samples;
            if samples.len() > self.derived.input_chunk_frames {
                samples.truncate(self.derived.input_chunk_frames);
            }
            core.post(samples);
        }
        Ok(())
    }

    /// Process multiple independent tracks.
    ///
    /// Each input slice is treated as its own stream with no inter-track context. See
    /// `batch_gapless()` for gapless processing of multiple tracks.
    ///
    /// Enable the `rayon` feature for parallel processing.
    pub fn batch(&self, inputs: Vec<PlanarVecs<T>>) -> Result<Vec<PlanarVecs<T>>, Error>
    where
        T: Send + Sync,
    {
        #[cfg(feature = "rayon")]
        {
            inputs
                .into_par_iter()
                .map(|input| {
                    let channel_outputs = Self::batch_process_track(self.config.clone(), input.as_slice(), None, None)?;
                    PlanarVecs::new(channel_outputs)
                })
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .into_iter()
                .map(|input| {
                    let channel_outputs = Self::batch_process_track(self.config.clone(), input.as_slice(), None, None)?;
                    PlanarVecs::new(channel_outputs)
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
    pub fn batch_gapless(&self, inputs: Vec<PlanarVecs<T>>) -> Result<Vec<PlanarVecs<T>>, Error>
    where
        T: Send + Sync,
    {
        let context_chunk_size = self.input_chunk_size() / self.config.channels;

        #[cfg(feature = "rayon")]
        {
            inputs
                .par_iter()
                .enumerate()
                .map(|(track_idx, input)| {
                    let pre = track_idx
                        .checked_sub(1)
                        .map(|idx| Self::batch_track_tail_context(inputs[idx].as_slice(), context_chunk_size));
                    let post = inputs
                        .get(track_idx + 1)
                        .map(|input| Self::batch_track_head_context(input.as_slice(), context_chunk_size));
                    let channel_outputs = Self::batch_process_track(self.config.clone(), input.as_slice(), pre, post)?;
                    PlanarVecs::new(channel_outputs)
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
                        .map(|idx| Self::batch_track_tail_context(inputs[idx].as_slice(), context_chunk_size));
                    let post = inputs
                        .get(track_idx + 1)
                        .map(|input| Self::batch_track_head_context(input.as_slice(), context_chunk_size));
                    let channel_outputs = Self::batch_process_track(self.config.clone(), input.as_slice(), pre, post)?;
                    PlanarVecs::new(channel_outputs)
                })
                .collect()
        }
    }

    fn batch_process_track(
        config: Config,
        channel_inputs: &[Vec<T>],
        pre: Option<Vec<Vec<T>>>,
        post: Option<Vec<Vec<T>>>,
    ) -> Result<Vec<Vec<T>>, Error>
    where
        T: Send + Sync,
    {
        // Validate the track
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

        // Create a resampler just for this track
        let mut resampler = PlanarResampler::new(config)?;

        // Set the pre and post context if provided
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

        // Process cores for each channel
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

    fn batch_track_tail_context(channel_inputs: &[Vec<T>], context_chunk_size: usize) -> Vec<Vec<T>> {
        channel_inputs
            .iter()
            .map(|channel| {
                let start = channel.len().saturating_sub(context_chunk_size);
                channel[start..].to_vec()
            })
            .collect()
    }

    fn batch_track_head_context(channel_inputs: &[Vec<T>], context_chunk_size: usize) -> Vec<Vec<T>> {
        channel_inputs
            .iter()
            .map(|channel| {
                let end = channel.len().min(context_chunk_size);
                channel[..end].to_vec()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TaperType, core::ArdftsrcCore, test_utils::assert_no_nans};
    use dasp_signal::Signal;

    fn input_chunk_frames(resampler: &PlanarResampler<f32>) -> usize {
        resampler.input_chunk_size() / resampler.config().channels
    }

    fn output_chunk_frames(resampler: &PlanarResampler<f32>) -> usize {
        resampler.output_chunk_size() / resampler.config().channels
    }

    fn deinterleave_samples(samples: &[f32], channels: usize) -> Result<Vec<Vec<f32>>, Error> {
        if !samples.len().is_multiple_of(channels) {
            return Err(Error::MalformedInputLength {
                channels,
                samples: samples.len(),
            });
        }

        let frames = samples.len() / channels;
        let mut planar = vec![Vec::with_capacity(frames); channels];
        for frame in samples.chunks_exact(channels) {
            for (channel_idx, sample) in frame.iter().enumerate() {
                planar[channel_idx].push(*sample);
            }
        }

        Ok(planar)
    }

    fn interleave_written(planar: &[Vec<f32>], written: usize, channels: usize, output: &mut [f32]) {
        assert_eq!(written % channels, 0);
        let written_frames = written / channels;
        for frame_idx in 0..written_frames {
            for channel_idx in 0..channels {
                output[frame_idx * channels + channel_idx] = planar[channel_idx][frame_idx];
            }
        }
    }

    fn process_all_samples(resampler: &mut PlanarResampler<f32>, input: &[f32]) -> Result<Vec<f32>, Error> {
        let channels = resampler.config().channels;
        let input_planar = deinterleave_samples(input, channels)?;
        let input_refs: Vec<_> = input_planar.iter().map(Vec::as_slice).collect();
        let output = resampler.process_all(&input_refs)?.interleave();
        assert_no_nans(&output, "chunk::process_all_samples output");
        Ok(output)
    }

    fn process_chunk_samples(
        resampler: &mut PlanarResampler<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, Error> {
        let channels = resampler.config().channels;
        let input_planar = deinterleave_samples(input, channels)?;
        let mut output_planar = deinterleave_samples(output, channels)?;
        let input_refs: Vec<_> = input_planar.iter().map(Vec::as_slice).collect();
        let mut output_refs: Vec<_> = output_planar.iter_mut().map(Vec::as_mut_slice).collect();
        let written = resampler.process_chunk(&input_refs, &mut output_refs)?;
        interleave_written(&output_planar, written, channels, output);
        assert_no_nans(&output[..written], "chunk::process_chunk_samples output");
        Ok(written)
    }

    fn process_chunk_final_samples(
        resampler: &mut PlanarResampler<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, Error> {
        let channels = resampler.config().channels;
        let input_planar = deinterleave_samples(input, channels)?;
        let mut output_planar = deinterleave_samples(output, channels)?;
        let input_refs: Vec<_> = input_planar.iter().map(Vec::as_slice).collect();
        let mut output_refs: Vec<_> = output_planar.iter_mut().map(Vec::as_mut_slice).collect();
        let written = resampler.process_chunk_final(&input_refs, &mut output_refs)?;
        interleave_written(&output_planar, written, channels, output);
        assert_no_nans(&output[..written], "chunk::process_chunk_final_samples output");
        Ok(written)
    }

    fn finalize_samples_chunk(resampler: &mut PlanarResampler<f32>, output: &mut [f32]) -> Result<usize, Error> {
        let channels = resampler.config().channels;
        let mut output_planar = deinterleave_samples(output, channels)?;
        let mut output_refs: Vec<_> = output_planar.iter_mut().map(Vec::as_mut_slice).collect();
        let written = resampler.finalize(&mut output_refs)?;
        interleave_written(&output_planar, written, channels, output);
        assert_no_nans(&output[..written], "chunk::finalize_samples_chunk output");
        Ok(written)
    }

    fn mono_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config {
            input_sample_rate,
            output_sample_rate,
            channels: 1,
            quality: 64,
            bandwidth: 0.95,
            taper_type: TaperType::Cosine(3.45),
            ..Config::default()
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
        let mut resampler = PlanarResampler::new(config).unwrap();
        let channels = resampler.config().channels;
        if let Some(pre) = pre {
            resampler.pre(deinterleave_samples(pre, channels).unwrap()).unwrap();
        }
        if let Some(post) = post {
            resampler.post(deinterleave_samples(post, channels).unwrap()).unwrap();
        }

        let input_buffer_size = resampler.input_chunk_size();
        let full_chunks = input.len() / input_buffer_size;
        let has_partial = !input.len().is_multiple_of(input_buffer_size);
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut output = vec![0.0; output_blocks * resampler.output_chunk_size()];
        let mut written = 0;
        let mut offset = 0;
        while offset + input_buffer_size <= input.len() {
            written += process_chunk_samples(
                &mut resampler,
                &input[offset..offset + input_buffer_size],
                &mut output[written..],
            )
            .unwrap();
            offset += input_buffer_size;
        }
        written += process_chunk_final_samples(&mut resampler, &input[offset..], &mut output[written..]).unwrap();
        written += finalize_samples_chunk(&mut resampler, &mut output[written..]).unwrap();
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
        let mut offset = 0;
        let input_chunk = core.input_buffer_size();

        while offset + input_chunk <= input.len() {
            let written = core.process_chunk(&input[offset..offset + input_chunk], false).unwrap();
            output.extend_from_slice(written);
            offset += input_chunk;
        }

        let final_input = &input[offset..];
        let written = core.process_chunk(final_input, true).unwrap();
        output.extend_from_slice(written);

        let written = core.finalize().unwrap();
        output.extend_from_slice(written);

        assert_no_nans(&output, "chunk::run_core_process_all output");
        output
    }

    fn deinterleave_channel(samples: &[f32], channels: usize, channel: usize) -> Vec<f32> {
        samples.chunks_exact(channels).map(|frame| frame[channel]).collect()
    }

    #[test]
    fn silence_stays_silent() {
        let mut resampler = PlanarResampler::new(mono_config(48_000, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];

        let output = process_all_samples(&mut resampler, &input).unwrap();

        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn same_rate_passthrough_preserves_samples() {
        let mut resampler = PlanarResampler::new(mono_config(48_000, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&resampler) * 2 + 7)
            .map(|frame| (frame as f32 * 0.013).cos())
            .collect();

        let output = process_all_samples(&mut resampler, &input).unwrap();

        assert_eq!(output, input);
        assert_eq!(resampler.output_delay_frames(), 0);
    }

    #[test]
    fn impulse_output_is_finite() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let mut input = vec![0.0; input_chunk_frames(&resampler)];
        input[0] = 1.0;

        let output = process_all_samples(&mut resampler, &input).unwrap();

        assert_eq!(output.len(), resampler.expected_output_size(input.len()));
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn first_chunk_lpc_start_edge_is_finite() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&resampler))
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let written = process_chunk_samples(&mut resampler, &input, &mut output).unwrap();

        assert_eq!(
            written,
            output_chunk_frames(&resampler) - resampler.output_delay_frames()
        );
        assert!(output[..written].iter().all(|sample| sample.is_finite()));
        assert!(output[..written].iter().any(|sample| sample.abs() > 1e-6));
    }

    #[test]
    fn short_final_chunk_lpc_stop_edge_is_finite_and_bounded() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input_frames = input_chunk_frames(&resampler) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.02).sin() * 0.1)
            .collect();

        let output = process_all_samples(&mut resampler, &input).unwrap();

        assert_eq!(output.len(), resampler.expected_output_size(input_frames));
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().all(|sample| sample.abs() < 1.0));
    }

    #[test]
    fn expected_output_size_matches_frame_count_conversion() {
        let resampler = PlanarResampler::<f32>::new(stereo_config(44_100, 48_000)).unwrap();
        let input_samples = resampler.input_chunk_size() * 2 + 14;
        let input_frames = input_samples / resampler.config().channels;

        let expected_from_frames = resampler.expected_output_size(input_frames) * resampler.config().channels;
        let expected_from_samples = resampler.expected_output_size(input_samples);
        assert_eq!(expected_from_samples, expected_from_frames);
    }

    #[test]
    fn expected_output_size_accepts_non_interleaved_length() {
        let resampler = PlanarResampler::<f32>::new(stereo_config(44_100, 48_000)).unwrap();
        let expected = (3usize * 48_000).div_ceil(44_100);
        assert_eq!(resampler.expected_output_size(3), expected);
    }

    #[test]
    fn stereo_wrapper_matches_channel_core_outputs() {
        let config = stereo_config(44_100, 48_000);
        let mut wrapper = PlanarResampler::<f32>::new(config.clone()).unwrap();
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

        wrapper.pre(deinterleave_samples(&pre, 2).unwrap()).unwrap();
        wrapper.post(deinterleave_samples(&post, 2).unwrap()).unwrap();
        let wrapped_output = process_all_samples(&mut wrapper, &input).unwrap();

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
    fn pre_and_post_reject_wrong_channel_count() {
        let mut resampler = PlanarResampler::<f32>::new(stereo_config(44_100, 48_000)).unwrap();
        let mono_context = [0.0f32; 8];

        assert!(matches!(
            resampler.pre(vec![mono_context.to_vec()]),
            Err(Error::WrongChannelCount { expected: 2, actual: 1 })
        ));
        assert!(matches!(
            resampler.post(vec![mono_context.to_vec()]),
            Err(Error::WrongChannelCount { expected: 2, actual: 1 })
        ));
    }

    #[test]
    fn pre_context_changes_first_chunk_output() {
        let mut with_zero_pre = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let mut with_one_pre = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&with_zero_pre))
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();
        let mut out_zero = vec![0.0; output_chunk_frames(&with_zero_pre)];
        let mut out_one = vec![0.0; output_chunk_frames(&with_one_pre)];
        let pre_zero = vec![0.0; input_chunk_frames(&with_zero_pre)];
        let pre_one = vec![1.0; input_chunk_frames(&with_one_pre)];
        with_zero_pre.pre(vec![pre_zero]).unwrap();
        with_one_pre.pre(vec![pre_one]).unwrap();

        let written_zero = process_chunk_samples(&mut with_zero_pre, &input, &mut out_zero).unwrap();
        let written_one = process_chunk_samples(&mut with_one_pre, &input, &mut out_one).unwrap();

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
        let mut with_zero_post = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let mut with_one_post = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&with_zero_post))
            .map(|frame| (frame as f32 * 0.015).cos() * 0.125)
            .collect();
        let mut chunk_out = vec![0.0; output_chunk_frames(&with_zero_post)];
        let mut flush_zero = vec![0.0; output_chunk_frames(&with_zero_post)];
        let mut flush_one = vec![0.0; output_chunk_frames(&with_one_post)];
        let post_zero = vec![0.0; input_chunk_frames(&with_zero_post)];
        let post_one = vec![1.0; input_chunk_frames(&with_one_post)];
        with_zero_post.post(vec![post_zero]).unwrap();
        with_one_post.post(vec![post_one]).unwrap();

        process_chunk_samples(&mut with_zero_post, &input, &mut chunk_out).unwrap();
        process_chunk_samples(&mut with_one_post, &input, &mut chunk_out).unwrap();
        let written_zero = finalize_samples_chunk(&mut with_zero_post, &mut flush_zero).unwrap();
        let written_one = finalize_samples_chunk(&mut with_one_post, &mut flush_one).unwrap();

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
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input: Vec<f32> = (0..input_chunk_frames(&resampler))
            .map(|frame| (frame as f32 * 0.015).cos() * 0.125)
            .collect();
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let first_written = process_chunk_samples(&mut resampler, &input, &mut output).unwrap();
        let flush_written = finalize_samples_chunk(&mut resampler, &mut output).unwrap();

        assert_eq!(first_written + flush_written, output_chunk_frames(&resampler));
        assert!(output[..flush_written].iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn streaming_and_offline_paths_match() {
        let config = mono_config(44_100, 48_000);
        let mut offline = PlanarResampler::new(config.clone()).unwrap();
        let mut streaming = PlanarResampler::new(config).unwrap();
        let input_frames = input_chunk_frames(&streaming) * 2 + input_chunk_frames(&streaming) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.01).sin() * 0.25)
            .collect();

        let offline_output = process_all_samples(&mut offline, &input).unwrap();

        let full_chunks = input.len() / input_chunk_frames(&streaming);
        let has_partial = !input.len().is_multiple_of(input_chunk_frames(&streaming));
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut streaming_output = vec![0.0; output_blocks * output_chunk_frames(&streaming)];
        let mut written = 0;
        let chunk_len = input_chunk_frames(&streaming);
        let mut offset = 0;
        while offset + chunk_len <= input.len() {
            written += process_chunk_samples(
                &mut streaming,
                &input[offset..offset + chunk_len],
                &mut streaming_output[written..],
            )
            .unwrap();
            offset += chunk_len;
        }
        written +=
            process_chunk_final_samples(&mut streaming, &input[offset..], &mut streaming_output[written..]).unwrap();
        written += finalize_samples_chunk(&mut streaming, &mut streaming_output[written..]).unwrap();
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
        let mut full_resampler = PlanarResampler::new(config.clone()).unwrap();
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

        let full_output = process_all_samples(&mut full_resampler, &input).unwrap();

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
        let mut full_resampler = PlanarResampler::new(config.clone()).unwrap();
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

        let full_output = process_all_samples(&mut full_resampler, &input).unwrap();

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
        let mut reused = PlanarResampler::new(config.clone()).unwrap();
        let mut reference = PlanarResampler::new(config).unwrap();
        let first_input: Vec<f32> = (0..(input_chunk_frames(&reused) * 2 + 11))
            .map(|frame| (frame as f32 * 0.009).sin() * 0.2)
            .collect();
        let second_input: Vec<f32> = (0..(input_chunk_frames(&reused) + 7))
            .map(|frame| (frame as f32 * 0.013).cos() * 0.15)
            .collect();

        let first_reused = process_all_samples(&mut reused, &first_input).unwrap();
        let first_reference = process_all_samples(&mut reference, &first_input).unwrap();
        assert_eq!(first_reused.len(), first_reference.len());
        for (left, right) in first_reused.iter().zip(first_reference.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }

        let second_reused = process_all_samples(&mut reused, &second_input).unwrap();
        let second_reference = process_all_samples(&mut reference, &second_input).unwrap();
        assert_eq!(second_reused.len(), second_reference.len());
        for (left, right) in second_reused.iter().zip(second_reference.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn batch_gapless_matches_manual_pre_post() {
        let config = mono_config(44_100, 48_000);
        let batch = PlanarResampler::<f32>::new(config.clone()).unwrap();
        let context_chunk_size = batch.input_chunk_size() / batch.config().channels;
        let track_frames = context_chunk_size * 2 + 17;

        let tracks: Vec<PlanarVecs<f32>> = (0..3)
            .map(|track_idx| {
                let channel = (0..track_frames)
                    .map(|frame| {
                        let continuous_frame = track_idx * track_frames + frame;
                        (continuous_frame as f32 * 0.017).sin() * 0.25
                    })
                    .collect();
                PlanarVecs::new(vec![channel]).unwrap()
            })
            .collect();

        let outputs = batch.batch_gapless(tracks.clone()).unwrap();

        for (track_idx, (input, output)) in tracks.iter().zip(outputs.iter()).enumerate() {
            let mut resampler = PlanarResampler::new(config.clone()).unwrap();
            if let Some(previous) = track_idx.checked_sub(1).map(|idx| tracks[idx].get_channel(0).unwrap()) {
                let pre_context = previous[previous.len().saturating_sub(context_chunk_size)..].to_vec();
                resampler.pre(vec![pre_context]).unwrap();
            }
            if let Some(next) = tracks.get(track_idx + 1).map(|track| track.get_channel(0).unwrap()) {
                let post_context = next[..next.len().min(context_chunk_size)].to_vec();
                resampler.post(vec![post_context]).unwrap();
            }

            let expected_input = input.get_channel(0).unwrap();
            let expected_inputs = [expected_input];
            let expected = resampler.process_all(&expected_inputs).unwrap();
            assert_eq!(
                output.get_channel(0).unwrap().len(),
                expected.get_channel(0).unwrap().len()
            );
            for (actual, expected) in output
                .get_channel(0)
                .unwrap()
                .iter()
                .zip(expected.get_channel(0).unwrap().iter())
            {
                assert!((*actual - *expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn batch_matches_independent_process_all() {
        let config = mono_config(44_100, 48_000);
        let batch = PlanarResampler::<f32>::new(config.clone()).unwrap();
        let chunk = batch.input_chunk_size() / batch.config().channels;
        let tracks: Vec<PlanarVecs<f32>> = vec![
            PlanarVecs::new(vec![
                (0..(chunk + 17))
                    .map(|frame| (frame as f32 * 0.009).sin() * 0.2)
                    .collect(),
            ])
            .unwrap(),
            PlanarVecs::new(vec![
                (0..(chunk * 2 + 5))
                    .map(|frame| (frame as f32 * 0.012).cos() * 0.15)
                    .collect(),
            ])
            .unwrap(),
            PlanarVecs::new(vec![
                (0..(chunk / 2 + 11))
                    .map(|frame| (frame as f32 * 0.021).sin() * 0.25)
                    .collect(),
            ])
            .unwrap(),
        ];

        let expected: Vec<PlanarVecs<f32>> = tracks
            .iter()
            .map(|track| {
                let mut resampler = PlanarResampler::new(config.clone()).unwrap();
                let input_refs = [track.get_channel(0).unwrap()];
                resampler.process_all(&input_refs).unwrap()
            })
            .collect();
        let actual = batch.batch(tracks).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_track, expected_track) in actual.iter().zip(expected.iter()) {
            assert_eq!(
                actual_track.get_channel(0).unwrap().len(),
                expected_track.get_channel(0).unwrap().len()
            );
            for (left, right) in actual_track
                .get_channel(0)
                .unwrap()
                .iter()
                .zip(expected_track.get_channel(0).unwrap().iter())
            {
                assert!((*left - *right).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn stereo_channels_are_processed_independently() {
        let mut resampler = PlanarResampler::new(stereo_config(44_100, 48_000)).unwrap();
        let mut input = vec![0.0; resampler.input_chunk_size()];
        input[0] = 1.0;

        let output = process_all_samples(&mut resampler, &input).unwrap();

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
        let mut offline = PlanarResampler::new(config.clone()).unwrap();
        let mut streaming = PlanarResampler::new(config).unwrap();
        let input_frames = input_chunk_frames(&streaming) * 2 + input_chunk_frames(&streaming) / 3;
        let mut input = Vec::with_capacity(input_frames * 2);
        for frame in 0..input_frames {
            input.push((frame as f32 * 0.01).sin() * 0.25);
            input.push((frame as f32 * 0.017).cos() * 0.125);
        }

        let offline_output = process_all_samples(&mut offline, &input).unwrap();

        let input_buffer_size = streaming.input_chunk_size();
        let full_chunks = input.len() / input_buffer_size;
        let has_partial = !input.len().is_multiple_of(input_buffer_size);
        let output_blocks = full_chunks + usize::from(has_partial) + 1;
        let mut streaming_output = vec![0.0; output_blocks * streaming.output_chunk_size()];
        let mut written = 0;
        let mut offset = 0;
        while offset + input_buffer_size <= input.len() {
            written += process_chunk_samples(
                &mut streaming,
                &input[offset..offset + input_buffer_size],
                &mut streaming_output[written..],
            )
            .unwrap();
            offset += input_buffer_size;
        }
        written +=
            process_chunk_final_samples(&mut streaming, &input[offset..], &mut streaming_output[written..]).unwrap();
        written += finalize_samples_chunk(&mut streaming, &mut streaming_output[written..]).unwrap();
        streaming_output.truncate(written);
        streaming_output.truncate(offline_output.len());

        assert_eq!(offline_output.len(), streaming_output.len());
        for (left, right) in offline_output.iter().zip(streaming_output) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn rejects_wrong_streaming_chunk_size() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler) - 1];

        assert!(process_chunk_samples(&mut resampler, &input, &mut []).is_err());
    }

    #[test]
    fn too_small_output_does_not_advance_stream_state() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut too_small = vec![0.0; output_chunk_frames(&resampler) - 1];

        assert!(matches!(
            process_chunk_samples(&mut resampler, &input, &mut too_small),
            Err(Error::InsufficientOutputBuffer { .. })
        ));
        assert_eq!(resampler.input_sample_processed(), 0);

        let mut output = vec![0.0; output_chunk_frames(&resampler)];
        let expected = output_chunk_frames(&resampler) - resampler.output_delay_frames();
        assert_eq!(
            process_chunk_samples(&mut resampler, &input, &mut output).unwrap(),
            expected
        );
        assert_eq!(resampler.input_sample_processed(), input.len());
    }

    #[test]
    fn input_sample_count_tracks_full_and_partial_stream_input() {
        let mut resampler = PlanarResampler::new(stereo_config(44_100, 48_000)).unwrap();
        let mut output = vec![0.0; resampler.output_chunk_size()];
        let full = vec![0.0; resampler.input_chunk_size()];
        let partial = vec![0.0; 10];

        assert_eq!(resampler.input_sample_processed(), 0);
        let _ = process_chunk_samples(&mut resampler, &full, &mut output).unwrap();
        assert_eq!(resampler.input_sample_processed(), full.len());
        let _ = process_chunk_final_samples(&mut resampler, &partial, &mut output).unwrap();
        assert_eq!(resampler.input_sample_processed(), full.len() + partial.len());
    }

    #[test]
    fn too_small_finish_output_does_not_mark_flushed() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut first_output = vec![0.0; output_chunk_frames(&resampler)];
        process_chunk_samples(&mut resampler, &input, &mut first_output).unwrap();

        let mut too_small = vec![0.0; resampler.output_delay_frames() - 1];

        assert!(matches!(
            finalize_samples_chunk(&mut resampler, &mut too_small),
            Err(Error::InsufficientOutputBuffer { .. })
        ));

        let mut output = vec![0.0; output_chunk_frames(&resampler)];
        assert_eq!(
            finalize_samples_chunk(&mut resampler, &mut output).unwrap(),
            resampler.output_delay_frames()
        );
    }

    #[test]
    fn first_chunk_is_delay_trimmed_and_flushes_tail() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let first_written = process_chunk_samples(&mut resampler, &input, &mut output).unwrap();
        let flush_written = finalize_samples_chunk(&mut resampler, &mut output).unwrap();

        assert_eq!(first_written + flush_written, output_chunk_frames(&resampler));
        assert_eq!(
            first_written,
            output_chunk_frames(&resampler) - resampler.output_delay_frames()
        );
        assert_eq!(flush_written, resampler.output_delay_frames());
    }

    #[test]
    fn finalize_without_explicit_final_chunk_caps_to_expected_total() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let input_samples = input.len() * 2;
        let expected_total = resampler.expected_output_size(input_samples);
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        let first_written = process_chunk_samples(&mut resampler, &input, &mut output).unwrap();
        let second_written = process_chunk_samples(&mut resampler, &input, &mut output).unwrap();
        let flush_written = finalize_samples_chunk(&mut resampler, &mut output).unwrap();

        assert_eq!(first_written + second_written + flush_written, expected_total);
    }

    #[test]
    fn empty_finish_does_not_emit_extra_block() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; input_chunk_frames(&resampler)];
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        process_chunk_samples(&mut resampler, &input, &mut output).unwrap();
        assert_eq!(
            process_chunk_final_samples(&mut resampler, &[], &mut output).unwrap(),
            0
        );
        assert_eq!(
            finalize_samples_chunk(&mut resampler, &mut output).unwrap(),
            resampler.output_delay_frames()
        );
    }

    #[test]
    fn empty_stream_flushes_no_samples() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();
        let mut output = vec![0.0; output_chunk_frames(&resampler)];

        assert_eq!(
            process_chunk_final_samples(&mut resampler, &[], &mut output).unwrap(),
            0
        );
        assert_eq!(finalize_samples_chunk(&mut resampler, &mut output).unwrap(), 0);
    }

    #[test]
    fn second_finish_after_reset_returns_zero_without_input() {
        let mut resampler = PlanarResampler::new(mono_config(44_100, 48_000)).unwrap();

        let mut output = vec![0.0; output_chunk_frames(&resampler)];
        assert_eq!(finalize_samples_chunk(&mut resampler, &mut output).unwrap(), 0);
        assert!(matches!(
            finalize_samples_chunk(&mut resampler, &mut output),
            Err(Error::AlreadyFinalized)
        ));
    }
}
