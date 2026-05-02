use std::sync::Arc;

use num_traits::Float;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, FftNum, RealFftPlanner, RealToComplex};

use crate::config::DerivedConfig;
use crate::lpc::{ExtrapolateFallback, extrapolate_backward, extrapolate_forward};
use crate::{Config, Error};

pub struct Ardftsrc<T = f32>
where
    T: Float + FftNum,
{
    /// User-supplied runtime config (rates/channels/quality), kept immutable after construction.
    config: Config,
    /// Precomputed FFT/chunk/offset dimensions and taper.
    derived: DerivedConfig<T>,
    /// Planned forward real FFT instance reused across all chunks.
    forward: Arc<dyn RealToComplex<T>>,
    /// Planned inverse real FFT that maps resized spectra back into time-domain output windows.
    inverse: Arc<dyn ComplexToReal<T>>,
    /// Shared scratch workspace reused per channel.
    scratch: Scratch<T>,
    /// Per-channel overlap buffers that carry second-half iFFT energy into the next output block.
    overlap: Vec<Vec<T>>,
    /// Interleaved block staging buffer used before delay-trim copy into caller output.
    output_block: Vec<T>,
    /// Per-channel previous input windows used for channel-local stop extrapolation.
    channel_prev_input_windows: Vec<Vec<T>>,
    /// Set when the final chunk is accepted so later chunk calls are ignored by stream contract.
    final_input_seen: bool,
    /// One-shot guard that enforces flush semantics and prevents duplicate tail emission.
    flushed: bool,
    /// Remaining output frames to skip so algorithmic startup delay is trimmed exactly once.
    trim_remaining: usize,
    /// Remaining tail frames to emit on flush after accounting for short final-chunk padding.
    flush_remaining: usize,
    /// Optional interleaved previous-track tail used as real start-edge context.
    pre: Option<Vec<T>>,
    /// Optional interleaved next-track head used as real stop-edge context.
    post: Option<Vec<T>>,
    /// Total number of interleaved input samples for the current stream.
    input_sample_count: usize,
    /// Total number of interleaved output samples for the current stream.
    output_sample_count: usize,
}

/// Reusable FFT working buffers for one transform pass.
///
/// This groups temporary vectors that are mutated on each channel transform so the hot path can
/// avoid repeated heap allocation and keep a stable memory layout for predictable performance.
struct Scratch<T>
where
    T: Float + FftNum,
{
    /// Time-domain window fed to forward FFT; receives deinterleaved input at configured offset.
    rdft_in: Vec<T>,
    /// Forward-transform spectrum before tapering and rate-domain bin remapping.
    spectrum: Vec<Complex<T>>,
    /// Sized output spectrum for inverse FFT, zero-filled beyond copied bins to avoid leakage.
    resampled_spectrum: Vec<Complex<T>>,
    /// Time-domain iFFT output used both for immediate writeout and overlap accumulation.
    rdft_out: Vec<T>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransformMode {
    /// Normal streaming block: write output and carry second half into overlap.
    Normal,
    /// Start-edge priming block: keep output muted, stage second half into overlap.
    Start,
    /// Finalize-tail block: accumulate first half into overlap buffer.
    End,
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
        let mut planner = RealFftPlanner::<T>::new();
        let forward = planner.plan_fft_forward(derived.input_fft_size);
        let inverse = planner.plan_fft_inverse(derived.output_fft_size);
        let output_offset = derived.output_offset;
        let scratch = Scratch {
            rdft_in: forward.make_input_vec(),
            spectrum: forward.make_output_vec(),
            resampled_spectrum: inverse.make_input_vec(),
            rdft_out: inverse.make_output_vec(),
        };
        let overlap = vec![vec![T::zero(); derived.output_chunk_frames]; config.channels];
        let output_block = vec![T::zero(); derived.output_chunk_frames * config.channels];
        let channel_prev_input_windows = vec![vec![T::zero(); derived.input_chunk_frames * 2]; config.channels];

        Ok(Self {
            config,
            derived,
            forward,
            inverse,
            scratch,
            overlap,
            output_block,
            channel_prev_input_windows,
            final_input_seen: false,
            flushed: false,
            trim_remaining: output_offset,
            flush_remaining: output_offset,
            pre: None,
            post: None,
            input_sample_count: 0,
            output_sample_count: 0,
        })
    }

    /// Returns the configuration this instance was built with.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the total number of interleaved input samples processed.
    pub fn input_sample_count(&self) -> usize {
        self.input_sample_count
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
        self.pre = self.normalize_context(pre)?;
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
        self.post = self.normalize_context(post)?;
        Ok(())
    }

    /// Output frames for a complete input length.
    ///
    /// Returns the ceil-rounded number of output frames expected for `input_frames`.
    pub fn output_frame_count(&self, input_frames: usize) -> usize {
        output_frame_count(
            input_frames,
            self.config.input_sample_rate,
            self.config.output_sample_rate,
        )
    }

    /// Output samples needed for a complete interleaved input length.
    ///
    /// This validates that `input_samples` is divisible by channel count, then returns the number of output samples need.
    /// This can be used to size the output buffer for the entire input stream.
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
    /// This is a convenience wrapper around the streaming API. It resets internal state before
    /// processing and again before returning, so repeated calls on the same instance are
    /// independent and do not share stream history.
    pub fn process_all(&mut self, input: &[T]) -> Result<Vec<T>, Error> {
        self.reset();
        let expected_samples = self.output_sample_count(input.len())?;

        let mut output = Vec::with_capacity(expected_samples);
        let mut offset = 0;
        let input_buffer_size = self.input_buffer_size();
        let output_buffer_size = self.output_buffer_size();
        let mut chunk_output_buffer = vec![T::zero(); output_buffer_size];

        while offset + input_buffer_size <= input.len() {
            let written = self.process_chunk(&input[offset..offset + input_buffer_size], &mut chunk_output_buffer)?;
            output.extend_from_slice(&chunk_output_buffer[..written]);
            offset += input_buffer_size;
        }

        let written = self.process_chunk_final(&input[offset..], &mut chunk_output_buffer)?;
        output.extend_from_slice(&chunk_output_buffer[..written]);

        let written = self.finalize(&mut chunk_output_buffer)?;
        output.extend_from_slice(&chunk_output_buffer[..written]);

        Ok(output)
    }

    /// Resets internal streaming state so the next input is treated as a new, independent stream.
    ///
    /// Call this between unrelated audio inputs (for example, between files) when reusing the
    /// same resampler instance, so edge/history state from one input cannot bleed into the next.
    pub fn reset(&mut self) {
        let zero = Complex::new(T::zero(), T::zero());
        self.scratch.rdft_in.fill(T::zero());
        self.scratch.spectrum.fill(zero);
        self.scratch.resampled_spectrum.fill(zero);
        self.scratch.rdft_out.fill(T::zero());
        for overlap in &mut self.overlap {
            overlap.fill(T::zero());
        }
        self.output_block.fill(T::zero());
        for prev_input_window in &mut self.channel_prev_input_windows {
            prev_input_window.fill(T::zero());
        }
        self.final_input_seen = false;
        self.flushed = false;
        self.trim_remaining = self.derived.output_offset;
        self.flush_remaining = self.derived.output_offset;
        self.input_sample_count = 0;
        self.output_sample_count = 0;
        self.pre = None;
        self.post = None;
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
        let expected = self.derived.input_chunk_frames * self.config.channels;
        if input.len() != expected {
            return Err(Error::WrongChunkLength {
                expected,
                actual: input.len(),
            });
        }

        self.process_chunk_inner(input, self.derived.input_chunk_frames, output, false)
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
        let frames = self.interleaved_frames(input)?;
        if frames > self.derived.input_chunk_frames {
            return Err(Error::WrongChunkLength {
                expected: self.derived.input_chunk_frames * self.config.channels,
                actual: input.len(),
            });
        }

        self.process_chunk_inner(input, frames, output, true)
    }

    /// Emits delayed tail samples, then resets stream state.
    ///
    /// This flushes any remaining overlap/delay samples that were held back by the chunked
    /// processing pipeline. It is the terminal step of a stream and should be called once per
    /// stream. If `process_chunk_final()` was not called, this treats the last accepted full chunk
    /// as terminal input.
    ///
    /// Returns the number of samples written to the output buffer.
    pub fn finalize(&mut self, output: &mut [T]) -> Result<usize, Error> {
        if self.flushed {
            return Err(Error::AlreadyFlushed);
        }
        self.final_input_seen = true;

        let written = if self.is_passthrough() || self.input_sample_count == 0 {
            self.flushed = true;
            0
        } else {
            let flush_candidate = self.flush_remaining * self.config.channels;
            let written_samples = self.cap_write_to_output_budget(flush_candidate);
            self.ensure_output_sample_capacity(output, written_samples)?;
            self.flushed = true;

            self.add_synthetic_finalize_tail_to_overlap()?;

            let scale = T::from(self.derived.output_chunk_frames).unwrap_or(T::one())
                / T::from(self.derived.input_chunk_frames).unwrap_or(T::one());
            let written_frames = written_samples / self.config.channels;

            if self.config.channels == 1 {
                let overlap = &self.overlap[0][..written_frames];
                for (dst, src) in output[..written_frames].iter_mut().zip(overlap.iter()) {
                    *dst = *src * scale;
                }
            } else {
                for frame in 0..written_frames {
                    for channel in 0..self.config.channels {
                        output[frame * self.config.channels + channel] = self.overlap[channel][frame] * scale;
                    }
                }
            }
            written_samples
        };

        self.output_sample_count += written;
        self.reset();
        Ok(written)
    }

    /// Validates interleaved sample length and converts it into frame count.
    ///
    /// Returns the number of frames represented by `input`, or
    /// `Error::MalformedInputLength` when sample count is not divisible by channels.
    fn interleaved_frames(&self, input: &[T]) -> Result<usize, Error> {
        if !input.len().is_multiple_of(self.config.channels) {
            return Err(Error::MalformedInputLength {
                channels: self.config.channels,
                samples: input.len(),
            });
        }
        Ok(input.len() / self.config.channels)
    }

    /// Returns expected total output samples once final stream extent is known.
    ///
    /// Before final input is seen, stream extent is unknown and this returns `None`.
    fn expected_total_output_samples(&self) -> Option<usize> {
        if !self.final_input_seen {
            return None;
        }

        let input_frames = self.input_sample_count / self.config.channels;
        Some(self.output_frame_count(input_frames) * self.config.channels)
    }

    /// Returns remaining output budget once final stream extent is known.
    fn remaining_output_budget_samples(&self) -> Option<usize> {
        self.expected_total_output_samples()
            .map(|expected_total| expected_total.saturating_sub(self.output_sample_count))
    }

    /// Caps a candidate write size to the remaining output budget when known.
    fn cap_write_to_output_budget(&self, candidate_samples: usize) -> usize {
        self.remaining_output_budget_samples()
            .map_or(candidate_samples, |remaining| candidate_samples.min(remaining))
    }

    /// Validates interleaved edge context and normalizes empty vectors to `None`.
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

    /// Processes one interleaved chunk through the streaming resampler.
    ///
    /// This internal entry point assumes `frames` has already been validated from `input`.
    /// It handles stream-level control flow (passthrough, first/final chunk state, per-channel
    /// dispatch, and trim/flush accounting), while `process_channel` performs channel-local FFT
    /// preparation and transform work.
    ///
    /// Behavior by mode:
    ///
    /// - If final input was already seen, returns `Error::StreamFinished`.
    /// - In passthrough mode (equal rates), it copies input directly to output.
    /// - In FFT mode, it processes every channel for one chunk, then applies startup trim and
    ///   writes the resulting contiguous samples from `output_block`.
    ///
    /// # Parameters
    ///
    /// - `input`: Interleaved input samples for this chunk.
    /// - `frames`: Number of frames represented by `input`.
    /// - `output`: Destination buffer for produced interleaved output samples.
    /// - `is_final`: Marks this chunk as the final chunk in the stream.
    ///
    /// Returns the number of samples written into `output`.
    fn process_chunk_inner(
        &mut self,
        input: &[T],
        frames: usize,
        output: &mut [T],
        is_final: bool,
    ) -> Result<usize, Error> {
        // Once final input has been consumed, reject additional chunk calls.
        if self.final_input_seen {
            return Err(Error::StreamFinished);
        }

        if self.is_passthrough() {
            // Fast path: exact-rate streams are copied directly without FFT processing.
            self.ensure_output_sample_capacity(output, input.len())?;
            if is_final {
                self.final_input_seen = true;
            }
            self.input_sample_count += input.len();
            let written_samples = self.cap_write_to_output_budget(input.len());
            output[..written_samples].copy_from_slice(&input[..written_samples]);
            self.output_sample_count += written_samples;
            return Ok(written_samples);
        }

        // FFT path always writes at most one full output chunk per call before trim.
        self.ensure_output_sample_capacity(output, self.output_buffer_size())?;

        if is_final {
            // Mark final before processing so downstream logic sees final-state consistently.
            self.final_input_seen = true;
            if frames == 0 {
                // Empty final chunk only flips stream state; it produces no samples.
                return Ok(0);
            }
        }

        // Process all channels for this chunk with first-input/final flags.
        let is_first_input = self.input_sample_count == 0;
        for channel in 0..self.config.channels {
            self.process_channel(input, frames, channel, is_first_input, is_final)?;
        }
        self.input_sample_count += frames * self.config.channels;

        // Apply startup trim, then copy the remaining contiguous interleaved output samples.
        let skip_frames = self.trim_remaining.min(self.derived.output_chunk_frames);
        self.trim_remaining -= skip_frames;
        let written_frames = self.derived.output_chunk_frames - skip_frames;
        let channel_count = self.config.channels;
        let candidate_samples = written_frames * channel_count;
        let written_samples = self.cap_write_to_output_budget(candidate_samples);
        let src_start = skip_frames * channel_count;
        output[..written_samples].copy_from_slice(&self.output_block[src_start..src_start + written_samples]);

        self.output_sample_count += written_samples;
        Ok(written_samples)
    }

    /// Rejects undersized output slices before mutating stream state.
    ///
    /// Returns `Ok(())` when `output` can hold at least `expected` samples, or
    /// `Error::InsufficientOutputBuffer` when it cannot.
    fn ensure_output_sample_capacity(&self, output: &[T], expected: usize) -> Result<(), Error> {
        if output.len() < expected {
            return Err(Error::InsufficientOutputBuffer {
                expected,
                actual: output.len(),
            });
        }
        Ok(())
    }

    /// Returns true when rates match and FFT processing can be bypassed losslessly.
    fn is_passthrough(&self) -> bool {
        self.config.input_sample_rate == self.config.output_sample_rate
    }

    /// Processes one channel for a single input chunk, including edge handling.
    ///
    /// This is the per-channel orchestrator that prepares `scratch.rdft_in` and dispatches to
    /// `transform_channel` with the correct edge policy:
    ///
    /// - Loads interleaved input samples into the channel-local analysis window.
    /// - On the first non-empty chunk, synthesizes start context so overlap-add has a stable
    ///   leading edge.
    /// - On a short final chunk, synthesize missing final-block samples from history + tail extrapolation.
    /// - Runs the main transform and overlap-add emission for this channel.
    /// - Persists channel history unless final-block missing-sample synthesis already consumed/advanced it.
    ///
    /// # Parameters
    ///
    /// - `input`: Interleaved input samples for this chunk.
    /// - `frames`: Number of valid input frames in `input`.
    /// - `channel`: Channel index to process from the interleaved stream.
    /// - `is_first_input`: `true` for the first non-empty chunk observed by the stream.
    /// - `is_final`: `true` when this chunk is the final call for the stream.
    ///
    /// Returns `Ok(())` after channel processing state and output staging are updated, or an
    /// error if downstream FFT processing fails.
    fn process_channel(
        &mut self,
        input: &[T],
        frames: usize,
        channel: usize,
        is_first_input: bool,
        is_final: bool,
    ) -> Result<(), Error> {
        self.copy_input_to_window(input, frames, channel);

        if is_first_input {
            // First chunk: pre-seed overlap with synthesized pre-roll so the start edge blends.
            self.synthesize_start_context(channel, frames)?;
            // Restore the "real" window after start-context synthesis consumed scratch buffers.
            self.copy_input_to_window(input, frames, channel);
        }

        let is_short_final = is_final && frames < self.derived.input_chunk_frames;
        if is_short_final {
            self.synthesize_final_block_missing_samples(input, frames, channel);
        }

        self.transform_channel(channel, TransformMode::Normal)?;

        if !is_short_final {
            // Keep history for future stop extrapolation unless short-final handling already consumed it.
            self.save_current_window(channel);
        }

        Ok(())
    }

    /// Loads one channel's interleaved input into the FFT window at the configured offset.
    fn copy_input_to_window(&mut self, input: &[T], frames: usize, channel: usize) {
        let channels = self.config.channels;
        self.scratch.rdft_in.fill(T::zero());
        let dst = &mut self.scratch.rdft_in[self.derived.input_offset..self.derived.input_offset + frames];
        Self::copy_interleaved_channel(dst, input, frames, channel, channels);
    }

    /// Copies one channel from interleaved input into `dst`.
    fn copy_interleaved_channel(dst: &mut [T], input: &[T], frames: usize, channel_idx: usize, channels_total: usize) {
        if channels_total == 1 {
            dst.copy_from_slice(&input[..frames]);
        } else {
            for (dst_sample, interleaved_frame) in dst.iter_mut().zip(input.chunks_exact(channels_total)) {
                *dst_sample = interleaved_frame[channel_idx];
            }
        }
    }

    /// Copies up to `dst.len()` trailing channel samples from `pre` into `dst`'s tail.
    fn copy_pre_tail(&self, channel: usize, dst: &mut [T]) -> usize {
        let Some(pre) = &self.pre else {
            return 0;
        };
        let pre_frames = pre.len() / self.config.channels;
        let copied = pre_frames.min(dst.len());
        let start_frame = pre_frames - copied;
        let dst_start = dst.len() - copied;
        if self.config.channels == 1 {
            dst[dst_start..].copy_from_slice(&pre[start_frame..start_frame + copied]);
        } else {
            let channels = self.config.channels;
            for (dst_sample, src_frame) in dst[dst_start..]
                .iter_mut()
                .zip(pre.chunks_exact(channels).skip(start_frame).take(copied))
            {
                *dst_sample = src_frame[channel];
            }
        }
        copied
    }

    /// Copies up to `dst.len()` leading channel samples from `post` into `dst`'s head.
    fn copy_post_head(&self, channel: usize, dst: &mut [T]) -> usize {
        let Some(post) = &self.post else {
            return 0;
        };
        let post_frames = post.len() / self.config.channels;
        let copied = post_frames.min(dst.len());
        if self.config.channels == 1 {
            dst[..copied].copy_from_slice(&post[..copied]);
        } else {
            for (dst_sample, src_frame) in dst[..copied]
                .iter_mut()
                .zip(post.chunks_exact(self.config.channels).take(copied))
            {
                *dst_sample = src_frame[channel];
            }
        }
        copied
    }

    /// Synthesizes start-edge context by backward extrapolation for the first non-empty chunk.
    ///
    /// Returns `Ok(())` after start context is prepared (or when no work is needed), or an error
    /// if the FFT pipeline fails while staging overlap state.
    ///
    /// This makes a huge difference to the Gapless Sine Test suite in HydrogenAudio's SRC tests.
    ///
    /// Returns `Ok(())` after start context is synthesized, or an error if the FFT pipeline fails.
    fn synthesize_start_context(&mut self, channel: usize, frames: usize) -> Result<(), Error> {
        if frames == 0 {
            return Ok(());
        }

        let input_start = self.derived.input_offset;
        let input_end = input_start + frames;
        let mut predicted = vec![T::zero(); input_start];
        let copied = self.copy_pre_tail(channel, &mut predicted);
        if copied < input_start {
            let fallback_len = input_start - copied;
            let fallback = extrapolate_backward(
                &self.scratch.rdft_in[input_start..input_end],
                fallback_len,
                ExtrapolateFallback::Hold,
            );
            predicted[..fallback_len].copy_from_slice(&fallback);
        }

        self.scratch.rdft_in.fill(T::zero());
        let tail_start = self.derived.input_chunk_frames;
        self.scratch.rdft_in[tail_start..tail_start + predicted.len()].copy_from_slice(&predicted);
        self.transform_channel(channel, TransformMode::Start)?;

        Ok(())
    }

    /// Fills a synthetic forward tail from `post` first, then LPC extrapolation fallback.
    fn build_tail_prediction(&self, channel: usize, base: &[T], needed: usize) -> Vec<T> {
        let mut predicted = vec![T::zero(); needed];
        let copied = self.copy_post_head(channel, &mut predicted);
        if copied < needed {
            let mut seed = Vec::with_capacity(base.len() + copied);
            seed.extend_from_slice(base);
            seed.extend_from_slice(&predicted[..copied]);
            let fallback = extrapolate_forward(&seed, needed - copied, ExtrapolateFallback::Hold);
            predicted[copied..].copy_from_slice(&fallback);
        }
        predicted
    }

    /// Builds stop-edge window from prior history for a final short chunk.
    fn assemble_short_final_work_window(
        &self,
        input: &[T],
        frames: usize,
        channel: usize,
        input_frames: usize,
        pad_frames: usize,
    ) -> Vec<T> {
        let mut work = vec![T::zero(); input_frames * 2];
        let state = &self.channel_prev_input_windows[channel];
        work[..pad_frames].copy_from_slice(&state[frames..frames + pad_frames]);
        Self::copy_interleaved_channel(
            &mut work[pad_frames..pad_frames + frames],
            input,
            frames,
            channel,
            self.config.channels,
        );
        work
    }

    /// Predicts and writes the synthetic short-final tail into `work`.
    fn fill_short_final_predicted_tail(&self, channel: usize, work: &mut [T], input_frames: usize) -> Vec<T> {
        let predicted = self.build_tail_prediction(channel, &work[..input_frames], input_frames);
        work[input_frames..input_frames * 2].copy_from_slice(&predicted);
        predicted
    }

    /// Commits short-final history mutations used by future finalize paths.
    fn commit_short_final_history(
        &mut self,
        channel: usize,
        frames: usize,
        pad_frames: usize,
        predicted: &[T],
        input_frames: usize,
    ) {
        if frames == 0 {
            return;
        }
        let state = &mut self.channel_prev_input_windows[channel];
        state[..frames].copy_from_slice(&predicted[pad_frames..pad_frames + frames]);
        state[frames..input_frames].fill(T::zero());
        state[input_frames..input_frames * 2].fill(T::zero());
    }

    /// Stages synthesized short-final window into `scratch.rdft_in`.
    fn stage_short_final_rdft_input_from_work(&mut self, work: &[T], pad_frames: usize, input_frames: usize) {
        self.scratch.rdft_in.fill(T::zero());
        let window_start = self.derived.input_offset;
        self.scratch.rdft_in[window_start..window_start + input_frames]
            .copy_from_slice(&work[pad_frames..pad_frames + input_frames]);
    }

    /// Builds stop-edge window from prior history for a final short chunk.
    fn synthesize_final_block_missing_samples(&mut self, input: &[T], frames: usize, channel: usize) {
        let input_frames = self.derived.input_chunk_frames;
        let pad_frames = input_frames - frames;

        let mut work = self.assemble_short_final_work_window(input, frames, channel, input_frames, pad_frames);
        let predicted = self.fill_short_final_predicted_tail(channel, &mut work, input_frames);
        self.commit_short_final_history(channel, frames, pad_frames, &predicted, input_frames);
        self.stage_short_final_rdft_input_from_work(&work, pad_frames, input_frames);
    }

    /// Runs one channel through the FFT-domain resampling pipeline for the current window.
    ///
    /// This method assumes `self.scratch.rdft_in` has already been prepared for a single
    /// channel (windowing, zero-padding, and stop-edge preparation if needed). It then:
    ///
    /// - Performs a forward real FFT.
    /// - Copies/tapers frequency bins into `resampled_spectrum` and clears unused bins.
    /// - Enforces real-valued DC/Nyquist bins required by `realfft`.
    /// - Performs an inverse real FFT back into `rdft_out`.
    /// - Applies mode-specific overlap/output handling for steady-state, start-edge priming,
    ///   or finalize-tail accumulation.
    ///
    /// The three boolean flags separate "produce output now" from "stage overlap state" so the
    /// caller can compose start/steady/stop edge behavior without duplicating transform logic.
    ///
    /// # Parameters
    ///
    /// - `channel`: Channel index to read overlap state from and write overlap state to.
    /// - `mode`: Selects whether to emit output and how overlap state is updated.
    ///
    /// Returns `Ok(())` on successful transform and overlap/output updates, or an FFT error from
    /// the backend.
    fn transform_channel(&mut self, channel: usize, mode: TransformMode) -> Result<(), Error> {
        // Transform the prepared time-domain window into frequency bins.
        self.forward
            .process(&mut self.scratch.rdft_in, &mut self.scratch.spectrum)
            .map_err(|err| Error::Fft(err.to_string()))?;

        // Apply spectral tapering while remapping to the output bin count.
        // Any bins that do not have a source counterpart are explicitly zeroed.
        let zero = Complex::new(T::zero(), T::zero());
        let bins = self
            .scratch
            .resampled_spectrum
            .len()
            .min(self.scratch.spectrum.len())
            .min(self.derived.taper.len());
        for (dst, (src, taper)) in self.scratch.resampled_spectrum[..bins].iter_mut().zip(
            self.scratch.spectrum[..bins]
                .iter()
                .zip(self.derived.taper[..bins].iter()),
        ) {
            *dst = *src * *taper;
        }
        if bins < self.scratch.resampled_spectrum.len() {
            self.scratch.resampled_spectrum[bins..].fill(zero);
        }
        // `realfft` requires the DC and Nyquist bins to be purely real.
        if let Some(dc_bin) = self.scratch.resampled_spectrum.get_mut(0) {
            dc_bin.im = T::zero();
        }
        if self.scratch.resampled_spectrum.len() > 1 {
            let nyquist_bin = self.scratch.resampled_spectrum.len() - 1;
            self.scratch.resampled_spectrum[nyquist_bin].im = T::zero();
        }

        // Return to the time domain after spectral shaping.
        self.inverse
            .process(&mut self.scratch.resampled_spectrum, &mut self.scratch.rdft_out)
            .map_err(|err| Error::Fft(err.to_string()))?;

        // `realfft` inverse is unnormalized, so divide by FFT size.
        // `scale` converts between chunk sizes (time-stretch / sample-rate ratio).
        let normalize = T::one() / T::from(self.derived.output_fft_size).unwrap_or(T::one());
        let scale = T::from(self.derived.output_chunk_frames).unwrap_or(T::one())
            / T::from(self.derived.input_chunk_frames).unwrap_or(T::one());
        let output_frames = self.derived.output_chunk_frames;

        if matches!(mode, TransformMode::Normal) {
            // Emit the first half and overlap-add with carry from the previous block.
            if self.config.channels == 1 {
                let overlap = &self.overlap[channel][..output_frames];
                let output = &mut self.output_block[..output_frames];
                for frame in 0..output_frames {
                    output[frame] = (self.scratch.rdft_out[frame] * normalize + overlap[frame]) * scale;
                }
            } else {
                for frame in 0..output_frames {
                    let sample = self.scratch.rdft_out[frame] * normalize + self.overlap[channel][frame];
                    self.output_block[frame * self.config.channels + channel] = sample * scale;
                }
            }
        }

        if matches!(mode, TransformMode::End) {
            // Some edge modes need to accumulate this block's first half into overlap.
            for (overlap, rdft) in self.overlap[channel][..output_frames]
                .iter_mut()
                .zip(self.scratch.rdft_out[..output_frames].iter())
            {
                *overlap = *overlap + *rdft * normalize;
            }
        }

        if matches!(mode, TransformMode::Normal | TransformMode::Start) {
            // Next block starts from the second half.
            for (overlap, rdft) in self.overlap[channel][..output_frames]
                .iter_mut()
                .zip(self.scratch.rdft_out[output_frames..output_frames * 2].iter())
            {
                *overlap = *rdft * normalize;
            }
        }

        Ok(())
    }

    /// Persists the current window so later stop extrapolation has channel-local history.
    fn save_current_window(&mut self, channel: usize) {
        let history_start = self.derived.input_offset;
        let history_end = history_start + self.derived.input_chunk_frames;
        let state = &mut self.channel_prev_input_windows[channel];
        state[..self.derived.input_chunk_frames].copy_from_slice(&self.scratch.rdft_in[history_start..history_end]);
        state[self.derived.input_chunk_frames..].fill(T::zero());
    }

    /// Adds synthetic stop tails into overlap when the final chunk was not short.
    ///
    /// If the final chunk was short, we've already done this.
    ///
    /// Returns `Ok(())` after all remaining channels have contributed their flush overlap, or an
    /// error if any channel's transform fails.
    fn add_synthetic_finalize_tail_to_overlap(&mut self) -> Result<(), Error> {
        // A non-empty short final chunk already synthesized stop edges in process_channel().
        if self.final_input_seen && !self.input_sample_count.is_multiple_of(self.input_buffer_size()) {
            return Ok(());
        }

        for channel in 0..self.config.channels {
            self.scratch.rdft_in.fill(T::zero());
            let input_frames = self.derived.input_chunk_frames;
            let input_offset = self.derived.input_offset;
            let base = self.channel_prev_input_windows[channel][..input_frames].to_vec();
            let predicted = self.build_tail_prediction(channel, &base, input_offset);
            let state = &mut self.channel_prev_input_windows[channel];
            state[input_frames..input_frames * 2].fill(T::zero());
            state[input_frames..input_frames + predicted.len()].copy_from_slice(&predicted);
            let input_start = self.derived.input_offset;
            self.scratch.rdft_in[input_start..input_start + input_frames]
                .copy_from_slice(&state[input_frames..input_frames + input_frames]);

            self.transform_channel(channel, TransformMode::End)?;
        }

        Ok(())
    }
}

/// Computes ceil-rounded output frames for rational-rate conversion sizing.
fn output_frame_count(input_frames: usize, input_rate: usize, output_rate: usize) -> usize {
    (input_frames * output_rate).div_ceil(input_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaperType;
    use dasp_signal::Signal;

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
        let mut max_abs_error_idx = 0usize;
        for (idx, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs_error = (left - right).abs();
            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_abs_error_idx = idx;
            }
        }
        (max_abs_error, max_abs_error_idx)
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

        assert!(matches!(
            resampler.process_chunk(&input, &mut []),
            Err(Error::WrongChunkLength { .. })
        ));
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

}