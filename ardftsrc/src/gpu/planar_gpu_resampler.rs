use num_traits::Float;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{Config, PlanarVecs};

use super::buffer::GpuScalar;
use super::context::GpuContext;
use super::error::GpuError;
use super::fft_program::FromF64;
use super::gpu_core::GpuCore;

/// GPU-accelerated, chunked sample-rate converter for planar audio buffers.
///
/// This is the GPU analogue of [`crate::PlanarResampler`], built on top of [`GpuCore`]. Because
/// GPU work is submitted and completed asynchronously, this type offers two APIs:
///
/// - A non-blocking family ([`push_chunk`](Self::push_chunk), [`push_ready`](Self::push_ready),
///   [`pull_chunk`](Self::pull_chunk), [`flush`](Self::flush)) that mirrors [`GpuCore`]'s own
///   push/pull/poll surface directly and preserves cross-chunk GPU batching.
/// - A blocking family ([`process_chunk`](Self::process_chunk),
///   [`process_chunk_final`](Self::process_chunk_final), [`finalize`](Self::finalize)) that gives
///   [`crate::PlanarResampler`]-shaped ergonomics (push in, get output back immediately) at the
///   cost of flushing the GPU pipeline on every call, forgoing the throughput benefit of
///   [`GpuContext`]'s `group_chunks` batching.
///
/// 1. Construct with [`PlanarGpuResampler::new`] or [`PlanarGpuResampler::with_config`].
/// 2. Query [`input_buffer_size()`](Self::input_buffer_size) and
///    [`output_buffer_size()`](Self::output_buffer_size).
/// 3. Stream chunks with either API family.
/// 4. Call [`finalize(...)`](Self::finalize) once per stream to emit delayed tail samples.
///
/// To end a stream early, call [`reset()`](Self::reset).
pub struct PlanarGpuResampler<T = f32> {
    core: GpuCore<T>,
    config: Config,
    output_delay_frames: usize,

    // Bounded scratch used only by the blocking drain helper, so a per-call `pull_output` request
    // never exceeds one chunk's worth -- without this, draining directly into shrinking
    // sub-slices of the caller's `output` could hit exactly zero remaining capacity right as more
    // output is still buffered inside `GpuCore`, and silently stop looping instead of erroring.
    output_scratch: Vec<Vec<T>>,

    finalized: bool,
    input_sample_processed: usize,
    output_sample_processed: usize,
}

#[allow(private_bounds)]
impl<T: Float + GpuScalar + FromF64> PlanarGpuResampler<T> {
    /// Builds a resampler from an already-constructed [`GpuContext`] and ring depth.
    ///
    /// `group_chunks` (GPU batching depth) is baked into `context` and cannot be changed later.
    /// Passing an existing context lets callers share one Vulkan device and compiled-shader cache
    /// across multiple resamplers.
    pub fn new(context: GpuContext<T>, ring_slots: usize) -> Result<Self, GpuError> {
        let config = context.config().clone();
        let output_delay_frames = context.derived().output_offset;
        let channels = config.channels;
        let core = GpuCore::new(context, ring_slots)?;
        let output_scratch = vec![vec![T::zero(); core.output_chunk_frames()]; channels];

        Ok(Self {
            core,
            config,
            output_delay_frames,
            output_scratch,
            finalized: false,
            input_sample_processed: 0,
            output_sample_processed: 0,
        })
    }

    /// Convenience constructor that auto-selects a Vulkan device and builds its own
    /// [`GpuContext`].
    pub fn with_config(config: Config, group_chunks: usize, ring_slots: usize) -> Result<Self, GpuError> {
        let context = GpuContext::new(config, group_chunks)?;
        Self::new(context, ring_slots)
    }

    /// Returns the configuration this instance was built with.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Number of FFT chunks batched into one GPU submission.
    pub fn group_chunks(&self) -> usize {
        self.core.group_chunks()
    }

    /// Number of pre-allocated GPU buffer-sets in the fixed ring.
    pub fn ring_slots(&self) -> usize {
        self.core.ring_slots()
    }

    /// Returns the total number of input samples processed across all channels.
    #[inline]
    pub fn input_sample_processed(&self) -> usize {
        self.input_sample_processed
    }

    /// Returns the total number of output samples processed across all channels.
    #[inline]
    pub fn output_sample_processed(&self) -> usize {
        self.output_sample_processed
    }

    /// Returns the required total `input` length for each non-final chunk call. Divide by channel
    /// count to get the required frame count per channel.
    #[must_use]
    #[inline]
    pub fn input_buffer_size(&self) -> usize {
        self.core.input_chunk_frames() * self.config.channels
    }

    /// Returns the recommended total per-call `output` capacity. Divide by channel count to get
    /// the recommended frame capacity per channel.
    #[must_use]
    #[inline]
    pub fn output_buffer_size(&self) -> usize {
        self.core.output_chunk_frames() * self.config.channels
    }

    /// Returns algorithmic latency to trim/flush.
    #[must_use]
    #[inline]
    pub fn output_delay_frames(&self) -> usize {
        self.output_delay_frames
    }

    /// Returns the expected output length for a given input length.
    ///
    /// `input_size` can be expressed in either frames or samples; the returned value uses the
    /// same unit.
    #[must_use]
    #[inline]
    pub fn expected_output_size(&self, input_size: usize) -> usize {
        self.core.output_sample_count_for_input(input_size)
    }

    /// Returns true once [`finalize()`](Self::finalize) has completed successfully.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Non-blocking check for whether pushing another chunk is unlikely to block. See
    /// [`GpuCore::push_input_ready`].
    pub fn push_ready(&self) -> Result<bool, GpuError> {
        self.core.push_input_ready()
    }

    /// Pushes one full planar chunk (one slice per channel) into the pipeline without waiting for
    /// output.
    ///
    /// Each channel slice must contain exactly `input_buffer_size() / channels` samples. May
    /// block only if the GPU ring has no free buffer-set (see [`GpuCore::push_input`]).
    pub fn push_chunk(&mut self, input: &[&[T]]) -> Result<(), GpuError> {
        self.push_chunk_inner(input, false)
    }

    /// Pushes the final (possibly undersized) planar chunk into the pipeline.
    pub fn push_chunk_final(&mut self, input: &[&[T]]) -> Result<(), GpuError> {
        self.push_chunk_inner(input, true)
    }

    fn push_chunk_inner(&mut self, input: &[&[T]], is_final: bool) -> Result<(), GpuError> {
        if self.finalized {
            return Err(GpuError::AlreadyFinalized);
        }
        self.ensure_input_buffer_shape(input, is_final)?;
        self.core.push_input(input, is_final)?;
        self.input_sample_processed += input.iter().map(|channel| channel.len()).sum::<usize>();
        Ok(())
    }

    /// Copies as many samples as fit into `output` (one mutable slice per channel) from what's
    /// ready so far. Never blocks. Unlike the blocking family, each channel slice in `output` may
    /// be any length -- pass short slices to poll for a small amount of ready output, the same way
    /// [`GpuCore::pull_output`] accepts channel slices of any length.
    pub fn pull_chunk<'a>(&mut self, output: &'a mut [&'a mut [T]]) -> Result<usize, GpuError> {
        self.ensure_output_channel_count(output.len())?;
        let written_frames = self.core.pull_output(output)?;
        let written_samples = written_frames * self.config.channels;
        self.output_sample_processed += written_samples;
        Ok(written_samples)
    }

    /// Forces any partially-filled GPU group out and waits for outstanding GPU work to complete.
    /// The stream stays open. See [`GpuCore::flush`].
    pub fn flush(&mut self) -> Result<(), GpuError> {
        self.core.flush()
    }

    /// Resets this resampler so the next chunk starts an independent new stream. See
    /// [`GpuCore::reset`].
    pub fn reset(&mut self) -> Result<(), GpuError> {
        self.core.reset()?;
        self.finalized = false;
        self.input_sample_processed = 0;
        self.output_sample_processed = 0;
        Ok(())
    }

    /// Resamples complete planar channel inputs and returns all output samples.
    ///
    /// This is a convenience wrapper around the streaming API that drives [`GpuCore`]'s own
    /// push/pull surface directly (checking [`GpuCore::push_input_ready`] between pushes, and
    /// falling back to [`GpuCore::pull_output_blocking`] plus a full drain once it isn't) rather
    /// than flushing after every chunk -- unlike [`process_chunk`](Self::process_chunk), this
    /// keeps the full `group_chunks` GPU batching benefit while still returning a complete result
    /// synchronously.
    ///
    /// If this instance was already finalized by a previous call, it is reset first (cheap: by
    /// that point no GPU work is outstanding) so each call starts an independent stream, matching
    /// [`crate::PlanarResampler::process_all`]'s behavior of never requiring an explicit
    /// [`reset()`](Self::reset) between calls.
    pub fn process_all(&mut self, input: &[&[T]]) -> Result<PlanarVecs<T>, GpuError> {
        if self.finalized {
            self.reset()?;
        }
        self.ensure_process_all_input_shape(input)?;

        let output = run_stream_to_completion(&mut self.core, input)?;
        self.finalized = true;
        self.input_sample_processed += input.iter().map(|channel| channel.len()).sum::<usize>();
        self.output_sample_processed += output.iter().map(Vec::len).sum::<usize>();

        Ok(PlanarVecs::new(output).expect("channel lengths are equal by construction"))
    }

    /// Processes a planar chunk and blocks until its output is available.
    ///
    /// Convenience wrapper equivalent to [`push_chunk`](Self::push_chunk) followed by
    /// [`flush`](Self::flush) and a full drain. This forces every chunk through its own GPU
    /// submission, forgoing the throughput benefit of `group_chunks` batching -- prefer the
    /// non-blocking family ([`push_chunk`](Self::push_chunk)/[`pull_chunk`](Self::pull_chunk)) for
    /// GPU-batched throughput.
    pub fn process_chunk(&mut self, input: &[&[T]], output: &mut [&mut [T]]) -> Result<usize, GpuError> {
        self.push_chunk(input)?;
        self.core.flush()?;
        self.drain_all_ready_into(output)
    }

    /// Processes the final (possibly undersized) planar chunk and blocks until its output is
    /// available. See [`process_chunk`](Self::process_chunk) for the batching tradeoff.
    pub fn process_chunk_final(&mut self, input: &[&[T]], output: &mut [&mut [T]]) -> Result<usize, GpuError> {
        self.push_chunk_final(input)?;
        self.core.flush()?;
        self.drain_all_ready_into(output)
    }

    /// Emits delayed tail samples, then marks the stream finalized.
    ///
    /// Blocks until all outstanding GPU work completes. This is the terminal step of a stream and
    /// should be called once per stream.
    pub fn finalize(&mut self, output: &mut [&mut [T]]) -> Result<usize, GpuError> {
        if self.finalized {
            return Err(GpuError::AlreadyFinalized);
        }
        self.core.finalize()?;
        let written = self.drain_all_ready_into(output)?;
        self.finalized = true;
        Ok(written)
    }

    /// Loops [`GpuCore::pull_output`] until it returns `0`, copying each round out of a bounded
    /// scratch buffer into successive frame ranges of `output`. Used by the blocking API family,
    /// where every push is followed by a `flush()` so all resulting output is guaranteed ready by
    /// the time this is called.
    fn drain_all_ready_into(&mut self, output: &mut [&mut [T]]) -> Result<usize, GpuError> {
        self.ensure_output_buffer_shape(output)?;
        let channels = self.config.channels;
        let output_capacity_frames = output.iter().map(|channel| channel.len()).min().unwrap_or(0);
        let mut total_written_frames = 0;
        loop {
            let written_frames = self.core.pull_output(&mut self.output_scratch)?;
            if written_frames == 0 {
                break;
            }
            if total_written_frames + written_frames > output_capacity_frames {
                return Err(GpuError::InsufficientOutputBuffer {
                    expected: (total_written_frames + written_frames) * channels,
                    actual: output_capacity_frames * channels,
                });
            }
            for (channel_idx, channel_output) in output.iter_mut().enumerate() {
                channel_output[total_written_frames..total_written_frames + written_frames]
                    .copy_from_slice(&self.output_scratch[channel_idx][..written_frames]);
            }
            total_written_frames += written_frames;
        }
        let total_written_samples = total_written_frames * channels;
        self.output_sample_processed += total_written_samples;
        Ok(total_written_samples)
    }

    /// Sets previous-track context. See [`crate::PlanarResampler::pre`].
    pub fn pre(&mut self, pre: Vec<Vec<T>>) -> Result<(), GpuError> {
        if pre.len() != self.config.channels {
            return Err(GpuError::WrongChannelCount {
                expected: self.config.channels,
                actual: pre.len(),
            });
        }

        let max_samples = self.core.input_chunk_frames();
        let pre: Vec<Vec<T>> = pre
            .into_iter()
            .map(|mut samples| {
                if samples.len() > max_samples {
                    samples = samples.split_off(samples.len() - max_samples);
                }
                samples
            })
            .collect();
        self.core.pre(pre);
        Ok(())
    }

    /// Sets next-track context. See [`crate::PlanarResampler::post`].
    pub fn post(&mut self, post: Vec<Vec<T>>) -> Result<(), GpuError> {
        if post.len() != self.config.channels {
            return Err(GpuError::WrongChannelCount {
                expected: self.config.channels,
                actual: post.len(),
            });
        }

        let max_samples = self.core.input_chunk_frames();
        let post: Vec<Vec<T>> = post
            .into_iter()
            .map(|mut samples| {
                samples.truncate(max_samples);
                samples
            })
            .collect();
        self.core.post(post);
        Ok(())
    }

    /// Process multiple independent tracks.
    ///
    /// Each input is treated as its own stream with no inter-track context. See
    /// [`batch_gapless()`](Self::batch_gapless) for gapless processing of multiple tracks.
    ///
    /// Each track gets its own [`GpuCore`], built from a [`GpuContext::clone_shared`] clone of
    /// this resampler's context so it shares the same Vulkan device and already-compiled shaders
    /// (no per-track shader recompilation) without touching this instance's own stream. With the
    /// `rayon` feature enabled, tracks are processed in parallel -- safe because
    /// [`GpuDevice`](super::GpuDevice) internally serializes actual Vulkan submission, so
    /// concurrent tracks still get real overlap on the host-side work (windowing, extrapolation,
    /// downloads) and on waiting for each other's GPU fences.
    pub fn batch(&self, inputs: Vec<PlanarVecs<T>>) -> Result<Vec<PlanarVecs<T>>, GpuError>
    where
        T: Send + Sync,
    {
        // Extracted up front rather than captured as `&self` inside the closures below: `GpuCore`
        // is deliberately not `Sync` (its GPU buffers use interior mutability guarded by Vulkan
        // fences, not Rust's borrow checker), but `GpuContext` and `usize` are, so per-track work
        // only needs to share those.
        let context = self.core.context();
        let ring_slots = self.core.ring_slots();
        let channels = self.config.channels;

        #[cfg(feature = "rayon")]
        {
            inputs
                .into_par_iter()
                .map(|input| Self::batch_process_track(context, ring_slots, channels, &input, None, None))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .into_iter()
                .map(|input| Self::batch_process_track(context, ring_slots, channels, &input, None, None))
                .collect()
        }
    }

    /// Process multiple tracks as one gapless sequence.
    ///
    /// Adjacent inputs are treated as tracks from the same album or other back-to-back material.
    /// Each track is returned separately, but the previous track's tail and next track's head are
    /// used as edge context to improve gapless playback. See [`batch()`](Self::batch) for the
    /// per-track `GpuCore`/parallelism notes.
    pub fn batch_gapless(&self, inputs: Vec<PlanarVecs<T>>) -> Result<Vec<PlanarVecs<T>>, GpuError>
    where
        T: Send + Sync,
    {
        let context = self.core.context();
        let ring_slots = self.core.ring_slots();
        let channels = self.config.channels;
        let context_chunk_size = self.core.input_chunk_frames();

        #[cfg(feature = "rayon")]
        {
            inputs
                .par_iter()
                .enumerate()
                .map(|(track_idx, input)| {
                    let pre = track_idx
                        .checked_sub(1)
                        .map(|idx| Self::batch_track_tail_context(&inputs[idx], context_chunk_size));
                    let post = inputs
                        .get(track_idx + 1)
                        .map(|next| Self::batch_track_head_context(next, context_chunk_size));
                    Self::batch_process_track(context, ring_slots, channels, input, pre, post)
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
                        .map(|idx| Self::batch_track_tail_context(&inputs[idx], context_chunk_size));
                    let post = inputs
                        .get(track_idx + 1)
                        .map(|next| Self::batch_track_head_context(next, context_chunk_size));
                    Self::batch_process_track(context, ring_slots, channels, input, pre, post)
                })
                .collect()
        }
    }

    /// Runs one independent track to completion on its own freshly built [`GpuCore`] (cloned from
    /// `context` via [`GpuContext::clone_shared`], so it shares the device and any already-
    /// compiled shaders). Takes `context`/`ring_slots`/`channels` explicitly rather than `&self`
    /// so it can be called from parallel `rayon` closures without requiring `GpuCore` -- which is
    /// deliberately not `Sync` -- to be shared across threads.
    fn batch_process_track(
        context: &GpuContext<T>,
        ring_slots: usize,
        channels: usize,
        input: &PlanarVecs<T>,
        pre: Option<Vec<Vec<T>>>,
        post: Option<Vec<Vec<T>>>,
    ) -> Result<PlanarVecs<T>, GpuError> {
        if input.channels() != channels {
            return Err(GpuError::WrongChannelCount {
                expected: channels,
                actual: input.channels(),
            });
        }

        let mut track_core = GpuCore::new(context.clone_shared(), ring_slots)?;
        if let Some(pre) = pre {
            track_core.pre(pre);
        }
        if let Some(post) = post {
            track_core.post(post);
        }

        let refs: Vec<&[T]> = (0..input.channels())
            .map(|c| input.get_channel(c).expect("channel index in range"))
            .collect();
        let output = run_stream_to_completion(&mut track_core, &refs[..])?;
        Ok(PlanarVecs::new(output).expect("channel lengths are equal by construction"))
    }

    fn batch_track_tail_context(input: &PlanarVecs<T>, context_chunk_size: usize) -> Vec<Vec<T>> {
        (0..input.channels())
            .map(|c| {
                let channel = input.get_channel(c).expect("channel index in range");
                let start = channel.len().saturating_sub(context_chunk_size);
                channel[start..].to_vec()
            })
            .collect()
    }

    fn batch_track_head_context(input: &PlanarVecs<T>, context_chunk_size: usize) -> Vec<Vec<T>> {
        (0..input.channels())
            .map(|c| {
                let channel = input.get_channel(c).expect("channel index in range");
                let end = channel.len().min(context_chunk_size);
                channel[..end].to_vec()
            })
            .collect()
    }

    #[inline]
    fn ensure_process_all_input_shape(&self, input: &[&[T]]) -> Result<(), GpuError> {
        if input.len() != self.config.channels {
            return Err(GpuError::WrongChannelCount {
                expected: self.config.channels,
                actual: input.len(),
            });
        }
        let frames = input.first().map_or(0, |channel| channel.len());
        if let Some(channel) = input.iter().find(|channel| channel.len() != frames) {
            return Err(GpuError::WrongFrameCount {
                expected: frames,
                actual: channel.len(),
            });
        }
        Ok(())
    }

    #[inline]
    fn ensure_input_buffer_shape(&self, input: &[&[T]], is_final: bool) -> Result<(), GpuError> {
        if input.len() != self.config.channels {
            return Err(GpuError::WrongChannelCount {
                expected: self.config.channels,
                actual: input.len(),
            });
        }

        let expected_frames = self.core.input_chunk_frames();
        if !is_final {
            if let Some(channel) = input.iter().find(|channel| channel.len() != expected_frames) {
                return Err(GpuError::WrongFrameCount {
                    expected: expected_frames,
                    actual: channel.len(),
                });
            }
        } else if let Some(channel) = input.iter().find(|channel| channel.len() > expected_frames) {
            return Err(GpuError::WrongFrameCount {
                expected: expected_frames,
                actual: channel.len(),
            });
        }

        Ok(())
    }

    #[inline]
    fn ensure_output_channel_count(&self, channel_count: usize) -> Result<(), GpuError> {
        if channel_count != self.config.channels {
            return Err(GpuError::WrongChannelCount {
                expected: self.config.channels,
                actual: channel_count,
            });
        }
        Ok(())
    }

    #[inline]
    fn ensure_output_buffer_shape(&self, output: &[&mut [T]]) -> Result<(), GpuError> {
        self.ensure_output_channel_count(output.len())?;

        let expected_frames = self.core.output_chunk_frames();
        if let Some(channel) = output.iter().find(|channel| channel.len() < expected_frames) {
            return Err(GpuError::InsufficientOutputBuffer {
                expected: expected_frames,
                actual: channel.len(),
            });
        }

        Ok(())
    }
}

/// Drives `core` through one complete stream given full per-channel `input`, using [`GpuCore`]'s
/// own async push/pull/poll surface directly rather than flushing after every chunk, so
/// `group_chunks` batching is preserved even though this returns a complete, synchronous result.
/// Used both for [`PlanarGpuResampler::process_all`] (against the instance's own core) and for
/// each independent track in [`PlanarGpuResampler::batch`]/[`PlanarGpuResampler::batch_gapless`]
/// (against a freshly built per-track core).
fn run_stream_to_completion<T: Float + GpuScalar + FromF64>(
    core: &mut GpuCore<T>,
    input: &[&[T]],
) -> Result<Vec<Vec<T>>, GpuError> {
    let channels = input.len();
    let chunk_frames = core.input_chunk_frames();
    let total_frames = input.first().map_or(0, |channel| channel.len());
    let expected_total = core.output_sample_count_for_input(total_frames);

    let mut output: Vec<Vec<T>> = (0..channels).map(|_| Vec::with_capacity(expected_total)).collect();
    let mut scratch: Vec<Vec<T>> = vec![vec![T::zero(); core.output_chunk_frames()]; channels];

    let append = |output: &mut [Vec<T>], scratch: &[Vec<T>], written: usize| {
        for (dst, src) in output.iter_mut().zip(scratch) {
            dst.extend_from_slice(&src[..written]);
        }
    };

    let mut offset = 0;
    let mut pushed_final = false;
    while !pushed_final {
        while core.push_input_ready()? {
            let remaining = total_frames - offset;
            let is_final = remaining <= chunk_frames;
            let this_chunk = if is_final { remaining } else { chunk_frames };
            let refs: Vec<&[T]> = input.iter().map(|channel| &channel[offset..offset + this_chunk]).collect();
            core.push_input(&refs[..], is_final)?;
            offset += this_chunk;
            if is_final {
                pushed_final = true;
                break;
            }
        }

        let written = core.pull_output_blocking(&mut scratch)?;
        append(&mut output, &scratch, written);
        loop {
            let written = core.pull_output(&mut scratch)?;
            if written == 0 {
                break;
            }
            append(&mut output, &scratch, written);
        }
    }

    core.finalize()?;
    loop {
        let written = core.pull_output(&mut scratch)?;
        if written == 0 {
            break;
        }
        append(&mut output, &scratch, written);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{GpuDevice, PlanarResampler};

    const RING_SLOTS: usize = 4;
    const GROUP_CHUNKS: usize = 4;

    fn mono_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config::new(input_sample_rate, output_sample_rate, 1)
    }

    fn stereo_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config::new(input_sample_rate, output_sample_rate, 2)
    }

    fn tone_planar(channels: usize, frames: usize, sample_rate: usize) -> Vec<Vec<f32>> {
        (0..channels)
            .map(|c| {
                let hz = 400.0 + 137.0 * c as f64;
                (0..frames)
                    .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / sample_rate as f64).sin() as f32)
                    .collect()
            })
            .collect()
    }

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Builds a GPU wrapper for `config`, or returns `None` (with a diagnostic on stderr) when no
    /// usable Vulkan device is present, so tests skip gracefully in environments without a GPU.
    fn try_build_gpu(config: Config, group_chunks: usize) -> Option<PlanarGpuResampler<f32>> {
        let device = match GpuDevice::auto_select() {
            Ok(device) => Arc::new(device),
            Err(err) => {
                eprintln!("skipping GPU test: {err}");
                return None;
            }
        };
        let context = GpuContext::with_device(device, config, group_chunks).expect("build GpuContext");
        Some(PlanarGpuResampler::new(context, RING_SLOTS).expect("build PlanarGpuResampler"))
    }

    /// Builds real pre/input/post planar context around `total_frames` of useful input, and
    /// returns the derived per-channel chunk frame count. Real pre/post context sidesteps the
    /// known, separately tracked `Extrapolation::Lpc`/GPU divergence without it (see
    /// `gpu_core.rs`'s `lpc_extrapolation_has_known_gpu_divergence_without_context`).
    #[allow(clippy::type_complexity)]
    fn make_pre_input_post(
        config: &Config,
        total_frames: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, usize) {
        let derived = config.derive_config::<f32>().unwrap();
        let chunk_frames = derived.raw_input_chunk_frames();
        let channels = config.channels;
        let full = tone_planar(channels, chunk_frames * 2 + total_frames, config.input_sample_rate);
        let pre: Vec<Vec<f32>> = full.iter().map(|c| c[..chunk_frames].to_vec()).collect();
        let input: Vec<Vec<f32>> = full
            .iter()
            .map(|c| c[chunk_frames..chunk_frames + total_frames].to_vec())
            .collect();
        let post: Vec<Vec<f32>> = full.iter().map(|c| c[chunk_frames + total_frames..].to_vec()).collect();
        (pre, input, post, chunk_frames)
    }

    fn channel_refs(planar: &[Vec<f32>]) -> Vec<&[f32]> {
        planar.iter().map(Vec::as_slice).collect()
    }

    fn cpu_reference(config: Config, input: &[Vec<f32>], pre: Vec<Vec<f32>>, post: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let channels = config.channels;
        let mut resampler = PlanarResampler::<f32>::new(config).unwrap();
        resampler.pre(pre).unwrap();
        resampler.post(post).unwrap();

        let chunk = resampler.input_buffer_size() / channels;
        let out_cap = resampler.output_buffer_size() / channels;
        let total_frames = input[0].len();
        let mut output = vec![Vec::new(); channels];
        let mut offset = 0;
        while offset + chunk <= total_frames {
            let input_slices: Vec<&[f32]> = input.iter().map(|c| &c[offset..offset + chunk]).collect();
            let mut scratch = vec![vec![0.0f32; out_cap]; channels];
            let mut scratch_refs: Vec<&mut [f32]> = scratch.iter_mut().map(Vec::as_mut_slice).collect();
            let written = resampler
                .process_chunk(&input_slices, &mut scratch_refs[..])
                .unwrap()
                / channels;
            for c in 0..channels {
                output[c].extend_from_slice(&scratch[c][..written]);
            }
            offset += chunk;
        }
        let input_slices: Vec<&[f32]> = input.iter().map(|c| &c[offset..]).collect();
        let mut scratch = vec![vec![0.0f32; out_cap]; channels];
        let mut scratch_refs: Vec<&mut [f32]> = scratch.iter_mut().map(Vec::as_mut_slice).collect();
        let written = resampler
            .process_chunk_final(&input_slices, &mut scratch_refs[..])
            .unwrap()
            / channels;
        for c in 0..channels {
            output[c].extend_from_slice(&scratch[c][..written]);
        }

        let mut scratch = vec![vec![0.0f32; out_cap]; channels];
        let mut scratch_refs: Vec<&mut [f32]> = scratch.iter_mut().map(Vec::as_mut_slice).collect();
        let written = resampler.finalize(&mut scratch_refs[..]).unwrap() / channels;
        for c in 0..channels {
            output[c].extend_from_slice(&scratch[c][..written]);
        }

        output
    }

    /// Streams `input` through `gpu` via the blocking API family and returns the full output.
    fn gpu_blocking_stream(gpu: &mut PlanarGpuResampler<f32>, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let channels = gpu.config().channels;
        let chunk = gpu.input_buffer_size() / channels;
        let out_cap = gpu.output_buffer_size() / channels;
        let total_frames = input[0].len();
        let mut output = vec![Vec::new(); channels];
        let mut offset = 0;
        while offset + chunk <= total_frames {
            let input_slices = channel_refs(input);
            let input_slices: Vec<&[f32]> = input_slices.iter().map(|c| &c[offset..offset + chunk]).collect();
            let mut scratch = vec![vec![0.0f32; out_cap]; channels];
            let mut scratch_refs: Vec<&mut [f32]> = scratch.iter_mut().map(Vec::as_mut_slice).collect();
            let written = gpu.process_chunk(&input_slices, &mut scratch_refs[..]).unwrap() / channels;
            for c in 0..channels {
                output[c].extend_from_slice(&scratch[c][..written]);
            }
            offset += chunk;
        }
        let input_slices = channel_refs(input);
        let input_slices: Vec<&[f32]> = input_slices.iter().map(|c| &c[offset..]).collect();
        let mut scratch = vec![vec![0.0f32; out_cap]; channels];
        let mut scratch_refs: Vec<&mut [f32]> = scratch.iter_mut().map(Vec::as_mut_slice).collect();
        let written = gpu
            .process_chunk_final(&input_slices, &mut scratch_refs[..])
            .unwrap()
            / channels;
        for c in 0..channels {
            output[c].extend_from_slice(&scratch[c][..written]);
        }

        let mut scratch = vec![vec![0.0f32; out_cap]; channels];
        let mut scratch_refs: Vec<&mut [f32]> = scratch.iter_mut().map(Vec::as_mut_slice).collect();
        let written = gpu.finalize(&mut scratch_refs[..]).unwrap() / channels;
        for c in 0..channels {
            output[c].extend_from_slice(&scratch[c][..written]);
        }

        output
    }

    fn flatten(planar: &[Vec<f32>]) -> Vec<f32> {
        let channels = planar.len();
        let frames = planar.first().map_or(0, Vec::len);
        let mut out = Vec::with_capacity(frames * channels);
        for frame in 0..frames {
            for channel in planar {
                out.push(channel[frame]);
            }
        }
        out
    }

    fn chunk_frames_probe(config: &Config) -> usize {
        config.derive_config::<f32>().unwrap().raw_input_chunk_frames()
    }

    #[test]
    fn blocking_api_matches_cpu_mono() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let (pre, input, post, chunk_frames) = make_pre_input_post(&config, chunk_frames_probe(&config) * 3 + 137);
        assert_eq!(gpu.input_buffer_size(), chunk_frames);
        gpu.pre(pre.clone()).unwrap();
        gpu.post(post.clone()).unwrap();

        let cpu_output = cpu_reference(config, &input, pre, post);
        let gpu_output = gpu_blocking_stream(&mut gpu, &input);

        assert_eq!(flatten(&cpu_output).len(), flatten(&gpu_output).len());
        assert!(max_abs_error(&flatten(&cpu_output), &flatten(&gpu_output)) < 1e-3);
    }

    #[test]
    fn blocking_api_matches_cpu_stereo_short_final() {
        let config = stereo_config(48_000, 44_100);
        let Some(mut gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let (pre, input, post, _) = make_pre_input_post(&config, 777);
        gpu.pre(pre.clone()).unwrap();
        gpu.post(post.clone()).unwrap();

        let cpu_output = cpu_reference(config, &input, pre, post);
        let gpu_output = gpu_blocking_stream(&mut gpu, &input);

        assert_eq!(flatten(&cpu_output).len(), flatten(&gpu_output).len());
        assert!(max_abs_error(&flatten(&cpu_output), &flatten(&gpu_output)) < 1e-3);
    }

    #[test]
    fn nonblocking_api_matches_cpu() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let (pre, input, post, chunk_frames) = make_pre_input_post(&config, chunk_frames_probe(&config) * 3 + 137);
        gpu.pre(pre.clone()).unwrap();
        gpu.post(post.clone()).unwrap();

        let cpu_output = cpu_reference(config, &input, pre, post);

        let chunk = chunk_frames;
        let total_frames = input[0].len();
        let mut tiny = [0.0f32; 37];
        let mut gpu_output: Vec<f32> = Vec::new();
        let mut offset = 0;
        loop {
            let remaining = total_frames - offset;
            let is_final = remaining <= chunk;
            let this_chunk = if is_final { remaining } else { chunk };
            let input_slices: Vec<&[f32]> = input.iter().map(|c| &c[offset..offset + this_chunk]).collect();
            if is_final {
                gpu.push_chunk_final(&input_slices).unwrap();
            } else {
                gpu.push_chunk(&input_slices).unwrap();
            }
            loop {
                let mut refs: [&mut [f32]; 1] = [&mut tiny];
                let written = gpu.pull_chunk(&mut refs[..]).unwrap();
                gpu_output.extend_from_slice(&tiny[..written]);
                if written < tiny.len() {
                    break;
                }
            }
            offset += this_chunk;
            if is_final {
                break;
            }
        }
        gpu.core.finalize().unwrap();
        loop {
            let mut refs: [&mut [f32]; 1] = [&mut tiny];
            let written = gpu.pull_chunk(&mut refs[..]).unwrap();
            gpu_output.extend_from_slice(&tiny[..written]);
            if written == 0 {
                break;
            }
        }

        assert_eq!(flatten(&cpu_output).len(), gpu_output.len());
        assert!(max_abs_error(&flatten(&cpu_output), &gpu_output) < 1e-3);
    }

    #[test]
    fn flush_makes_partial_group_output_available_without_ending_stream() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let chunk = gpu.input_buffer_size();
        let input = vec![0.5f32; chunk];
        let input_slices: [&[f32]; 1] = [&input];

        gpu.push_chunk(&input_slices).unwrap();
        let mut scratch = vec![0.0f32; gpu.output_buffer_size()];
        let mut scratch_refs: [&mut [f32]; 1] = [&mut scratch];
        assert_eq!(
            gpu.pull_chunk(&mut scratch_refs[..]).unwrap(),
            0,
            "output shouldn't be ready before an explicit flush"
        );

        gpu.flush().unwrap();
        let mut scratch_refs: [&mut [f32]; 1] = [&mut scratch];
        assert!(
            gpu.pull_chunk(&mut scratch_refs[..]).unwrap() > 0,
            "flush should force the partial group out"
        );
    }

    #[test]
    fn wrong_channel_count_is_rejected() {
        let config = stereo_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let mono_context = vec![0.0f32; 8];
        assert!(matches!(
            gpu.pre(vec![mono_context.clone()]),
            Err(GpuError::WrongChannelCount { expected: 2, actual: 1 })
        ));
        assert!(matches!(
            gpu.post(vec![mono_context]),
            Err(GpuError::WrongChannelCount { expected: 2, actual: 1 })
        ));

        let one_channel = vec![0.0f32; gpu.input_buffer_size() / 2];
        let one_channel_refs: [&[f32]; 1] = [&one_channel];
        assert!(matches!(
            gpu.push_chunk(&one_channel_refs),
            Err(GpuError::WrongChannelCount { expected: 2, actual: 1 })
        ));
    }

    #[test]
    fn wrong_frame_count_is_rejected() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let too_short = vec![0.0f32; gpu.input_buffer_size() - 1];
        let too_short_refs: [&[f32]; 1] = [&too_short];
        assert!(matches!(
            gpu.push_chunk(&too_short_refs),
            Err(GpuError::WrongFrameCount { .. })
        ));

        let too_long = vec![0.0f32; gpu.input_buffer_size() + 1];
        let too_long_refs: [&[f32]; 1] = [&too_long];
        assert!(matches!(
            gpu.push_chunk_final(&too_long_refs),
            Err(GpuError::WrongFrameCount { .. })
        ));
    }

    #[test]
    fn insufficient_output_buffer_is_rejected() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let input = vec![0.0f32; gpu.input_buffer_size()];
        let input_refs: [&[f32]; 1] = [&input];
        let mut too_small = vec![0.0f32; gpu.output_buffer_size() - 1];
        let mut too_small_refs: [&mut [f32]; 1] = [&mut too_small];
        assert!(matches!(
            gpu.process_chunk(&input_refs, &mut too_small_refs[..]),
            Err(GpuError::InsufficientOutputBuffer { .. })
        ));
    }

    #[test]
    fn lifecycle_tracks_finalized_and_resets() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        assert!(!gpu.is_finalized());

        let input = vec![0.0f32; gpu.input_buffer_size()];
        let input_refs: [&[f32]; 1] = [&input];
        let mut scratch = vec![0.0f32; gpu.output_buffer_size()];
        let mut scratch_refs: [&mut [f32]; 1] = [&mut scratch];
        gpu.process_chunk(&input_refs, &mut scratch_refs[..]).unwrap();
        let mut scratch_refs: [&mut [f32]; 1] = [&mut scratch];
        gpu.finalize(&mut scratch_refs[..]).unwrap();
        assert!(gpu.is_finalized());
        let mut scratch_refs: [&mut [f32]; 1] = [&mut scratch];
        assert!(matches!(
            gpu.finalize(&mut scratch_refs[..]),
            Err(GpuError::AlreadyFinalized)
        ));

        gpu.reset().unwrap();
        assert!(!gpu.is_finalized());
        assert_eq!(gpu.input_sample_processed(), 0);
        assert_eq!(gpu.output_sample_processed(), 0);

        let input_refs: [&[f32]; 1] = [&input];
        let mut scratch_refs: [&mut [f32]; 1] = [&mut scratch];
        let written = gpu.process_chunk(&input_refs, &mut scratch_refs[..]).unwrap();
        assert!(written > 0 || gpu.output_delay_frames() > 0);
    }

    #[test]
    fn accessors_match_derived_config() {
        let config = stereo_config(44_100, 48_000);
        let Some(gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let derived = config.derive_config::<f32>().unwrap();
        assert_eq!(gpu.input_buffer_size(), derived.raw_input_chunk_frames() * 2);
        assert_eq!(gpu.output_buffer_size(), derived.output_chunk_frames * 2);
        assert_eq!(gpu.output_delay_frames(), derived.output_offset);
        assert_eq!(
            gpu.expected_output_size(3),
            (3usize * config.output_sample_rate).div_ceil(config.input_sample_rate)
        );
    }

    #[test]
    fn process_all_matches_cpu() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let (pre, input, post, _) = make_pre_input_post(&config, chunk_frames_probe(&config) * 3 + 137);
        gpu.pre(pre.clone()).unwrap();
        gpu.post(post.clone()).unwrap();

        let cpu_output = flatten(&cpu_reference(config, &input, pre, post));

        let input_refs = channel_refs(&input);
        let gpu_output = gpu.process_all(&input_refs).unwrap().interleave();

        assert_eq!(cpu_output.len(), gpu_output.len());
        assert!(max_abs_error(&cpu_output, &gpu_output) < 1e-3);
    }

    #[test]
    fn process_all_can_be_called_repeatedly_without_explicit_reset() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let chunk = chunk_frames_probe(gpu.config());
        let input: Vec<f32> = (0..chunk * 2 + 11).map(|f| (f as f32 * 0.01).sin() * 0.2).collect();
        let input_refs: [&[f32]; 1] = [&input];

        let first = gpu.process_all(&input_refs).unwrap();
        assert!(gpu.is_finalized());

        // No explicit reset() call here -- process_all should reset on its own since the last
        // stream was already fully drained and finalized.
        let second = gpu.process_all(&input_refs).unwrap();
        assert!(gpu.is_finalized());

        assert_eq!(first.get_channel(0).unwrap(), second.get_channel(0).unwrap());
    }

    #[test]
    fn batch_matches_independent_process_all() {
        let config = mono_config(44_100, 48_000);
        let Some(gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let chunk = chunk_frames_probe(&config);
        let tracks: Vec<PlanarVecs<f32>> = vec![
            PlanarVecs::new(vec![
                (0..(chunk + 17)).map(|f| (f as f32 * 0.009).sin() * 0.2).collect(),
            ])
            .unwrap(),
            PlanarVecs::new(vec![
                (0..(chunk * 2 + 5)).map(|f| (f as f32 * 0.012).cos() * 0.15).collect(),
            ])
            .unwrap(),
            PlanarVecs::new(vec![
                (0..(chunk / 2 + 11)).map(|f| (f as f32 * 0.021).sin() * 0.25).collect(),
            ])
            .unwrap(),
        ];

        let expected: Vec<Vec<f32>> = tracks
            .iter()
            .map(|track| {
                let mut solo = try_build_gpu(config.clone(), GROUP_CHUNKS).unwrap();
                let input_refs = [track.get_channel(0).unwrap()];
                solo.process_all(&input_refs).unwrap().get_channel(0).unwrap().to_vec()
            })
            .collect();

        let actual = gpu.batch(tracks).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_track, expected_track) in actual.iter().zip(expected.iter()) {
            let actual_channel = actual_track.get_channel(0).unwrap();
            assert_eq!(actual_channel.len(), expected_track.len());
            assert!(max_abs_error(actual_channel, expected_track) < 1e-3);
        }
    }

    #[test]
    fn batch_gapless_matches_manual_pre_post() {
        let config = mono_config(44_100, 48_000);
        let Some(gpu) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let context_chunk_size = chunk_frames_probe(&config);
        let track_frames = context_chunk_size * 2 + 17;

        let tracks: Vec<PlanarVecs<f32>> = (0..3)
            .map(|track_idx| {
                let channel: Vec<f32> = (0..track_frames)
                    .map(|frame| {
                        let continuous_frame = track_idx * track_frames + frame;
                        (continuous_frame as f32 * 0.017).sin() * 0.25
                    })
                    .collect();
                PlanarVecs::new(vec![channel]).unwrap()
            })
            .collect();

        let outputs = gpu.batch_gapless(tracks.clone()).unwrap();

        for (track_idx, (input, output)) in tracks.iter().zip(outputs.iter()).enumerate() {
            let mut solo = try_build_gpu(config.clone(), GROUP_CHUNKS).unwrap();
            if let Some(previous) = track_idx.checked_sub(1).map(|idx| tracks[idx].get_channel(0).unwrap()) {
                let pre_context = previous[previous.len().saturating_sub(context_chunk_size)..].to_vec();
                solo.pre(vec![pre_context]).unwrap();
            }
            if let Some(next) = tracks.get(track_idx + 1).map(|track| track.get_channel(0).unwrap()) {
                let post_context = next[..next.len().min(context_chunk_size)].to_vec();
                solo.post(vec![post_context]).unwrap();
            }

            let expected_input = input.get_channel(0).unwrap();
            let expected_refs = [expected_input];
            let expected = solo.process_all(&expected_refs).unwrap();

            let output_channel = output.get_channel(0).unwrap();
            let expected_channel = expected.get_channel(0).unwrap();
            assert_eq!(output_channel.len(), expected_channel.len());
            assert!(max_abs_error(output_channel, expected_channel) < 1e-3);
        }
    }
}
