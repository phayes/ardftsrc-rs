use num_traits::Float;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{Config, PlanarVecs};

use super::buffer::GpuScalar;
use super::context::GpuContext;
use super::error::GpuError;
use super::fft_program::FromF64;
use super::gpu_core::GpuCore;
use super::planar_gpu_resampler::PlanarGpuResampler;

/// GPU-accelerated, chunked sample-rate converter for interleaved audio buffers.
///
/// This is the GPU analogue of [`crate::InterleavedResampler`], built on top of [`GpuCore`].
/// Because GPU work is submitted and completed asynchronously, this type offers two APIs:
///
/// - A non-blocking family ([`push_chunk`](Self::push_chunk), [`push_ready`](Self::push_ready),
///   [`pull_chunk`](Self::pull_chunk), [`flush`](Self::flush)) that mirrors [`GpuCore`]'s own
///   push/pull/poll surface directly and preserves cross-chunk GPU batching.
/// - A blocking family ([`process_chunk`](Self::process_chunk),
///   [`process_chunk_final`](Self::process_chunk_final), [`finalize`](Self::finalize)) that gives
///   [`crate::InterleavedResampler`]-shaped ergonomics (push in, get output back immediately) at
///   the cost of flushing the GPU pipeline on every call, forgoing the throughput benefit of
///   [`GpuContext`]'s `group_chunks` batching.
///
/// 1. Construct with [`InterleavedGpuResampler::new`] or [`InterleavedGpuResampler::with_config`].
/// 2. Query [`input_buffer_size()`](Self::input_buffer_size) and
///    [`output_buffer_size()`](Self::output_buffer_size).
/// 3. Stream chunks with either API family.
/// 4. Call [`finalize(...)`](Self::finalize) once per stream to emit delayed tail samples.
///
/// To end a stream early, call [`reset()`](Self::reset).
pub struct InterleavedGpuResampler<T = f32> {
    core: GpuCore<T>,
    output_delay_frames: usize,

    // Staging area for non-planar input/output; GpuCore only speaks per-channel/planar shapes.
    input_staging: Vec<Vec<T>>,
    output_staging: Vec<Vec<T>>,

    finalized: bool,
    input_sample_processed: usize,
    output_sample_processed: usize,
}

#[allow(private_bounds)]
impl<T: Float + GpuScalar + FromF64> InterleavedGpuResampler<T> {
    /// Builds a resampler from an already-constructed [`GpuContext`].
    ///
    /// GPU batching and ring depth are baked into `context` and cannot be changed later.
    /// Passing an existing context lets callers share one Vulkan device and compiled-shader cache
    /// across multiple resamplers.
    pub fn new(context: GpuContext<T>) -> Result<Self, GpuError> {
        let output_delay_frames = context.derived().output_offset;
        let channels = context.config().channels;
        let core = GpuCore::new(context)?;

        let input_staging = vec![vec![T::zero(); core.input_chunk_frames()]; channels];
        let output_staging = vec![vec![T::zero(); core.output_chunk_frames()]; channels];

        Ok(Self {
            core,
            output_delay_frames,
            input_staging,
            output_staging,
            finalized: false,
            input_sample_processed: 0,
            output_sample_processed: 0,
        })
    }

    /// Convenience constructor that auto-selects a Vulkan device and builds its own
    /// [`GpuContext`].
    pub fn with_config(config: Config) -> Result<Self, GpuError> {
        let context = GpuContext::new(config)?;
        Self::new(context)
    }

    /// Returns the configuration this instance was built with.
    #[must_use]
    pub fn config(&self) -> &Config {
        self.core.context().config()
    }

    /// Number of FFT chunks batched into one GPU submission.
    pub fn group_chunks(&self) -> usize {
        self.core.group_chunks()
    }

    /// Number of pre-allocated GPU buffer-sets in the fixed ring.
    pub fn ring_slots(&self) -> usize {
        self.core.ring_slots()
    }

    /// Returns the total number of interleaved input samples processed.
    #[inline]
    pub fn input_sample_processed(&self) -> usize {
        self.input_sample_processed
    }

    /// Returns the total number of interleaved output samples processed.
    #[inline]
    pub fn output_sample_processed(&self) -> usize {
        self.output_sample_processed
    }

    /// Returns the required `input` length (interleaved samples) for each non-final chunk call.
    #[must_use]
    #[inline]
    pub fn input_buffer_size(&self) -> usize {
        self.core.input_chunk_frames() * self.config().channels
    }

    /// Returns the recommended per-call `output` capacity in interleaved samples.
    #[must_use]
    #[inline]
    pub fn output_buffer_size(&self) -> usize {
        self.core.output_chunk_frames() * self.config().channels
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

    /// Pushes one full interleaved chunk into the pipeline without waiting for output.
    ///
    /// `input` must contain exactly [`input_buffer_size()`](Self::input_buffer_size) samples.
    /// May block only if the GPU ring has no free buffer-set (see [`GpuCore::push_input`]).
    pub fn push_chunk(&mut self, input: &[T]) -> Result<(), GpuError> {
        self.push_chunk_inner(input, false)
    }

    /// Pushes the final (possibly undersized) interleaved chunk into the pipeline.
    pub fn push_chunk_final(&mut self, input: &[T]) -> Result<(), GpuError> {
        self.push_chunk_inner(input, true)
    }

    fn push_chunk_inner(&mut self, input: &[T], is_final: bool) -> Result<(), GpuError> {
        if self.finalized {
            return Err(GpuError::AlreadyFinalized);
        }
        self.ensure_input_buffer_shape(input, is_final)?;
        self.deinterleave_into_staging(input);
        self.core.push_input(&self.input_staging, is_final)?;
        self.input_sample_processed += input.len();
        Ok(())
    }

    /// Copies as many interleaved output samples as fit into `output` from what's ready so far.
    /// Never blocks. Unlike the blocking family, `output` may be any size (a multiple of the
    /// channel count) -- pass a small buffer to poll for a small amount of ready output, the same
    /// way [`GpuCore::pull_output`] accepts channel slices of any length.
    pub fn pull_chunk(&mut self, output: &mut [T]) -> Result<usize, GpuError> {
        let channels = self.config().channels;
        if !output.len().is_multiple_of(channels) {
            return Err(GpuError::MalformedInputLength {
                channels,
                samples: output.len(),
            });
        }
        let requested_frames = (output.len() / channels).min(self.core.output_chunk_frames());
        let mut staging_refs: Vec<&mut [T]> = self
            .output_staging
            .iter_mut()
            .map(|channel| &mut channel[..requested_frames])
            .collect();
        let written_frames = self.core.pull_output(&mut staging_refs[..])?;
        let written_samples = written_frames * channels;
        self.interleave_staging_into(output, written_frames);
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
        for channel in &mut self.input_staging {
            channel.clear();
        }
        for channel in &mut self.output_staging {
            channel.clear();
        }
        self.finalized = false;
        self.input_sample_processed = 0;
        self.output_sample_processed = 0;
        Ok(())
    }

    /// Resamples a complete interleaved input buffer and returns all output samples as planar
    /// channels.
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
    /// [`crate::InterleavedResampler::process_all`]'s behavior of never requiring an explicit
    /// [`reset()`](Self::reset) between calls.
    pub fn process_all(&mut self, input: &[T]) -> Result<PlanarVecs<T>, GpuError> {
        if self.finalized {
            self.reset()?;
        }
        let channels = self.config().channels;
        if !input.len().is_multiple_of(channels) {
            return Err(GpuError::MalformedInputLength {
                channels,
                samples: input.len(),
            });
        }

        let input_planar: Vec<Vec<T>> = (0..channels)
            .map(|c| input[c..].iter().step_by(channels).copied().collect())
            .collect();
        let refs: Vec<&[T]> = input_planar.iter().map(Vec::as_slice).collect();

        let output = run_stream_to_completion(&mut self.core, &refs[..])?;
        self.finalized = true;
        self.input_sample_processed += input.len();
        self.output_sample_processed += output.iter().map(Vec::len).sum::<usize>();

        Ok(PlanarVecs::new(output).expect("channel lengths are equal by construction"))
    }

    /// Processes an interleaved chunk and blocks until its output is available.
    ///
    /// Convenience wrapper equivalent to [`push_chunk`](Self::push_chunk) followed by
    /// [`flush`](Self::flush) and a full drain. This forces every chunk through its own GPU
    /// submission, forgoing the throughput benefit of `group_chunks` batching -- prefer the
    /// non-blocking family ([`push_chunk`](Self::push_chunk)/[`pull_chunk`](Self::pull_chunk)) for
    /// GPU-batched throughput.
    pub fn process_chunk(&mut self, input: &[T], output: &mut [T]) -> Result<usize, GpuError> {
        self.push_chunk(input)?;
        self.core.flush()?;
        self.drain_all_ready_into(output)
    }

    /// Processes the final (possibly undersized) interleaved chunk and blocks until its output is
    /// available. See [`process_chunk`](Self::process_chunk) for the batching tradeoff.
    pub fn process_chunk_final(&mut self, input: &[T], output: &mut [T]) -> Result<usize, GpuError> {
        self.push_chunk_final(input)?;
        self.core.flush()?;
        self.drain_all_ready_into(output)
    }

    /// Emits delayed tail samples, then marks the stream finalized.
    ///
    /// Blocks until all outstanding GPU work completes. This is the terminal step of a stream and
    /// should be called once per stream.
    pub fn finalize(&mut self, output: &mut [T]) -> Result<usize, GpuError> {
        if self.finalized {
            return Err(GpuError::AlreadyFinalized);
        }
        self.core.finalize()?;
        let written = self.drain_all_ready_into(output)?;
        self.finalized = true;
        Ok(written)
    }

    /// Loops [`GpuCore::pull_output`] until it returns `0`, re-interleaving each round into
    /// `output`. Used by the blocking API family, where every push is followed by a `flush()` so
    /// all resulting output is guaranteed ready by the time this is called.
    fn drain_all_ready_into(&mut self, output: &mut [T]) -> Result<usize, GpuError> {
        self.ensure_output_buffer_shape(output)?;
        let mut total_written = 0;
        loop {
            let written_frames = self.core.pull_output(&mut self.output_staging)?;
            if written_frames == 0 {
                break;
            }
            let written_samples = written_frames * self.config().channels;
            if total_written + written_samples > output.len() {
                return Err(GpuError::InsufficientOutputBuffer {
                    expected: total_written + written_samples,
                    actual: output.len(),
                });
            }
            self.interleave_staging_into(&mut output[total_written..], written_frames);
            total_written += written_samples;
        }
        self.output_sample_processed += total_written;
        Ok(total_written)
    }

    /// Sets previous-track context. See [`crate::InterleavedResampler::pre`].
    pub fn pre(&mut self, pre: Vec<T>) -> Result<(), GpuError> {
        if !pre.len().is_multiple_of(self.config().channels) {
            return Err(GpuError::MalformedInputLength {
                channels: self.config().channels,
                samples: pre.len(),
            });
        }

        let channels = self.config().channels;
        let max_samples = self.input_buffer_size();
        let start = pre.len().saturating_sub(max_samples);
        let start = start.div_ceil(channels) * channels;

        let deinterleaved: Vec<Vec<T>> = (0..channels)
            .map(|c| pre[start + c..].iter().step_by(channels).copied().collect())
            .collect();
        self.core.pre(deinterleaved);
        Ok(())
    }

    /// Sets next-track context. See [`crate::InterleavedResampler::post`].
    pub fn post(&mut self, post: Vec<T>) -> Result<(), GpuError> {
        if !post.len().is_multiple_of(self.config().channels) {
            return Err(GpuError::MalformedInputLength {
                channels: self.config().channels,
                samples: post.len(),
            });
        }

        let channels = self.config().channels;
        let end = post.len().min(self.input_buffer_size());

        let deinterleaved: Vec<Vec<T>> = (0..channels)
            .map(|c| post[c..end].iter().step_by(channels).copied().collect())
            .collect();
        self.core.post(deinterleaved);
        Ok(())
    }

    /// Process multiple independent tracks.
    ///
    /// Each input slice is treated as its own stream with no inter-track context. See
    /// [`batch_gapless()`](Self::batch_gapless) for gapless processing of multiple tracks.
    ///
    /// De-interleaves each track, then delegates to a [`PlanarGpuResampler`] built from a
    /// [`GpuContext::clone_shared`] clone of this resampler's context (sharing the same Vulkan
    /// device and already-compiled shaders, so no per-track shader recompilation) -- mirroring
    /// how [`crate::InterleavedResampler::batch`] delegates to [`crate::PlanarResampler`] today.
    /// See [`PlanarGpuResampler::batch`] for the per-track `GpuCore`/parallelism notes.
    pub fn batch(&self, inputs: &[&[T]]) -> Result<Vec<PlanarVecs<T>>, GpuError>
    where
        T: Send + Sync,
    {
        let prepared_inputs = self.batch_prepare_interleaved_inputs(inputs)?;
        let planar = PlanarGpuResampler::new(self.core.context().clone_shared())?;
        planar.batch(prepared_inputs)
    }

    /// Process multiple tracks as one gapless sequence.
    ///
    /// Adjacent inputs are treated as tracks from the same album or other back-to-back material.
    /// Each track is returned separately, but the previous track's tail and next track's head are
    /// used as edge context to improve gapless playback. See [`batch()`](Self::batch) for the
    /// delegation/parallelism notes.
    pub fn batch_gapless(&self, inputs: &[&[T]]) -> Result<Vec<PlanarVecs<T>>, GpuError>
    where
        T: Send + Sync,
    {
        let prepared_inputs = self.batch_prepare_interleaved_inputs(inputs)?;
        let planar = PlanarGpuResampler::new(self.core.context().clone_shared())?;
        planar.batch_gapless(prepared_inputs)
    }

    fn batch_prepare_interleaved_inputs(&self, inputs: &[&[T]]) -> Result<Vec<PlanarVecs<T>>, GpuError>
    where
        T: Send + Sync,
    {
        // `channels` is extracted up front rather than read via `self.config().channels` inside the
        // closures below: capturing `self` (and thus `InterleavedGpuResampler`'s `GpuCore`, which
        // is deliberately not `Sync`) would make the closure unusable with `rayon`.
        let channels = self.config().channels;

        #[cfg(feature = "rayon")]
        {
            inputs
                .par_iter()
                .map(|input| Self::batch_deinterleave_track(input, channels))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            inputs
                .iter()
                .map(|input| Self::batch_deinterleave_track(input, channels))
                .collect()
        }
    }

    fn batch_deinterleave_track(input: &[T], channels: usize) -> Result<PlanarVecs<T>, GpuError> {
        if !input.len().is_multiple_of(channels) {
            return Err(GpuError::MalformedInputLength {
                channels,
                samples: input.len(),
            });
        }

        let frames = input.len() / channels;
        let mut per_channel = vec![Vec::with_capacity(frames); channels];
        for frame in input.chunks_exact(channels) {
            for (channel_idx, sample) in frame.iter().copied().enumerate() {
                per_channel[channel_idx].push(sample);
            }
        }

        Ok(PlanarVecs::new(per_channel).expect("channel lengths are equal by construction"))
    }

    #[inline]
    fn ensure_input_buffer_shape(&self, input: &[T], is_final: bool) -> Result<(), GpuError> {
        if !input.len().is_multiple_of(self.config().channels) {
            return Err(GpuError::MalformedInputLength {
                channels: self.config().channels,
                samples: input.len(),
            });
        }

        let frames = input.len() / self.config().channels;
        let expected_frames = self.core.input_chunk_frames();
        if (!is_final && frames != expected_frames) || (is_final && frames > expected_frames) {
            return Err(GpuError::WrongFrameCount {
                expected: expected_frames,
                actual: frames,
            });
        }

        Ok(())
    }

    #[inline]
    fn ensure_output_buffer_shape(&self, output: &[T]) -> Result<(), GpuError> {
        let expected = self.output_buffer_size();
        if output.len() < expected {
            return Err(GpuError::InsufficientOutputBuffer {
                expected,
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn deinterleave_into_staging(&mut self, input: &[T]) {
        let channels = self.config().channels;
        let frames = input.len() / channels;

        for channel in &mut self.input_staging {
            channel.resize(frames, T::zero());
        }

        for (frame_idx, frame) in input.chunks_exact(channels).enumerate() {
            for (channel_idx, sample) in frame.iter().copied().enumerate() {
                self.input_staging[channel_idx][frame_idx] = sample;
            }
        }
    }

    fn interleave_staging_into(&self, output: &mut [T], frames: usize) {
        let channels = self.config().channels;
        for frame_idx in 0..frames {
            for channel_idx in 0..channels {
                output[frame_idx * channels + channel_idx] = self.output_staging[channel_idx][frame_idx];
            }
        }
    }
}

/// Drives `core` through one complete stream given full per-channel `input`, using [`GpuCore`]'s
/// own async push/pull/poll surface directly rather than flushing after every chunk, so
/// `group_chunks` batching is preserved even though this returns a complete, synchronous result.
/// Used by [`InterleavedGpuResampler::process_all`] against the instance's own core.
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
            let refs: Vec<&[T]> = input
                .iter()
                .map(|channel| &channel[offset..offset + this_chunk])
                .collect();
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
    use crate::{GpuDevice, InterleavedResampler};

    const RING_SLOTS: usize = 4;
    const GROUP_CHUNKS: usize = 4;

    fn mono_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config::new(input_sample_rate, output_sample_rate, 1)
    }

    fn stereo_config(input_sample_rate: usize, output_sample_rate: usize) -> Config {
        Config::new(input_sample_rate, output_sample_rate, 2)
    }

    fn tone_interleaved(channels: usize, frames: usize, sample_rate: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * channels);
        for i in 0..frames {
            for c in 0..channels {
                let hz = 400.0 + 137.0 * c as f64;
                out.push((2.0 * std::f64::consts::PI * hz * i as f64 / sample_rate as f64).sin() as f32);
            }
        }
        out
    }

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Builds a GPU wrapper for `config`, or returns `None` (with a diagnostic on stderr) when no
    /// usable Vulkan device is present, so tests skip gracefully in environments without a GPU.
    fn try_build_gpu(config: Config, group_chunks: usize) -> Option<InterleavedGpuResampler<f32>> {
        let device = match GpuDevice::auto_select() {
            Ok(device) => Arc::new(device),
            Err(err) => {
                eprintln!("skipping GPU test: {err}");
                return None;
            }
        };
        let config = config
            .with_gpu_group_chunks(group_chunks)
            .with_gpu_ring_slots(RING_SLOTS);
        let context = GpuContext::with_device(device, config).expect("build GpuContext");
        Some(InterleavedGpuResampler::new(context).expect("build InterleavedGpuResampler"))
    }

    /// Builds real pre/input/post interleaved context around `total_frames` of useful input, and
    /// returns the derived per-channel chunk frame count. Real pre/post context sidesteps the
    /// known, separately tracked `Extrapolation::Lpc`/GPU divergence without it (see
    /// `gpu_core.rs`'s `lpc_extrapolation_has_known_gpu_divergence_without_context`).
    fn make_pre_input_post(config: &Config, total_frames: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let derived = config.derive_config::<f32>().unwrap();
        let chunk_frames = derived.raw_input_chunk_frames();
        let channels = config.channels;
        let full = tone_interleaved(channels, chunk_frames * 2 + total_frames, config.input_sample_rate);
        let pre = full[..chunk_frames * channels].to_vec();
        let input = full[chunk_frames * channels..(chunk_frames + total_frames) * channels].to_vec();
        let post = full[(chunk_frames + total_frames) * channels..].to_vec();
        (pre, input, post, chunk_frames)
    }

    fn cpu_reference(config: Config, input: &[f32], pre: &[f32], post: &[f32]) -> Vec<f32> {
        let mut resampler = InterleavedResampler::<f32>::new(config).unwrap();
        resampler.pre(pre.to_vec()).unwrap();
        resampler.post(post.to_vec()).unwrap();

        let chunk = resampler.input_buffer_size();
        let mut scratch = vec![0.0f32; resampler.output_buffer_size()];
        let mut output = Vec::new();
        let mut offset = 0;
        while offset + chunk <= input.len() {
            let written = resampler
                .process_chunk(&input[offset..offset + chunk], &mut scratch)
                .unwrap();
            output.extend_from_slice(&scratch[..written]);
            offset += chunk;
        }
        let written = resampler.process_chunk_final(&input[offset..], &mut scratch).unwrap();
        output.extend_from_slice(&scratch[..written]);
        let written = resampler.finalize(&mut scratch).unwrap();
        output.extend_from_slice(&scratch[..written]);
        output
    }

    /// Streams `input` through `gpu` via the blocking API family and returns the full output.
    fn gpu_blocking_stream(gpu: &mut InterleavedGpuResampler<f32>, input: &[f32]) -> Vec<f32> {
        let chunk = gpu.input_buffer_size();
        let mut scratch = vec![0.0f32; gpu.output_buffer_size()];
        let mut output = Vec::new();
        let mut offset = 0;
        while offset + chunk <= input.len() {
            let written = gpu.process_chunk(&input[offset..offset + chunk], &mut scratch).unwrap();
            output.extend_from_slice(&scratch[..written]);
            offset += chunk;
        }
        let written = gpu.process_chunk_final(&input[offset..], &mut scratch).unwrap();
        output.extend_from_slice(&scratch[..written]);
        let written = gpu.finalize(&mut scratch).unwrap();
        output.extend_from_slice(&scratch[..written]);
        output
    }

    /// Streams `input` through `gpu` via the non-blocking `push_chunk`/`pull_chunk` API, draining
    /// with a deliberately tiny scratch buffer to exercise multi-call partial drains.
    fn gpu_nonblocking_stream(gpu: &mut InterleavedGpuResampler<f32>, input: &[f32]) -> Vec<f32> {
        let chunk = gpu.input_buffer_size();
        let mut tiny = [0.0f32; 37];
        let mut output = Vec::new();
        let mut offset = 0;
        loop {
            let remaining = input.len() - offset;
            let is_final = remaining <= chunk;
            let this_chunk = if is_final { remaining } else { chunk };
            if is_final {
                gpu.push_chunk_final(&input[offset..offset + this_chunk]).unwrap();
            } else {
                gpu.push_chunk(&input[offset..offset + this_chunk]).unwrap();
            }
            loop {
                let written = gpu.pull_chunk(&mut tiny).unwrap();
                output.extend_from_slice(&tiny[..written]);
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
            let written = gpu.pull_chunk(&mut tiny).unwrap();
            output.extend_from_slice(&tiny[..written]);
            if written == 0 {
                break;
            }
        }
        output
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

        let cpu_output = cpu_reference(config, &input, &pre, &post);
        let gpu_output = gpu_blocking_stream(&mut gpu, &input);

        assert_eq!(cpu_output.len(), gpu_output.len());
        assert!(max_abs_error(&cpu_output, &gpu_output) < 1e-3);
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

        let cpu_output = cpu_reference(config, &input, &pre, &post);
        let gpu_output = gpu_blocking_stream(&mut gpu, &input);

        assert_eq!(cpu_output.len(), gpu_output.len());
        assert!(max_abs_error(&cpu_output, &gpu_output) < 1e-3);
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

        let cpu_output = cpu_reference(config, &input, &pre, &post);
        let gpu_output = gpu_nonblocking_stream(&mut gpu, &input);

        assert_eq!(chunk_frames, gpu.core.input_chunk_frames());
        assert_eq!(cpu_output.len(), gpu_output.len());
        assert!(max_abs_error(&cpu_output, &gpu_output) < 1e-3);
    }

    #[test]
    fn flush_makes_partial_group_output_available_without_ending_stream() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let chunk = gpu.input_buffer_size();
        let input = vec![0.5f32; chunk];

        gpu.push_chunk(&input).unwrap();
        let mut scratch = vec![0.0f32; gpu.output_buffer_size()];
        assert_eq!(
            gpu.pull_chunk(&mut scratch).unwrap(),
            0,
            "output shouldn't be ready before an explicit flush"
        );

        gpu.flush().unwrap();
        assert!(
            gpu.pull_chunk(&mut scratch).unwrap() > 0,
            "flush should force the partial group out"
        );
    }

    #[test]
    fn malformed_input_length_is_rejected() {
        let config = stereo_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let odd = vec![0.0f32; 9];
        assert!(matches!(
            gpu.push_chunk(&odd),
            Err(GpuError::MalformedInputLength {
                channels: 2,
                samples: 9
            })
        ));
        assert!(matches!(
            gpu.pre(odd.clone()),
            Err(GpuError::MalformedInputLength {
                channels: 2,
                samples: 9
            })
        ));
        assert!(matches!(
            gpu.post(odd),
            Err(GpuError::MalformedInputLength {
                channels: 2,
                samples: 9
            })
        ));
    }

    #[test]
    fn wrong_frame_count_is_rejected() {
        let config = mono_config(44_100, 48_000);
        let Some(mut gpu) = try_build_gpu(config, GROUP_CHUNKS) else {
            return;
        };
        let too_short = vec![0.0f32; gpu.input_buffer_size() - 1];
        assert!(matches!(
            gpu.push_chunk(&too_short),
            Err(GpuError::WrongFrameCount { .. })
        ));

        let too_long_final = vec![0.0f32; gpu.input_buffer_size() + 1];
        assert!(matches!(
            gpu.push_chunk_final(&too_long_final),
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
        let mut too_small = vec![0.0f32; gpu.output_buffer_size() - 1];
        assert!(matches!(
            gpu.process_chunk(&input, &mut too_small),
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
        let mut scratch = vec![0.0f32; gpu.output_buffer_size()];
        gpu.process_chunk(&input, &mut scratch).unwrap();
        gpu.finalize(&mut scratch).unwrap();
        assert!(gpu.is_finalized());
        assert!(matches!(gpu.finalize(&mut scratch), Err(GpuError::AlreadyFinalized)));

        gpu.reset().unwrap();
        assert!(!gpu.is_finalized());
        assert_eq!(gpu.input_sample_processed(), 0);
        assert_eq!(gpu.output_sample_processed(), 0);

        // A second, independent stream through the same instance should behave like a fresh one.
        let written = gpu.process_chunk(&input, &mut scratch).unwrap();
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

    fn chunk_frames_probe(config: &Config) -> usize {
        config.derive_config::<f32>().unwrap().raw_input_chunk_frames()
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

        let cpu_output = cpu_reference(config, &input, &pre, &post);
        let gpu_output = gpu.process_all(&input).unwrap().interleave();

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

        let first = gpu.process_all(&input).unwrap();
        assert!(gpu.is_finalized());

        // No explicit reset() call here -- process_all should reset on its own since the last
        // stream was already fully drained and finalized.
        let second = gpu.process_all(&input).unwrap();
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
        let tracks: Vec<Vec<f32>> = vec![
            (0..(chunk + 17)).map(|f| (f as f32 * 0.009).sin() * 0.2).collect(),
            (0..(chunk * 2 + 5)).map(|f| (f as f32 * 0.012).cos() * 0.15).collect(),
            (0..(chunk / 2 + 11)).map(|f| (f as f32 * 0.021).sin() * 0.25).collect(),
        ];
        let input_refs: Vec<&[f32]> = tracks.iter().map(Vec::as_slice).collect();

        let expected: Vec<Vec<f32>> = tracks
            .iter()
            .map(|track| {
                let mut solo = try_build_gpu(config.clone(), GROUP_CHUNKS).unwrap();
                solo.process_all(track).unwrap().get_channel(0).unwrap().to_vec()
            })
            .collect();

        let actual = gpu.batch(&input_refs).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual_track, expected_track) in actual.iter().zip(expected.iter()) {
            let actual_channel = actual_track.get_channel(0).unwrap();
            assert_eq!(actual_channel.len(), expected_track.len());
            assert!(max_abs_error(actual_channel, expected_track) < 1e-3);
        }
    }

    #[test]
    fn batch_gapless_matches_planar_gapless() {
        let config = mono_config(44_100, 48_000);
        let Some(interleaved) = try_build_gpu(config.clone(), GROUP_CHUNKS) else {
            return;
        };
        let planar = PlanarGpuResampler::<f32>::with_config(
            config
                .clone()
                .with_gpu_group_chunks(GROUP_CHUNKS)
                .with_gpu_ring_slots(RING_SLOTS),
        )
        .unwrap();
        let context_chunk_size = chunk_frames_probe(&config);
        let track_frames = context_chunk_size * 2 + 17;

        let tracks: Vec<Vec<f32>> = (0..3)
            .map(|track_idx| {
                (0..track_frames)
                    .map(|frame| {
                        let continuous_frame = track_idx * track_frames + frame;
                        (continuous_frame as f32 * 0.017).sin() * 0.25
                    })
                    .collect()
            })
            .collect();
        let planar_tracks = tracks
            .iter()
            .map(|track| PlanarVecs::new(vec![track.clone()]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let input_refs = tracks.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let interleaved_outputs = interleaved.batch_gapless(&input_refs).unwrap();
        let planar_outputs = planar.batch_gapless(planar_tracks).unwrap();

        assert_eq!(interleaved_outputs.len(), planar_outputs.len());
        for (interleaved_output, planar_output) in interleaved_outputs.iter().zip(planar_outputs.iter()) {
            assert_eq!(interleaved_output, planar_output);
        }
    }
}
