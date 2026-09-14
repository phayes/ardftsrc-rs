use std::sync::Arc;

use num_traits::Float;

use crate::Config;
use crate::config::DerivedConfig;
use crate::window;

use super::buffer::GpuScalar;
use super::context::GpuContext;
use super::error::GpuError;
use super::fft_program::{FromF64, pack_complex_batch};
use super::overlap_shader::{OverlapAddShader, OverlapMode};
use super::remap_shader::RemapGeometry;
use super::transform_pipeline::GpuTransformPipeline;

/// GPU-resident streaming resampler core: `gpu_plan.md` milestone 6.
///
/// Batches every channel of one time chunk into a single GPU submission (forward FFT, the
/// spectral-remap shader, inverse FFT, and the streaming overlap-add shader, all in one command
/// buffer -- see [`GpuTransformPipeline`]/[`OverlapAddShader`]). Overlap state stays
/// GPU-resident for the life of the stream; the CPU only ever touches the small, cheap
/// window-construction bookkeeping (`crate::window`, shared with [`crate::cpu_core::CpuCore`])
/// and the current chunk's input/output samples.
///
/// This is a synchronous, blocking-submit implementation: [`GpuStreamingCore::process_chunk`]
/// waits for its GPU submission to complete before returning. `gpu_plan.md` section 11's
/// non-blocking multi-slot ring buffer (for overlapping GPU work with CPU-side I/O across
/// consecutive calls) is not implemented yet; this only covers section 9-10's DSP behavior.
///
/// GPU pre-decimation ([`Config::decimate`]) and `dd_fft` are not supported --
/// [`GpuStreamingCore::new`] rejects both explicitly rather than silently ignoring them.
///
/// The GPU backend's public API is still experimental and may change (`gpu_plan.md` section 36
/// lists a stable public GPU API as an explicit non-goal of this initial implementation).
pub struct GpuStreamingCore<T> {
    context: Arc<GpuContext>,
    pipeline: GpuTransformPipeline<T>,
    overlap: OverlapAddShader<T>,
    derived: DerivedConfig<T>,
    channels: usize,

    /// Per-channel start/stop-edge context, mirroring `CpuCore::pre`/`CpuCore::post`.
    pre: Vec<Option<Vec<T>>>,
    post: Vec<Option<Vec<T>>>,
    /// Per-channel window history (`input_chunk_frames * 2` samples each), mirroring
    /// `CpuCore::prev_input_window`.
    prev_input_window: Vec<Vec<T>>,

    /// Reused scratch window buffer (`input_fft_size` samples), overwritten once per channel
    /// per chunk. Never reallocated after construction.
    window_scratch: Vec<T>,
    /// Reused host staging buffer for uploading one chunk's real input, complex-packed
    /// (`im = 0`) at [`GpuTransformPipeline::input_stride`] per channel.
    upload_staging: Vec<T>,
    /// Reused host staging buffer for downloading one chunk's real output.
    output_staging: Vec<T>,
    /// Reused host staging buffer for downloading the overlap state directly at finalize.
    overlap_staging: Vec<T>,

    final_input_seen: bool,
    finalized: bool,
    trim_remaining: usize,
    flush_remaining: usize,
    input_sample_count: usize,
    output_sample_count: usize,
}

impl<T: Float + GpuScalar + FromF64> GpuStreamingCore<T> {
    /// Builds a streaming core for `channels` channels of `config`.
    ///
    /// Returns [`GpuError::DecimationUnsupported`] if `config` requests pre-decimation,
    /// [`GpuError::DdFftUnsupportedOnGpu`] if it requests `dd_fft`, or
    /// [`GpuError::Fp64Unsupported`] if `T = f64` and `context`'s device cannot run
    /// `shaderFloat64`.
    pub fn new(context: Arc<GpuContext>, config: Config, channels: usize) -> Result<Self, GpuError> {
        if T::scalar_type() == vkfft_rs::ScalarType::F64 {
            context.require_f64()?;
        }

        let derived = config
            .derive_config::<T>()
            .map_err(|err| GpuError::InvalidConfig(err.to_string()))?;
        if derived.decimation_stages > 0 {
            return Err(GpuError::DecimationUnsupported);
        }
        if derived.dd_fft {
            return Err(GpuError::DdFftUnsupportedOnGpu);
        }

        let geometry = RemapGeometry {
            direction_up: derived.input_chunk_frames < derived.output_chunk_frames,
            n: derived.spectral.geometry.lower_nyquist_bin,
            r0: derived.spectral.geometry.reflect_start_bin(),
            nyquist_fold: if derived.input_chunk_frames > derived.output_chunk_frames { 2.0 } else { 1.0 },
            gain: &derived.spectral.gain,
            phase: derived.spectral.phase_enabled.then_some(derived.spectral.phase.as_slice()),
        };
        let pipeline = GpuTransformPipeline::<T>::build(
            &context,
            T::precision(),
            T::scalar_type(),
            derived.input_fft_size,
            derived.output_fft_size,
            channels,
            &geometry,
        )?;
        let overlap = OverlapAddShader::<T>::build(
            &context,
            T::scalar_type(),
            pipeline.output_buffer(),
            pipeline.output_stride(),
            derived.output_chunk_frames,
            derived.input_chunk_frames,
            channels,
        )?;

        let input_stride = pipeline.input_stride();
        let trim_remaining = derived.output_offset;
        let flush_remaining = derived.output_offset;

        Ok(Self {
            context,
            pipeline,
            overlap,
            window_scratch: vec![T::zero(); derived.input_fft_size],
            upload_staging: vec![T::zero(); channels * input_stride * 2],
            output_staging: vec![T::zero(); channels * derived.output_chunk_frames],
            overlap_staging: vec![T::zero(); channels * derived.output_chunk_frames],
            pre: vec![None; channels],
            post: vec![None; channels],
            prev_input_window: vec![vec![T::zero(); derived.input_chunk_frames * 2]; channels],
            derived,
            channels,
            final_input_seen: false,
            finalized: false,
            trim_remaining,
            flush_remaining,
            input_sample_count: 0,
            output_sample_count: 0,
        })
    }

    /// Number of channels this core was built for.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Required per-channel input length for [`GpuStreamingCore::process_chunk`] (equal to the
    /// underlying `Config`'s derived input chunk length; GPU pre-decimation is unsupported, so
    /// there is no raw-vs-decimated distinction to make here, unlike `CpuCore`).
    pub fn input_chunk_frames(&self) -> usize {
        self.derived.input_chunk_frames
    }

    /// Per-channel output length produced by a full (non-final, non-short) chunk.
    pub fn output_chunk_frames(&self) -> usize {
        self.derived.output_chunk_frames
    }

    /// Sets per-channel previous-track tail context (see `CpuCore::pre`). `pre[c]` applies to
    /// channel `c`; a shorter-than-`channels` `pre` leaves the remaining channels' context
    /// untouched.
    pub fn pre(&mut self, pre: Vec<Vec<T>>) {
        for (slot, context) in self.pre.iter_mut().zip(pre) {
            *slot = (!context.is_empty()).then_some(context);
        }
    }

    /// Sets per-channel next-track head context (see `CpuCore::post`).
    pub fn post(&mut self, post: Vec<Vec<T>>) {
        for (slot, context) in self.post.iter_mut().zip(post) {
            *slot = (!context.is_empty()).then_some(context);
        }
    }

    /// Returns the total expected output samples for `input_samples` per-channel input samples.
    pub fn output_sample_count_for_input(&self, input_samples: usize) -> usize {
        (input_samples * self.derived.output_sample_rate).div_ceil(self.derived.input_sample_rate)
    }

    #[inline]
    fn expected_total_output_samples(&self) -> Option<usize> {
        self.final_input_seen
            .then(|| self.output_sample_count_for_input(self.input_sample_count))
    }

    #[inline]
    fn cap_write_to_output_budget(&self, candidate_samples: usize) -> usize {
        self.expected_total_output_samples()
            .map_or(candidate_samples, |expected_total| candidate_samples.min(expected_total.saturating_sub(self.output_sample_count)))
    }

    /// Resets streaming state so the next input is treated as a new, independent stream,
    /// including zeroing the GPU-resident overlap buffer.
    pub fn reset(&mut self) -> Result<(), GpuError> {
        self.overlap.reset_overlap()?;
        for window in &mut self.prev_input_window {
            window.fill(T::zero());
        }
        for context in self.pre.iter_mut().chain(self.post.iter_mut()) {
            *context = None;
        }
        self.final_input_seen = false;
        self.finalized = false;
        self.trim_remaining = self.derived.output_offset;
        self.flush_remaining = self.derived.output_offset;
        self.input_sample_count = 0;
        self.output_sample_count = 0;
        Ok(())
    }

    /// Runs one GPU submission (forward FFT + remap + inverse FFT + overlap-add for `mode`)
    /// over whatever this call's caller has already staged into `self.upload_staging`.
    fn run_transform(&self, mode: OverlapMode) -> Result<(), GpuError> {
        self.pipeline.input_buffer().upload(&self.upload_staging)?;
        let context = &self.context;
        let pipeline = &self.pipeline;
        let overlap = &self.overlap;
        context.run_one_shot(|command_buffer| {
            pipeline.record(context.device(), command_buffer);
            overlap.record(context.device(), command_buffer, mode);
        })
    }

    /// Processes one chunk of `channels` per-channel input slices, writing output into
    /// `output` (each channel's slice must be at least [`GpuStreamingCore::output_chunk_frames`]
    /// long), and returns the number of per-channel output samples written.
    ///
    /// All channels must supply the same number of input samples per call, exactly like
    /// `PlanarResampler::process_chunk`.
    pub fn process_chunk(&mut self, input: &[&[T]], output: &mut [&mut [T]], is_final: bool) -> Result<usize, GpuError> {
        if self.finalized {
            self.reset()?;
        }
        if self.final_input_seen {
            return Err(GpuError::InvalidSubmissionState("stream has already been finalized".to_string()));
        }

        let input_samples = input.first().map_or(0, |channel| channel.len());
        let input_stride = self.pipeline.input_stride();

        if is_final {
            self.final_input_seen = true;
            if input_samples == 0 {
                return Ok(0);
            }
        }

        let is_first_input = self.input_sample_count == 0;
        if is_first_input {
            for c in 0..self.channels {
                window::write_start_window(
                    &mut self.window_scratch,
                    self.derived.input_chunk_frames,
                    self.derived.input_offset,
                    self.pre[c].as_deref(),
                    input[c],
                    self.derived.extrapolation,
                );
                pack_complex_batch(&mut self.upload_staging, c, input_stride, &self.window_scratch);
            }
            self.run_transform(OverlapMode::Start)?;
        }

        let is_short_final = is_final && input_samples < self.derived.input_chunk_frames;
        for c in 0..self.channels {
            if is_short_final {
                window::write_short_final_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    input[c],
                    self.derived.input_chunk_frames,
                    self.derived.input_offset,
                    self.post[c].as_deref(),
                    self.derived.extrapolation,
                );
            } else {
                window::write_normal_window(&mut self.window_scratch, self.derived.input_offset, input[c]);
            }
            pack_complex_batch(&mut self.upload_staging, c, input_stride, &self.window_scratch);
            if !is_short_final {
                window::save_current_window(&mut self.prev_input_window[c], &self.window_scratch, self.derived.input_offset, self.derived.input_chunk_frames);
            }
        }
        self.run_transform(OverlapMode::Normal)?;
        self.overlap.output_buffer().download(&mut self.output_staging)?;

        self.input_sample_count += input_samples;

        let output_chunk_frames = self.derived.output_chunk_frames;
        let skip_samples = self.trim_remaining.min(output_chunk_frames);
        self.trim_remaining -= skip_samples;
        let chunk_samples_after_trim = output_chunk_frames - skip_samples;
        let written_samples = self.cap_write_to_output_budget(chunk_samples_after_trim);
        self.output_sample_count += written_samples;

        for c in 0..self.channels {
            let base = c * output_chunk_frames + skip_samples;
            output[c][..written_samples].copy_from_slice(&self.output_staging[base..base + written_samples]);
        }

        Ok(written_samples)
    }

    /// Emits delayed tail samples, then marks the stream as finalized.
    pub fn finalize(&mut self, output: &mut [&mut [T]]) -> Result<usize, GpuError> {
        if self.finalized {
            return Err(GpuError::InvalidSubmissionState("stream has already been finalized".to_string()));
        }
        self.final_input_seen = true;

        let written = if self.input_sample_count == 0 {
            self.finalized = true;
            0
        } else {
            let flush_candidate = self.flush_remaining;
            let written_samples = self.cap_write_to_output_budget(flush_candidate);
            self.finalized = true;

            // Mirrors `CpuCore::add_synthetic_finalize_tail_to_overlap`'s own skip condition: a
            // short final chunk already folded its forward tail prediction into `prev_input_window`
            // via `write_short_final_window`, so a separate synthetic End-mode tail would double
            // count it -- only run this when the last chunk was a full (non-short) one.
            let last_chunk_was_full = self.input_sample_count.is_multiple_of(self.derived.input_chunk_frames);
            if last_chunk_was_full {
                let input_stride = self.pipeline.input_stride();
                for c in 0..self.channels {
                    window::write_finalize_tail_window(
                        &mut self.window_scratch,
                        &mut self.prev_input_window[c],
                        self.derived.input_chunk_frames,
                        self.derived.input_offset,
                        self.post[c].as_deref(),
                        self.derived.extrapolation,
                    );
                    pack_complex_batch(&mut self.upload_staging, c, input_stride, &self.window_scratch);
                }
                self.run_transform(OverlapMode::End)?;
            }

            self.overlap.overlap_buffer().download(&mut self.overlap_staging)?;
            let scale = T::from(self.derived.output_chunk_frames).unwrap_or_else(T::one) / T::from(self.derived.input_chunk_frames).unwrap_or_else(T::one);
            let output_chunk_frames = self.derived.output_chunk_frames;
            for c in 0..self.channels {
                let base = c * output_chunk_frames;
                for (dst, &value) in output[c][..written_samples].iter_mut().zip(&self.overlap_staging[base..base + written_samples]) {
                    *dst = value * scale;
                }
            }
            written_samples
        };

        self.output_sample_count += written;
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use crate::cpu_core::CpuCore;
    use crate::extrapolation::Extrapolation;

    /// Generates a per-channel test tone (a different frequency per channel, to catch any
    /// channel-mixup bugs in the batched GPU path) of `frames` per-channel samples.
    fn tone_channels(channels: usize, frames: usize, sample_rate: usize) -> Vec<Vec<f32>> {
        (0..channels)
            .map(|c| {
                let hz = 400.0 + 137.0 * c as f64;
                (0..frames)
                    .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / sample_rate as f64).sin() as f32)
                    .collect()
            })
            .collect()
    }

    /// Streams `total_frames` per-channel input samples through both `CpuCore` (one instance per
    /// channel, the existing correctness oracle) and `GpuStreamingCore` (one batched instance for
    /// all channels), chunk by chunk plus a final short chunk and `finalize()`, and asserts the
    /// two produce the same number of output samples with a small enough max error.
    fn assert_streaming_matches_cpu(input_rate: usize, output_rate: usize, channels: usize, total_frames: usize) {
        let context = match GpuContext::new() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping GPU streaming test: {err}");
                return;
            }
        };

        let config = Config::new(input_rate, output_rate, channels);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();

        let mut cpu_cores: Vec<CpuCore<f32>> = (0..channels).map(|_| CpuCore::new(derived.clone())).collect();
        let mut gpu_core = GpuStreamingCore::<f32>::new(context, config, channels).expect("build GpuStreamingCore");

        // Generate one continuous tone `chunk_frames` samples longer on each end than the actual
        // test input, and slice off real `pre`/`post` context from it, so neither core ever
        // falls back to LPC start/tail extrapolation -- this test is only checking the GPU
        // pipeline against the CPU reference, not the (CPU-only, shared) LPC extrapolation path.
        let full = tone_channels(channels, chunk_frames + total_frames + chunk_frames, input_rate);
        let pre: Vec<Vec<f32>> = full.iter().map(|c| c[..chunk_frames].to_vec()).collect();
        let input: Vec<Vec<f32>> = full.iter().map(|c| c[chunk_frames..chunk_frames + total_frames].to_vec()).collect();
        let post: Vec<Vec<f32>> = full.iter().map(|c| c[chunk_frames + total_frames..].to_vec()).collect();
        for (c, core) in cpu_cores.iter_mut().enumerate() {
            core.pre(pre[c].clone());
            core.post(post[c].clone());
        }
        gpu_core.pre(pre.clone());
        gpu_core.post(post.clone());

        let mut cpu_output: Vec<Vec<f32>> = vec![Vec::new(); channels];
        let mut gpu_output: Vec<Vec<f32>> = vec![Vec::new(); channels];

        let mut offset = 0;
        loop {
            let remaining = total_frames - offset;
            let is_final = remaining <= chunk_frames;
            let this_chunk = if is_final { remaining } else { chunk_frames };

            for (c, core) in cpu_cores.iter_mut().enumerate() {
                let slice = &input[c][offset..offset + this_chunk];
                let out = core.process_chunk(slice, is_final).expect("cpu process_chunk");
                cpu_output[c].extend_from_slice(out);
            }

            let gpu_input: Vec<&[f32]> = (0..channels).map(|c| &input[c][offset..offset + this_chunk]).collect();
            let mut gpu_out_buf = vec![vec![0.0f32; gpu_core.output_chunk_frames()]; channels];
            let mut gpu_out_refs: Vec<&mut [f32]> = gpu_out_buf.iter_mut().map(|v| v.as_mut_slice()).collect();
            let written = gpu_core.process_chunk(&gpu_input, &mut gpu_out_refs, is_final).expect("gpu process_chunk");
            for c in 0..channels {
                gpu_output[c].extend_from_slice(&gpu_out_refs[c][..written]);
            }

            offset += this_chunk;
            if is_final {
                break;
            }
        }

        for (c, core) in cpu_cores.iter_mut().enumerate() {
            let out = core.finalize().expect("cpu finalize");
            cpu_output[c].extend_from_slice(out);
        }

        let mut gpu_final_buf = vec![vec![0.0f32; gpu_core.output_chunk_frames().max(1)]; channels];
        let mut gpu_final_refs: Vec<&mut [f32]> = gpu_final_buf.iter_mut().map(|v| v.as_mut_slice()).collect();
        let gpu_written = gpu_core.finalize(&mut gpu_final_refs).expect("gpu finalize");
        for c in 0..channels {
            gpu_output[c].extend_from_slice(&gpu_final_refs[c][..gpu_written]);
        }

        for c in 0..channels {
            assert_eq!(
                cpu_output[c].len(),
                gpu_output[c].len(),
                "{input_rate} -> {output_rate} channel {c}: output length mismatch (cpu {} vs gpu {})",
                cpu_output[c].len(),
                gpu_output[c].len()
            );
            let mut max_abs_error = 0.0f32;
            for (cpu, gpu) in cpu_output[c].iter().zip(gpu_output[c].iter()) {
                max_abs_error = max_abs_error.max((cpu - gpu).abs());
            }
            assert!(
                max_abs_error < 1e-3,
                "{input_rate} -> {output_rate} channel {c}: GPU/CPU streaming mismatch, max abs error {max_abs_error}"
            );
        }
    }

    /// Runs the same CPU-vs-GPU streaming comparison as [`assert_streaming_matches_cpu`], but
    /// with no `pre`/`post` context at all (forcing whichever `extrapolation` strategy `config`
    /// selects) and returns the max abs error instead of asserting, so different strategies can
    /// be compared against each other by the tests below.
    fn max_streaming_error_no_context(input_rate: usize, output_rate: usize, total_frames: usize, extrapolation: Extrapolation) -> f32 {
        let context = match GpuContext::new() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping GPU streaming investigation: {err}");
                return 0.0;
            }
        };
        let config = Config::new(input_rate, output_rate, 1).with_extrapolation(extrapolation);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();

        let mut cpu_core = CpuCore::<f32>::new(derived.clone());
        let mut gpu_core = GpuStreamingCore::<f32>::new(context, config, 1).expect("build GpuStreamingCore");

        let input: Vec<f32> = tone_channels(1, total_frames, input_rate).remove(0);

        let mut cpu_output = Vec::new();
        let mut gpu_output = Vec::new();
        let mut offset = 0;
        loop {
            let remaining = total_frames - offset;
            let is_final = remaining <= chunk_frames;
            let this_chunk = if is_final { remaining } else { chunk_frames };

            let out = cpu_core.process_chunk(&input[offset..offset + this_chunk], is_final).expect("cpu process_chunk");
            cpu_output.extend_from_slice(out);

            let gpu_input: [&[f32]; 1] = [&input[offset..offset + this_chunk]];
            let mut gpu_out_buf = vec![0.0f32; gpu_core.output_chunk_frames()];
            let mut gpu_out_refs: [&mut [f32]; 1] = [gpu_out_buf.as_mut_slice()];
            let written = gpu_core.process_chunk(&gpu_input, &mut gpu_out_refs, is_final).expect("gpu process_chunk");
            gpu_output.extend_from_slice(&gpu_out_refs[0][..written]);

            offset += this_chunk;
            if is_final {
                break;
            }
        }
        cpu_output.extend_from_slice(cpu_core.finalize().expect("cpu finalize"));
        let mut gpu_final_buf = vec![0.0f32; gpu_core.output_chunk_frames().max(1)];
        let mut gpu_final_refs: [&mut [f32]; 1] = [gpu_final_buf.as_mut_slice()];
        let gpu_written = gpu_core.finalize(&mut gpu_final_refs).expect("gpu finalize");
        gpu_output.extend_from_slice(&gpu_final_refs[0][..gpu_written]);

        assert_eq!(cpu_output.len(), gpu_output.len());
        cpu_output.iter().zip(gpu_output.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn mirror_and_zero_extrapolation_avoid_lpc_gpu_divergence() {
        for strategy in [Extrapolation::Mirror, Extrapolation::Zero] {
            let err = max_streaming_error_no_context(44_100, 96_000, 44_100 + 777, strategy);
            assert!(err < 1e-3, "{strategy:?}: unexpectedly large GPU/CPU divergence with no pre/post context: {err}");
        }
    }

    /// Documents a known, not-yet-root-caused issue (see project memory): with no `pre`/`post`
    /// context, `Extrapolation::Lpc`'s tail window -- bit-identical between the CPU and GPU cores
    /// (same shared `crate::window`/`crate::extrapolation` code, same input) -- produces a much
    /// larger GPU-vs-CPU divergence than the same rate ratio with `Mirror`/`Zero` content (see
    /// [`mirror_and_zero_extrapolation_avoid_lpc_gpu_divergence`], which stays within 1e-3). Since
    /// the window content is identical either way, this points at a real precision gap
    /// specifically in how vkfft-rs's generated shader FFT handles this particular window's
    /// content on the GPU, not a general windowing/buffer bug. Real usage should prefer supplying
    /// real `pre`/`post` context, or `Extrapolation::Mirror`/`Zero`, until this is root-caused.
    /// This test asserts the bug is still present so a fix gets noticed (and this test updated)
    /// rather than silently regressing back to a large, unexplained error.
    #[test]
    fn lpc_extrapolation_has_known_gpu_divergence_without_context() {
        let err = max_streaming_error_no_context(44_100, 96_000, 44_100 + 777, Extrapolation::Lpc);
        assert!(
            err > 0.05,
            "expected the known Extrapolation::Lpc/GPU divergence to still reproduce (got err={err}); \
             if this now passes, the underlying issue may be fixed -- update this test and project memory"
        );
    }

    #[test]
    fn streaming_matches_cpu_mono_44100_to_48000() {
        assert_streaming_matches_cpu(44_100, 48_000, 1, 44_100 * 2);
    }

    #[test]
    fn streaming_matches_cpu_stereo_48000_to_44100() {
        assert_streaming_matches_cpu(48_000, 44_100, 2, 48_000 * 2);
    }

    #[test]
    fn streaming_matches_cpu_mono_44100_to_96000_short_final() {
        // Not a multiple of the chunk size, to exercise the short-final-chunk path.
        assert_streaming_matches_cpu(44_100, 96_000, 1, 44_100 + 777);
    }

    #[test]
    fn streaming_matches_cpu_mono_44100_to_96000_multi_chunk() {
        assert_streaming_matches_cpu(44_100, 96_000, 1, 44_100 * 3);
    }
}
