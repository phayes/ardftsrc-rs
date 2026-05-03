use std::sync::Arc;

use num_traits::Float;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, FftNum, RealFftPlanner, RealToComplex};

use crate::Error;
use crate::config::DerivedConfig;
use crate::lpc::{ExtrapolateFallback, extrapolate_backward, extrapolate_forward};

pub(crate) struct ArdftsrcCore<T = f32>
where
    T: Float + FftNum,
{
    /// Precomputed FFT/chunk/offset dimensions and taper.
    derived: DerivedConfig<T>,
    /// Planned forward real FFT instance reused across all chunks.
    forward: Arc<dyn RealToComplex<T>>,
    /// Planned inverse real FFT that maps resized spectra back into time-domain output windows.
    inverse: Arc<dyn ComplexToReal<T>>,
    /// Reusable temporary FFT buffers for the hot path.
    scratch: Scratch<T>,
    /// Overlap buffer that carries second-half iFFT energy into the next output block.
    overlap: Vec<T>,
    /// Output block staging buffer used before delay-trim copy into caller output.
    output_block: Vec<T>,
    /// Previous input window used for stop-edge extrapolation.
    prev_input_window: Vec<T>,
    /// Set when the final chunk is accepted so later chunk calls are ignored by stream contract.
    final_input_seen: bool,
    /// One-shot guard that enforces flush semantics and prevents duplicate tail emission.
    flushed: bool,
    /// Remaining output samples to skip so algorithmic startup delay is trimmed exactly once.
    trim_remaining: usize,
    /// Remaining tail samples to emit on flush after accounting for short final-chunk padding.
    flush_remaining: usize,
    /// Optional previous-track tail used as real start-edge context.
    pre: Option<Vec<T>>,
    /// Optional next-track head used as real stop-edge context.
    post: Option<Vec<T>>,
    /// Total number of input samples for the current stream.
    input_sample_count: usize,
    /// Total number of output samples for the current stream.
    output_sample_count: usize,
}

/// Reusable FFT working buffers for one transform pass.
///
/// This groups temporary vectors mutated on each chunk transform so the hot path can avoid
/// repeated heap allocation and keep a stable memory layout.
struct Scratch<T>
where
    T: Float + FftNum,
{
    /// Time-domain window fed to forward FFT; receives input at configured offset.
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

impl<T> ArdftsrcCore<T>
where
    T: Float + FftNum,
{
    fn input_chunk_len_samples(&self) -> usize {
        self.derived.input_chunk_frames
    }

    fn output_chunk_len_samples(&self) -> usize {
        self.derived.output_chunk_frames
    }

    /// Constructs a single-channel core resampler from `derived`.
    ///
    /// Returns a ready-to-use core instance.
    pub fn new(derived: DerivedConfig<T>) -> Result<Self, Error> {
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
        let overlap = vec![T::zero(); derived.output_chunk_frames];
        let output_block = vec![T::zero(); derived.output_chunk_frames];
        let prev_input_window = vec![T::zero(); derived.input_chunk_frames * 2];

        Ok(Self {
            derived,
            forward,
            inverse,
            scratch,
            overlap,
            output_block,
            prev_input_window,
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

    /// Returns the total number of input samples processed.
    pub(crate) fn input_sample_count(&self) -> usize {
        self.input_sample_count
    }

    /// Returns the required non-final streaming chunk length in samples.
    pub(crate) fn input_chunk_samples(&self) -> usize {
        self.input_chunk_len_samples()
    }

    /// Returns the required `input` length for each `process_chunk()` call.
    ///
    /// Use this to allocate/read fixed-size streaming input buffers.
    pub(crate) fn input_buffer_size(&self) -> usize {
        self.input_chunk_samples()
    }

    /// Returns produced sample count for each non-passthrough transform block before trimming.
    pub(crate) fn output_chunk_samples(&self) -> usize {
        self.output_chunk_len_samples()
    }

    /// Returns the recommended per-call `output` capacity in samples.
    ///
    /// For chunked streaming, size output slices passed to chunk processing to at least this value.
    pub(crate) fn output_buffer_size(&self) -> usize {
        self.output_chunk_samples()
    }

    /// Sets previous-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the tail samples of the previous track.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the end of the previous track.
    /// - Query chunk size with `input_chunk_samples()`, or `input_buffer_size()`.
    ///
    /// Shorter buffers are still valid: any missing start context falls back to LPC
    /// extrapolation.
    pub fn pre(&mut self, pre: Vec<T>) -> Result<(), Error> {
        self.pre = self.normalize_context(pre);
        Ok(())
    }

    /// Sets next-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the head samples of the next track.
    ///
    /// This can be called any time before the final chunk is processed (`is_final = true`). If called
    /// multiple times, the most recent value is used for stop-edge tail prediction.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the start of the next track.
    /// - Query chunk size with `input_chunk_samples()`, or `input_buffer_size()`.
    ///
    /// Shorter buffers are still valid: any missing stop context falls back to LPC
    /// extrapolation.
    pub fn post(&mut self, post: Vec<T>) -> Result<(), Error> {
        self.post = self.normalize_context(post);
        Ok(())
    }

    /// Output samples for a complete input length.
    ///
    /// Returns the ceil-rounded number of output samples expected for `input_samples`.
    pub fn output_sample_count_for_input(&self, input_samples: usize) -> usize {
        (input_samples * self.derived.output_sample_rate).div_ceil(self.derived.input_sample_rate)
    }

    /// Output samples needed for a complete input length.
    ///
    /// This can be used to size the output buffer for the entire input stream.
    pub fn output_sample_count(&self, input_samples: usize) -> usize {
        self.output_sample_count_for_input(input_samples)
    }

    /// Resamples a complete single-channel input buffer and returns all output samples.
    ///
    /// This is a convenience wrapper around the chunked core API.
    pub fn process_all(&mut self, input: &[T]) -> Result<Vec<T>, Error> {
        let expected_samples = self.output_sample_count(input.len());
        let mut output = Vec::with_capacity(expected_samples);

        let mut offset = 0;
        let input_chunk_size = self.input_buffer_size();
        let mut chunk_output = vec![T::zero(); self.output_buffer_size()];

        while offset + input_chunk_size <= input.len() {
            let written = self.process_chunk(&input[offset..offset + input_chunk_size], &mut chunk_output, false)?;
            output.extend_from_slice(&chunk_output[..written]);
            offset += input_chunk_size;
        }

        let written = self.process_chunk(&input[offset..], &mut chunk_output, true)?;
        output.extend_from_slice(&chunk_output[..written]);

        let written = self.finalize(&mut chunk_output)?;
        output.extend_from_slice(&chunk_output[..written]);

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
        self.overlap.fill(T::zero());
        self.output_block.fill(T::zero());
        self.prev_input_window.fill(T::zero());
        self.final_input_seen = false;
        self.flushed = false;
        self.trim_remaining = self.derived.output_offset;
        self.flush_remaining = self.derived.output_offset;
        self.input_sample_count = 0;
        self.output_sample_count = 0;
        self.pre = None;
        self.post = None;
    }

    /// Emits delayed tail samples, then resets stream state.
    ///
    /// This flushes any remaining overlap/delay samples that were held back by the chunked
    /// processing pipeline. It is the terminal step of a stream and should be called once per
    /// stream. If the final chunk was not marked by `process_chunk_inner(..., is_final=true)`,
    /// this treats the last accepted full chunk as terminal input.
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
            let flush_candidate = self.flush_remaining;
            let written_samples = self.cap_write_to_output_budget(flush_candidate);
            debug_assert!(output.len() >= written_samples);
            self.flushed = true;

            self.add_synthetic_finalize_tail_to_overlap()?;

            let scale = T::from(self.output_chunk_len_samples()).unwrap_or(T::one())
                / T::from(self.input_chunk_len_samples()).unwrap_or(T::one());
            for (dst, src) in output[..written_samples]
                .iter_mut()
                .zip(self.overlap[..written_samples].iter())
            {
                *dst = *src * scale;
            }
            written_samples
        };

        self.output_sample_count += written;
        self.reset();
        Ok(written)
    }

    /// Returns expected total output samples once final stream extent is known.
    ///
    /// Before final input is seen, stream extent is unknown and this returns `None`.
    fn expected_total_output_samples(&self) -> Option<usize> {
        if !self.final_input_seen {
            return None;
        }
        Some(self.output_sample_count_for_input(self.input_sample_count))
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

    /// Processes one chunk through the streaming core resampler.
    ///
    /// This internal entry point assumes `input` has already been validated by the caller.
    /// It handles stream-level control flow (passthrough, first/final chunk state, transform
    /// dispatch, and trim/flush accounting).
    ///
    /// Behavior by mode:
    ///
    /// - If final input was already seen, returns `Error::StreamFinished`.
    /// - In passthrough mode (equal rates), it copies input directly to output.
    /// - In FFT mode, it processes one chunk, then applies startup trim and writes contiguous
    ///   samples from `output_block`.
    ///
    /// # Parameters
    ///
    /// - `input`: Input samples for this chunk.
    /// - `output`: Destination buffer for produced output samples.
    /// - `is_final`: Marks this chunk as the final chunk in the stream.
    ///
    /// Returns the number of samples written into `output`.
    pub(crate) fn process_chunk(&mut self, input: &[T], output: &mut [T], is_final: bool) -> Result<usize, Error> {
        let input_samples = input.len();

        if self.final_input_seen {
            return Err(Error::StreamFinished);
        }

        if self.is_passthrough() {
            debug_assert!(output.len() >= input.len());
            if is_final {
                self.final_input_seen = true;
            }
            self.input_sample_count += input.len();
            let written_samples = self.cap_write_to_output_budget(input.len());
            output[..written_samples].copy_from_slice(&input[..written_samples]);
            self.output_sample_count += written_samples;
            return Ok(written_samples);
        }

        debug_assert!(output.len() >= self.output_buffer_size());

        if is_final {
            self.final_input_seen = true;
            if input_samples == 0 {
                return Ok(0);
            }
        }

        self.copy_input_to_window(input, input_samples);

        let is_first_input = self.input_sample_count == 0;
        if is_first_input {
            self.synthesize_start_context(input_samples)?;
            self.copy_input_to_window(input, input_samples);
        }

        let is_short_final = is_final && input_samples < self.input_chunk_len_samples();
        if is_short_final {
            self.synthesize_final_block_missing_samples(input, input_samples);
        }

        self.transform_chunk(TransformMode::Normal)?;

        if !is_short_final {
            self.save_current_window();
        }

        self.input_sample_count += input_samples;

        let skip_samples = self.trim_remaining.min(self.output_chunk_len_samples());
        self.trim_remaining -= skip_samples;
        let chunk_samples_after_trim = self.output_chunk_len_samples() - skip_samples;
        let candidate_samples = chunk_samples_after_trim;
        let written_samples = self.cap_write_to_output_budget(candidate_samples);
        let src_start = skip_samples;
        output[..written_samples].copy_from_slice(&self.output_block[src_start..src_start + written_samples]);

        self.output_sample_count += written_samples;
        Ok(written_samples)
    }

    /// Returns true when rates match and FFT processing can be bypassed losslessly.
    fn is_passthrough(&self) -> bool {
        self.derived.input_sample_rate == self.derived.output_sample_rate
    }

    /// Normalizes empty edge context vectors to `None`.
    fn normalize_context(&self, context: Vec<T>) -> Option<Vec<T>> {
        if context.is_empty() { None } else { Some(context) }
    }

    /// Loads input samples into the FFT window at the configured offset.
    fn copy_input_to_window(&mut self, input: &[T], input_samples: usize) {
        self.scratch.rdft_in.fill(T::zero());
        let dst = &mut self.scratch.rdft_in[self.derived.input_offset..self.derived.input_offset + input_samples];
        dst.copy_from_slice(&input[..input_samples]);
    }

    /// Copies up to `dst.len()` trailing samples from `pre` into `dst`'s tail.
    fn copy_pre_tail(&self, dst: &mut [T]) -> usize {
        let Some(pre) = &self.pre else {
            return 0;
        };
        let copied = pre.len().min(dst.len());
        let start = pre.len() - copied;
        let dst_start = dst.len() - copied;
        dst[dst_start..].copy_from_slice(&pre[start..start + copied]);
        copied
    }

    /// Copies up to `dst.len()` leading samples from `post` into `dst`'s head.
    fn copy_post_head(&self, dst: &mut [T]) -> usize {
        let Some(post) = &self.post else {
            return 0;
        };
        let copied = post.len().min(dst.len());
        dst[..copied].copy_from_slice(&post[..copied]);
        copied
    }

    /// Synthesizes start-edge context by backward extrapolation for the first non-empty chunk.
    ///
    /// Returns `Ok(())` after start context is prepared (or when no work is needed), or an error
    /// if the FFT pipeline fails while staging overlap state.
    fn synthesize_start_context(&mut self, input_samples: usize) -> Result<(), Error> {
        if input_samples == 0 {
            return Ok(());
        }

        let input_start = self.derived.input_offset;
        let input_end = input_start + input_samples;
        let mut predicted = vec![T::zero(); input_start];
        let copied = self.copy_pre_tail(&mut predicted);
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
        let tail_start = self.input_chunk_len_samples();
        self.scratch.rdft_in[tail_start..tail_start + predicted.len()].copy_from_slice(&predicted);
        self.transform_chunk(TransformMode::Start)?;
        Ok(())
    }

    /// Fills a synthetic forward tail from `post` first, then LPC extrapolation fallback.
    fn build_tail_prediction(&self, base: &[T], needed: usize) -> Vec<T> {
        let mut predicted = vec![T::zero(); needed];
        let copied = self.copy_post_head(&mut predicted);
        if copied < needed {
            let mut seed = Vec::with_capacity(base.len() + copied);
            seed.extend_from_slice(base);
            seed.extend_from_slice(&predicted[..copied]);
            let fallback = extrapolate_forward(&seed, needed - copied, ExtrapolateFallback::Hold);
            predicted[copied..].copy_from_slice(&fallback);
        }
        predicted
    }

    /// Builds stop-edge work window from prior history for a final short chunk.
    fn assemble_short_final_work_window(
        &self,
        input: &[T],
        input_samples: usize,
        chunk_samples: usize,
        pad_samples: usize,
    ) -> Vec<T> {
        let mut work = vec![T::zero(); chunk_samples * 2];
        work[..pad_samples].copy_from_slice(&self.prev_input_window[input_samples..input_samples + pad_samples]);
        work[pad_samples..pad_samples + input_samples].copy_from_slice(&input[..input_samples]);
        work
    }

    /// Predicts and writes the synthetic short-final tail into `work`.
    fn fill_short_final_predicted_tail(&self, work: &mut [T], chunk_samples: usize) -> Vec<T> {
        let predicted = self.build_tail_prediction(&work[..chunk_samples], chunk_samples);
        work[chunk_samples..chunk_samples * 2].copy_from_slice(&predicted);
        predicted
    }

    /// Commits short-final history mutations used by later finalize paths.
    fn commit_short_final_history(
        &mut self,
        input_samples: usize,
        pad_samples: usize,
        predicted: &[T],
        chunk_samples: usize,
    ) {
        if input_samples == 0 {
            return;
        }
        self.prev_input_window[..input_samples].copy_from_slice(&predicted[pad_samples..pad_samples + input_samples]);
        self.prev_input_window[input_samples..chunk_samples].fill(T::zero());
        self.prev_input_window[chunk_samples..chunk_samples * 2].fill(T::zero());
    }

    /// Stages synthesized short-final window into `scratch.rdft_in`.
    fn stage_short_final_rdft_input_from_work(&mut self, work: &[T], pad_samples: usize, chunk_samples: usize) {
        self.scratch.rdft_in.fill(T::zero());
        let window_start = self.derived.input_offset;
        self.scratch.rdft_in[window_start..window_start + chunk_samples]
            .copy_from_slice(&work[pad_samples..pad_samples + chunk_samples]);
    }

    /// Builds stop-edge window from prior history for a final short chunk.
    fn synthesize_final_block_missing_samples(&mut self, input: &[T], input_samples: usize) {
        let chunk_samples = self.input_chunk_len_samples();
        let pad_samples = chunk_samples - input_samples;

        let mut work = self.assemble_short_final_work_window(input, input_samples, chunk_samples, pad_samples);
        let predicted = self.fill_short_final_predicted_tail(&mut work, chunk_samples);
        self.commit_short_final_history(input_samples, pad_samples, &predicted, chunk_samples);
        self.stage_short_final_rdft_input_from_work(&work, pad_samples, chunk_samples);
    }

    /// Runs one chunk through the FFT-domain resampling pipeline for the current window.
    ///
    /// This method assumes `self.scratch.rdft_in` has already been prepared (windowing,
    /// zero-padding, and stop-edge preparation if needed). It then:
    ///
    /// - Performs a forward real FFT.
    /// - Copies/tapers frequency bins into `resampled_spectrum` and clears unused bins.
    /// - Enforces real-valued DC/Nyquist bins required by `realfft`.
    /// - Performs an inverse real FFT back into `rdft_out`.
    /// - Applies mode-specific overlap/output handling for steady-state, start-edge priming,
    ///   or finalize-tail accumulation.
    ///
    /// # Parameters
    ///
    /// - `mode`: Selects whether to emit output and how overlap state is updated.
    ///
    /// Returns `Ok(())` on successful transform and overlap/output updates, or an FFT error from
    /// the backend.
    fn transform_chunk(&mut self, mode: TransformMode) -> Result<(), Error> {
        self.forward
            .process(&mut self.scratch.rdft_in, &mut self.scratch.spectrum)
            .map_err(|err| Error::Fft(err.to_string()))?;

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
        if let Some(dc_bin) = self.scratch.resampled_spectrum.get_mut(0) {
            dc_bin.im = T::zero();
        }
        if self.scratch.resampled_spectrum.len() > 1 {
            let nyquist_bin = self.scratch.resampled_spectrum.len() - 1;
            self.scratch.resampled_spectrum[nyquist_bin].im = T::zero();
        }

        self.inverse
            .process(&mut self.scratch.resampled_spectrum, &mut self.scratch.rdft_out)
            .map_err(|err| Error::Fft(err.to_string()))?;

        let normalize = T::one() / T::from(self.derived.output_fft_size).unwrap_or(T::one());
        let scale = T::from(self.output_chunk_len_samples()).unwrap_or(T::one())
            / T::from(self.input_chunk_len_samples()).unwrap_or(T::one());
        let output_chunk_samples = self.output_chunk_len_samples();

        if matches!(mode, TransformMode::Normal) {
            for sample_idx in 0..output_chunk_samples {
                self.output_block[sample_idx] =
                    (self.scratch.rdft_out[sample_idx] * normalize + self.overlap[sample_idx]) * scale;
            }
        }

        if matches!(mode, TransformMode::End) {
            for (overlap, rdft) in self.overlap[..output_chunk_samples]
                .iter_mut()
                .zip(self.scratch.rdft_out[..output_chunk_samples].iter())
            {
                *overlap = *overlap + *rdft * normalize;
            }
        }

        if matches!(mode, TransformMode::Normal | TransformMode::Start) {
            for (overlap, rdft) in self.overlap[..output_chunk_samples]
                .iter_mut()
                .zip(self.scratch.rdft_out[output_chunk_samples..output_chunk_samples * 2].iter())
            {
                *overlap = *rdft * normalize;
            }
        }

        Ok(())
    }

    /// Persists the current window so later stop extrapolation has sample-local history.
    fn save_current_window(&mut self) {
        let chunk_samples = self.input_chunk_len_samples();
        let history_start = self.derived.input_offset;
        let history_end = history_start + chunk_samples;
        self.prev_input_window[..chunk_samples].copy_from_slice(&self.scratch.rdft_in[history_start..history_end]);
        self.prev_input_window[chunk_samples..].fill(T::zero());
    }

    /// Adds synthetic stop tails into overlap when the final chunk was not short.
    ///
    /// If the final chunk was short, we've already done this.
    ///
    /// Returns `Ok(())` after flush overlap is contributed, or an error if the transform fails.
    fn add_synthetic_finalize_tail_to_overlap(&mut self) -> Result<(), Error> {
        if self.final_input_seen && !self.input_sample_count.is_multiple_of(self.input_buffer_size()) {
            return Ok(());
        }

        self.scratch.rdft_in.fill(T::zero());
        let chunk_samples = self.input_chunk_len_samples();
        let input_offset = self.derived.input_offset;
        let base = self.prev_input_window[..chunk_samples].to_vec();
        let predicted = self.build_tail_prediction(&base, input_offset);
        self.prev_input_window[chunk_samples..chunk_samples * 2].fill(T::zero());
        self.prev_input_window[chunk_samples..chunk_samples + predicted.len()].copy_from_slice(&predicted);
        let input_start = self.derived.input_offset;
        self.scratch.rdft_in[input_start..input_start + chunk_samples]
            .copy_from_slice(&self.prev_input_window[chunk_samples..chunk_samples + chunk_samples]);

        self.transform_chunk(TransformMode::End)?;
        Ok(())
    }
}
