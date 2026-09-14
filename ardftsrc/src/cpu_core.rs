use std::sync::Arc;

use num_traits::Float;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, FftNum, RealFftPlanner, RealToComplex};

use crate::Error;
use crate::config::DerivedConfig;
use crate::decimate::DecimationChain;
use crate::window;

pub struct CpuCore<T = f64>
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
    /// Mark stream as finalized. Will reset on next call to process_chunk().
    finalized: bool,
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
    /// Optional cascade of 2:1 pre-decimation stages run ahead of the FFT pipeline. Empty
    /// (zero-cost passthrough) unless [`Config::decimate`](crate::Config::decimate) is enabled
    /// and the rate ratio warrants it.
    decimation: DecimationChain<T>,
    /// The decimation cascade's own group delay, converted to output-domain samples. Added on
    /// top of `output_offset` when trimming startup silence. Constant for the life of this
    /// instance (derived purely from config).
    decimator_output_delay: usize,
    /// Reused scratch buffer for the decimated chunk, to avoid reallocating every call.
    decimation_scratch: Vec<T>,
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

impl<T> CpuCore<T>
where
    T: Float + FftNum,
{
    #[inline]
    fn input_chunk_len_samples(&self) -> usize {
        self.derived.input_chunk_frames
    }

    #[inline]
    fn output_chunk_len_samples(&self) -> usize {
        self.derived.output_chunk_frames
    }

    /// Constructs a single-channel core resampler from [`derived`](DerivedConfig).
    ///
    /// Returns a ready-to-use core instance.
    pub fn new(derived: DerivedConfig<T>) -> Self {
        let mut planner = RealFftPlanner::<T>::new();
        let forward = plan_forward(&mut planner, derived.input_fft_size, derived.f128);
        let inverse = plan_inverse(&mut planner, derived.output_fft_size, derived.f128);
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
        let decimation = DecimationChain::new(derived.decimation_stages, &derived.decimation_taps);
        // Converts the decimation cascade's raw-domain group delay into an equivalent number of
        // output-domain samples, using the true (pre-decimation) input rate -- this is the same
        // rate ratio the whole stream is nominally converting at, so a delay at the front of the
        // pipeline shows up scaled by that ratio at the output.
        let decimator_output_delay = if derived.decimation_stages > 0 {
            (decimation.raw_group_delay() * derived.output_sample_rate).div_ceil(derived.input_sample_rate)
        } else {
            0
        };

        Self {
            derived,
            forward,
            inverse,
            scratch,
            overlap,
            output_block,
            prev_input_window,
            final_input_seen: false,
            finalized: false,
            trim_remaining: output_offset + decimator_output_delay,
            flush_remaining: output_offset,
            pre: None,
            post: None,
            input_sample_count: 0,
            output_sample_count: 0,
            decimation,
            decimator_output_delay,
            decimation_scratch: Vec::new(),
        }
    }

    /// Returns the total number of input samples processed.
    #[inline]
    pub(crate) fn input_sample_processed(&self) -> usize {
        self.input_sample_count
    }

    #[inline]
    pub(crate) fn output_sample_processed(&self) -> usize {
        self.output_sample_count
    }

    /// Returns the required non-final streaming chunk length in samples, in the raw
    /// (pre-decimation) domain that callers of [`process_chunk()`](Self::process_chunk) work in.
    #[inline]
    pub(crate) fn input_chunk_samples(&self) -> usize {
        self.derived.raw_input_chunk_frames()
    }

    /// Returns the required `input` length for each [`process_chunk()`](Self::process_chunk) call.
    ///
    /// Use this to allocate/read fixed-size streaming input buffers.
    #[inline]
    pub(crate) fn input_buffer_size(&self) -> usize {
        self.input_chunk_samples()
    }

    /// Sets previous-track context.
    ///
    /// Use this when resampling gapless material, for example an album where tracks are played
    /// back-to-back. In that case, pass the tail samples of the previous track.
    ///
    /// Recommended size:
    ///
    /// - Pass one full input chunk from the end of the previous track.
    /// - Query chunk size with [`input_chunk_samples()`](Self::input_chunk_samples), or [`input_buffer_size()`](Self::input_buffer_size).
    ///
    /// Shorter buffers are still valid: any missing start context falls back to LPC
    /// extrapolation.
    #[inline]
    pub fn pre(&mut self, pre: Vec<T>) {
        let pre = self.decimate_context(&pre);
        self.pre = self.normalize_context(pre);
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
    /// - Query chunk size with [`input_chunk_samples()`](Self::input_chunk_samples), or [`input_buffer_size()`](Self::input_buffer_size).
    ///
    /// Shorter buffers are still valid: any missing stop context falls back to LPC
    /// extrapolation.
    #[inline]
    pub fn post(&mut self, post: Vec<T>) {
        let post = self.decimate_context(&post);
        self.post = self.normalize_context(post);
    }

    /// Runs raw-domain edge context through a fresh (zero-state) decimation chain, matching what
    /// the streaming path does to real input. Uses a scratch chain rather than `self.decimation`
    /// so setting `pre`/`post` never perturbs the live streaming filter state.
    fn decimate_context(&self, raw: &[T]) -> Vec<T> {
        if self.decimation.num_stages() == 0 {
            return raw.to_vec();
        }
        let mut chain = DecimationChain::new(self.decimation.num_stages(), &self.derived.decimation_taps);
        let mut out = Vec::new();
        chain.process(raw, &mut out);
        // Context is a one-shot static buffer (not a continuing stream), so flush immediately to
        // recover the trailing samples that would otherwise be left stuck in the delay lines.
        let mut flushed = Vec::new();
        chain.flush(&mut flushed);
        out.extend_from_slice(&flushed);
        out
    }

    /// Output samples for a complete input length.
    ///
    /// Returns the ceil-rounded number of output samples expected for `input_samples`.
    #[inline]
    pub fn output_sample_count_for_input(&self, input_samples: usize) -> usize {
        (input_samples * self.derived.output_sample_rate).div_ceil(self.derived.input_sample_rate)
    }

    /// Output samples needed for a complete input length.
    ///
    /// This can be used to size the output buffer for the entire input stream.
    #[inline]
    pub fn output_sample_count(&self, input_samples: usize) -> usize {
        self.output_sample_count_for_input(input_samples)
    }

    /// Resamples a complete single-channel input buffer and returns all output samples.
    ///
    /// This is a convenience wrapper around the chunked core API.
    pub fn process_all<'a>(&mut self, input: &[T]) -> Result<Vec<T>, Error> {
        let expected_samples = self.output_sample_count(input.len());
        let mut output = Vec::with_capacity(expected_samples);

        let mut offset = 0;
        let input_chunk_size = self.input_buffer_size();

        while offset + input_chunk_size <= input.len() {
            let chunk_output = self.process_chunk(&input[offset..offset + input_chunk_size], false)?;
            output.extend_from_slice(chunk_output);
            offset += input_chunk_size;
        }

        let final_chunk_output = self.process_chunk(&input[offset..], true)?;
        output.extend_from_slice(&final_chunk_output);

        let finalize_output = self.finalize()?;
        output.extend_from_slice(finalize_output);

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
        self.finalized = false;
        self.trim_remaining = self.derived.output_offset + self.decimator_output_delay;
        self.flush_remaining = self.derived.output_offset;
        self.input_sample_count = 0;
        self.output_sample_count = 0;
        self.pre = None;
        self.post = None;
        self.decimation.reset();
        self.decimation_scratch.clear();
    }

    /// Emits delayed tail samples, then marks stream as finalized.
    ///
    /// This flushes any remaining overlap/delay samples that were held back by the chunked
    /// processing pipeline. It is the terminal step of a stream and should be called once per
    /// stream. If the final chunk was not marked by `process_chunk_inner(..., is_final=true)`,
    /// this treats the last accepted full chunk as terminal input.
    ///
    /// Returns a reference to the finalize output samples.
    pub fn finalize<'a>(&'a mut self) -> Result<&'a [T], Error> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        self.final_input_seen = true;

        let written = if self.is_passthrough() || self.input_sample_count == 0 {
            self.finalized = true;
            0
        } else {
            let flush_candidate = self.flush_remaining;
            let written_samples = self.cap_write_to_output_budget(flush_candidate);
            self.finalized = true;

            self.add_synthetic_finalize_tail_to_overlap()?;

            let scale = T::from(self.output_chunk_len_samples()).unwrap_or(T::one())
                / T::from(self.input_chunk_len_samples()).unwrap_or(T::one());
            for (dst, src) in self.output_block[..written_samples]
                .iter_mut()
                .zip(self.overlap[..written_samples].iter())
            {
                *dst = *src * scale;
            }
            written_samples
        };

        self.output_sample_count += written;
        Ok(&self.output_block[..written])
    }

    #[must_use]
    #[inline]
    /// Check if this resampler is finalized.
    ///
    /// Once finalized, no more samples can be written. Samples can still be read until fully drained.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Returns expected total output samples once final stream extent is known.
    ///
    /// Before final input is seen, stream extent is unknown and this returns `None`.
    #[inline]
    fn expected_total_output_samples(&self) -> Option<usize> {
        if !self.final_input_seen {
            return None;
        }
        Some(self.output_sample_count_for_input(self.input_sample_count))
    }

    /// Returns remaining output budget once final stream extent is known.
    #[inline]
    fn remaining_output_budget_samples(&self) -> Option<usize> {
        self.expected_total_output_samples()
            .map(|expected_total| expected_total.saturating_sub(self.output_sample_count))
    }

    /// Caps a candidate write size to the remaining output budget when known.
    #[inline]
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
    /// - If final input was already seen, returns [`Error::StreamFinished`].
    /// - In passthrough mode (equal rates), it copies input directly to output.
    /// - In FFT mode, it processes one chunk, then applies startup trim and writes contiguous
    ///   samples from `output_block`.
    ///
    /// # Parameters
    ///
    /// - `input`: Input samples for this chunk.
    /// - `is_final`: Marks this chunk as the final chunk in the stream.
    ///
    /// Returns a reference to the output samples.
    pub(crate) fn process_chunk<'a>(&'a mut self, input: &'a [T], is_final: bool) -> Result<&'a [T], Error> {
        if self.finalized {
            self.reset();
        }

        if self.final_input_seen {
            return Err(Error::AlreadyFinalized);
        }

        // Shortcut for passthrough mode. Decimation never engages in passthrough mode (it only
        // ever activates for a real downsampling ratio), so `input` is untouched raw data here.
        if self.is_passthrough() {
            let input_samples = input.len();
            if is_final {
                self.final_input_seen = true;
            }
            self.input_sample_count += input_samples;
            let written_samples = self.cap_write_to_output_budget(input_samples);
            self.output_sample_count += written_samples;
            return Ok(&input[..written_samples]);
        }

        let (start, len) = if self.decimation.num_stages() == 0 {
            self.process_fft_chunk(input, input.len(), is_final)?
        } else {
            let raw_len = input.len();
            let mut decimated = std::mem::take(&mut self.decimation_scratch);
            self.decimation.process(input, &mut decimated);

            if is_final {
                // Flush the cascade's remaining delay-line state: real trailing samples are
                // still "in flight" inside the FIR filters at this point and would otherwise be
                // silently dropped rather than reaching the FFT stage at all.
                let mut flushed = Vec::new();
                self.decimation.flush(&mut flushed);
                decimated.extend_from_slice(&flushed);

                // The FFT stage's short-final-chunk path assumes strictly less than one full
                // (decimated-domain) chunk of input; keep that invariant here rather than
                // growing window/scratch buffers to accommodate a rare worst case. Each stage's
                // tap count is already capped (see `design_decimation_taps`) so the flush is
                // effectively lossless in the common case -- this only truncates in extreme
                // configurations where the cascade's delay exceeds a full chunk.
                let chunk_frames = self.derived.input_chunk_frames;
                if decimated.len() >= chunk_frames {
                    decimated.truncate(chunk_frames.saturating_sub(1));
                }
            }

            let outcome = self.process_fft_chunk(&decimated, raw_len, is_final);
            self.decimation_scratch = decimated;
            outcome?
        };

        Ok(&self.output_block[start..start + len])
    }

    /// Runs one chunk through the FFT-domain pipeline for an already-decimated (or, when
    /// decimation is disabled, raw) `input` buffer.
    ///
    /// `raw_input_len` is the number of *raw* (pre-decimation) samples this chunk represents,
    /// used for `input_sample_count`/output-budget bookkeeping, which is always expressed in the
    /// raw domain that callers of [`process_chunk()`](Self::process_chunk) work in. `input.len()`
    /// itself (the decimated-domain count) drives the FFT windowing logic below.
    ///
    /// Returns `(start, len)` indexing into `self.output_block`.
    fn process_fft_chunk(
        &mut self,
        input: &[T],
        raw_input_len: usize,
        is_final: bool,
    ) -> Result<(usize, usize), Error> {
        let input_samples = input.len();

        if is_final {
            self.final_input_seen = true;
            if input_samples == 0 {
                self.input_sample_count += raw_input_len;
                return Ok((0, 0));
            }
        }

        self.copy_input_to_window(input, input_samples);

        let is_first_input = self.input_sample_count == 0;
        if is_first_input {
            self.synthesize_start_context(&input[..input_samples])?;
            self.copy_input_to_window(input, input_samples);
        }

        let is_short_final = is_final && input_samples < self.input_chunk_len_samples();
        if is_short_final {
            self.synthesize_final_block_missing_samples(&input[..input_samples]);
        }

        self.transform_chunk(TransformMode::Normal)?;

        if !is_short_final {
            self.save_current_window();
        }

        self.input_sample_count += raw_input_len;

        let skip_samples = self.trim_remaining.min(self.output_chunk_len_samples());
        self.trim_remaining -= skip_samples;
        let chunk_samples_after_trim = self.output_chunk_len_samples() - skip_samples;
        let written_samples = self.cap_write_to_output_budget(chunk_samples_after_trim);
        let src_start = skip_samples;
        self.output_sample_count += written_samples;

        Ok((src_start, written_samples))
    }

    /// Returns true when rates match and no FFT-domain processing has been requested.
    #[inline]
    fn is_passthrough(&self) -> bool {
        self.derived.input_sample_rate == self.derived.output_sample_rate && !self.derived.spectral.phase_enabled
    }

    /// Normalizes empty edge context vectors to `None`.
    #[inline]
    fn normalize_context(&self, context: Vec<T>) -> Option<Vec<T>> {
        if context.is_empty() { None } else { Some(context) }
    }

    /// Loads input samples into the FFT window at the configured offset.
    #[inline]
    fn copy_input_to_window(&mut self, input: &[T], input_samples: usize) {
        window::write_normal_window(
            &mut self.scratch.rdft_in,
            self.derived.input_offset,
            &input[..input_samples],
        );
    }

    /// Synthesizes start-edge context by backward extrapolation for the first non-empty chunk.
    ///
    /// Returns `Ok(())` after start context is prepared (or when no work is needed), or an error
    /// if the FFT pipeline fails while staging overlap state.
    fn synthesize_start_context(&mut self, input: &[T]) -> Result<(), Error> {
        if input.is_empty() {
            return Ok(());
        }

        window::write_start_window(
            &mut self.scratch.rdft_in,
            self.derived.input_chunk_frames,
            self.derived.input_offset,
            self.pre.as_deref(),
            input,
            self.derived.extrapolation,
        );
        self.transform_chunk(TransformMode::Start)?;
        Ok(())
    }

    /// Builds stop-edge window from prior history for a final short chunk.
    fn synthesize_final_block_missing_samples(&mut self, input: &[T]) {
        window::write_short_final_window(
            &mut self.scratch.rdft_in,
            &mut self.prev_input_window,
            input,
            self.derived.input_chunk_frames,
            self.derived.input_offset,
            self.post.as_deref(),
            self.derived.extrapolation,
        );
    }

    /// Runs one chunk through the FFT-domain resampling pipeline for the current window.
    ///
    /// This method assumes `self.scratch.rdft_in` has already been prepared (windowing,
    /// zero-padding, and stop-edge preparation if needed). It then:
    ///
    /// - Performs a forward real FFT.
    /// - Maps frequency bins into `resampled_spectrum` via the precomputed
    ///   [`SpectralPlan`](crate::spectral::SpectralPlan) (gain, phase, folding/imaging, and
    ///   real-valued DC/Nyquist bins required by `realfft`).
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

        self.derived
            .spectral
            .apply(&self.scratch.spectrum, &mut self.scratch.resampled_spectrum);

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
        window::save_current_window(
            &mut self.prev_input_window,
            &self.scratch.rdft_in,
            self.derived.input_offset,
            self.derived.input_chunk_frames,
        );
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

        window::write_finalize_tail_window(
            &mut self.scratch.rdft_in,
            &mut self.prev_input_window,
            self.derived.input_chunk_frames,
            self.derived.input_offset,
            self.post.as_deref(),
            self.derived.extrapolation,
        );

        self.transform_chunk(TransformMode::End)?;
        Ok(())
    }
}

/// Plans the forward real FFT for one [`CpuCore`] instance.
///
/// When the `f128` feature is compiled in, `use_f128` selects the `f128`-precision engine for
/// `f64`; otherwise the stock `realfft` planner is used. `T` is a compile-time generic parameter,
/// so "is `T` `f64`" is checked with `TypeId`. The downcast is safe because it only runs after
/// confirming that `T` and `f64` are the same type.
///
/// See [`plan_inverse`] for the inverse counterpart.
fn plan_forward<T: Float + FftNum>(
    planner: &mut RealFftPlanner<T>,
    len: usize,
    use_f128: bool,
) -> Arc<dyn RealToComplex<T>> {
    #[cfg(feature = "f128")]
    {
        use std::any::{Any, TypeId};

        if use_f128 && TypeId::of::<T>() == TypeId::of::<f64>() {
            let f128: Arc<dyn RealToComplex<f64>> = crate::f128_fft::plan_fft_forward(len);
            let f128: Box<dyn Any> = Box::new(f128);
            return *f128
                .downcast::<Arc<dyn RealToComplex<T>>>()
                .expect("TypeId check above guarantees T == f64");
        }
    }

    #[cfg(not(feature = "f128"))]
    let _ = use_f128;

    planner.plan_fft_forward(len)
}

/// Inverse counterpart of [`plan_forward`]. Plans the inverse real FFT for one [`CpuCore`] instance.
///
/// When the `f128` feature is compiled in, `use_f128` selects the `f128`-precision engine for
/// `f64`; otherwise the stock `realfft` planner is used. `T` is a compile-time generic parameter,
/// so "is `T` `f64`" is checked with `TypeId`. The downcast is safe because it only runs after
/// confirming that `T` and `f64` are the same type.
fn plan_inverse<T: Float + FftNum>(
    planner: &mut RealFftPlanner<T>,
    len: usize,
    use_f128: bool,
) -> Arc<dyn ComplexToReal<T>> {
    #[cfg(feature = "f128")]
    {
        use std::any::{Any, TypeId};

        if use_f128 && TypeId::of::<T>() == TypeId::of::<f64>() {
            let f128: Arc<dyn ComplexToReal<f64>> = crate::f128_fft::plan_fft_inverse(len);
            let f128: Box<dyn Any> = Box::new(f128);
            return *f128
                .downcast::<Arc<dyn ComplexToReal<T>>>()
                .expect("TypeId check above guarantees T == f64");
        }
    }

    #[cfg(not(feature = "f128"))]
    let _ = use_f128;

    planner.plan_fft_inverse(len)
}

#[cfg(test)]
mod dd_backend_wiring_tests {
    use super::*;
    use crate::Config;

    #[test]
    fn f64_core_resamples_a_sine_correctly() {
        let config = Config::new(44_100, 48_000, 1);
        assert_f64_core_resamples_a_sine_correctly(config);
    }

    #[cfg(feature = "f128")]
    #[test]
    fn f128_core_resamples_a_sine_correctly() {
        let config = Config::new(44_100, 48_000, 1).with_f128(true);
        assert_f64_core_resamples_a_sine_correctly(config);
    }

    fn assert_f64_core_resamples_a_sine_correctly(config: Config) {
        let derived = config.derive_config::<f64>().unwrap();
        let mut core = CpuCore::<f64>::new(derived);

        let input_hz = 1_000.0;
        let input_rate = 44_100.0;
        let input: Vec<f64> = (0..8192)
            .map(|i| (2.0 * std::f64::consts::PI * input_hz * i as f64 / input_rate).sin())
            .collect();

        let output = core.process_all(&input).unwrap();

        assert!(!output.is_empty());
        assert!(
            output.iter().all(|sample| sample.is_finite()),
            "output contains non-finite samples"
        );

        // A steady-state 1kHz sine resampled 44.1k -> 48k should still look like a bounded
        // sine, not silence or a blown-up/garbage signal: check the back half (past startup
        // transients) stays within a sane amplitude envelope around the original's.
        let steady_state = &output[output.len() / 2..];
        let peak = steady_state
            .iter()
            .cloned()
            .fold(0.0f64, |acc, sample| acc.max(sample.abs()));
        assert!(
            peak > 0.5 && peak < 1.5,
            "unexpected steady-state peak amplitude: {peak}"
        );
    }
}
