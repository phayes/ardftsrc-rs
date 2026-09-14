use std::collections::VecDeque;
use std::sync::Arc;

use num_traits::Float;

use crate::Config;
use crate::config::DerivedConfig;
use crate::window;
use crate::{AsChannels, AsChannelsMut};

use super::batch_overlap_shader::BatchOverlapShader;
use super::buffer::GpuScalar;
use super::context::{GpuContext, GpuSubmission};
use super::error::GpuError;
use super::fft_program::{FromF64, pack_complex_batch};
use super::overlap_shader::OverlapMode;
use super::remap_shader::RemapGeometry;
use super::transform_pipeline::GpuTransformPipeline;

/// Default number of FFT chunks batched into one GPU submission ("group"), used only by tests
/// and examples -- [`GpuBatchCore::new`] always requires the caller to pass `group_chunks`
/// explicitly (see the type's own doc for why).
pub const DEFAULT_GROUP_CHUNKS: usize = 4;

/// [`GpuBatchCore::new`] floors any requested `ring_slots` to this minimum -- below it, the ring
/// is not just "smaller," it is unable to function at all: steady-state, three ring slots are
/// always occupied independent of any read-ahead (the seed-copy source, the currently-executing
/// group, and one queued-and-uploaded group), so a fourth is the minimum that lets a *new* group
/// actually be filled/uploaded while the GPU is still busy with a previous one. This is a
/// correctness floor, not a suggested default -- see the type's own doc for why picking the
/// actual number is the caller's call, not this crate's.
const MIN_RING_SLOTS: usize = 4;

/// GPU-resident, memory-bounded, pipelined GPU resampler core:
///
/// Unlike a one-shot "build every window for the whole file, submit once" design,
/// [`GpuBatchCore::push`] takes one fixed-size FFT chunk at a time (exactly
/// [`GpuBatchCore::input_chunk_frames`] samples per channel -- the same contract
/// `CpuCore::process_chunk`/`GpuStreamingCore::process_chunk` already use; a caller that wants to
/// feed arbitrary-sized reads assembles them into fixed-size chunks itself, the same way
/// `PlanarResampler` already does on top of `CpuCore`) and submits GPU work in bounded *groups*
/// of [`GpuBatchCore::group_chunks`] such chunks at a time -- each group is one GPU submission
/// (batched forward FFT / spectral remap / inverse FFT / overlap-add across every chunk in the
/// group, in one command buffer). This bounds GPU memory use to a small, fixed ring of
/// pre-allocated group buffer-sets (`ring_slots`, set once at construction and never grown or
/// shrunk), while still processing files far larger than that by submitting many groups in
/// sequence over the life of one core.
///
/// # Why groups can be submitted asynchronously without a GPU-side dependency chain
///
/// Every chunk's own FFT window depends only on that chunk's own raw samples (zero-padded, no
/// cross-chunk bleed -- see `crate::window::write_normal_window`); cross-chunk continuity comes
/// entirely from the overlap-add step (`output[k] = first_half(k) + second_half(k-1)`). That
/// means the *only* state that needs to cross a group boundary is `second_half` of the group's
/// last chunk -- a single `channels * output_chunk_frames`-sized buffer -- not the whole
/// transform. This core carries that value forward as a `vkCmdCopyBuffer` recorded as the first
/// command of the *next* group's own submission (`BatchOverlapShader::record_seed_copy`), so it
/// never leaves GPU memory and needs no separate synchronization primitive (no timeline
/// semaphores): the host simply waits for group `N`'s fence before recording group `N+1`'s copy
/// from its buffer, exactly as it already needed to wait to know group `N`'s *output* was ready.
///
/// # Pipelining and backpressure
///
/// While one group executes on the GPU, [`GpuBatchCore::push`] can keep accumulating and
/// uploading *later* groups into other ring slots without waiting -- this is what lets a
/// disk-bound caller's read loop overlap with GPU compute. `push` only blocks (backpressure)
/// when every ring slot is occupied (filling, queued, executing, or held as the current
/// seed-copy source) and a new one is needed; [`GpuBatchCore::pending_ready`] offers a
/// non-blocking check for callers that want to avoid blocking. There is no internal thread: all
/// of this happens on the caller's own thread, matching `gpu_plan.md` section 11's stated scope
/// ("actual wrapper threading is out of scope, but the core API must make nonblocking
/// integration possible").
///
/// # Sizing is the caller's responsibility
///
/// [`GpuBatchCore::new`] takes `group_chunks` and `ring_slots` directly, as plain counts -- there
/// is no GPU-memory budget, no device-memory query, and no attempt by this crate to guess a
/// "reasonable" size from bytes. This crate has no way to know what "reasonable" means for a
/// given caller's use case -- a disk-bound batch job tolerant of extra latency wants a deep ring
/// to smooth out I/O jitter, while a latency-sensitive caller wants the shallowest ring that
/// still works. Picking that tradeoff belongs to the caller, not this constructor. The only floor
/// this type enforces is `MIN_RING_SLOTS` itself, a correctness minimum below which the ring
/// simply cannot provide any read-ahead at all (see its own doc). The ring is *fixed* after
/// construction: it is never grown or shrunk at runtime.
pub struct GpuBatchCore<T> {
    context: Arc<GpuContext>,
    derived: DerivedConfig<T>,
    channels: usize,
    pre: Vec<Option<Vec<T>>>,
    post: Vec<Option<Vec<T>>>,

    group_chunks: usize,
    ring_slots: usize,

    /// Idle, reusable group buffer-sets (all built for `[Normal; group_chunks]` windows), ready
    /// to be filled with the next group's chunks.
    free_groups: Vec<Group<T>>,
    /// The group currently being assembled (host-side only; not yet uploaded/submitted).
    filling: Option<Filling<T>>,
    /// Groups fully filled and uploaded, waiting their turn to be submitted (strictly FIFO: a
    /// group's overlap seed comes from whichever group finished immediately before it).
    queued: VecDeque<Filled<T>>,
    /// The one group currently submitted to the GPU (at most one, ever -- see the module doc).
    executing: Option<(GpuSubmission, Filled<T>)>,
    /// The most recently completed group, kept alive only because `executing`'s own submission
    /// copies its ending overlap state as a seed; freed the moment `executing` itself completes.
    prev_completed: Option<Filled<T>>,

    /// Per-channel window history for real start/short-final/finalize-tail synthesis (mirrors
    /// `CpuCore::prev_input_window`); irrelevant to the steady-state group machinery above.
    prev_input_window: Vec<Vec<T>>,
    /// Reused scratch window buffer (`input_fft_size` samples), overwritten once per channel per
    /// chunk. Never reallocated after construction (mirrors `GpuStreamingCore`'s field of the
    /// same name).
    window_scratch: Vec<T>,
    /// Reused host download buffer for a completed group's per-chunk output, sized for the
    /// steady-state max (`channels * group_chunks * output_chunk_frames`); a group with fewer
    /// real chunks (the one-off start/end groups, or a truncated final group) downloads into a
    /// leading sub-slice of this, matching its own smaller GPU buffer length exactly. Never
    /// reallocated after construction.
    chunk_output_scratch: Vec<T>,
    /// Reused host download buffer for a completed group's ending overlap state (always exactly
    /// `channels * output_chunk_frames`, regardless of group size). Never reallocated after
    /// construction.
    overlap_output_scratch: Vec<T>,
    /// Drainable via [`GpuBatchCore::pull_output`].
    ready_output: Vec<Vec<T>>,

    started: bool,
    final_input_seen: bool,
    finalized: bool,
    trim_remaining: usize,
    input_sample_count: usize,
    output_sample_count: usize,
}

/// One pre-allocated GPU buffer-set for a group of `group_chunks` FFT chunks -- reused across
/// the whole stream once built (steady-state groups are structurally identical; only the
/// one-off true start/end groups, built separately, differ in shape and are never recycled).
struct Group<T> {
    pipeline: GpuTransformPipeline<T>,
    overlap: BatchOverlapShader<T>,
    upload_staging: Vec<T>,
}

/// A [`Group`] currently being assembled: chunks are packed into `upload_staging` as they
/// arrive, at increasing window indices, until `window_modes.len()` is reached.
struct Filling<T> {
    group: Group<T>,
    window_modes: Vec<OverlapMode>,
    windows_filled: usize,
    /// True once the true-stream-final window (short-final chunk or an explicit `End` window)
    /// has been appended; a group closed out this way is submitted immediately rather than
    /// waiting to fill up to `group_chunks`.
    is_final: bool,
    recyclable: bool,
}

/// A fully filled, uploaded [`Group`], queued or executing.
struct Filled<T> {
    group: Group<T>,
    num_real_chunks: usize,
    is_final: bool,
    recyclable: bool,
}

fn window_input_stride<T: GpuScalar + FromF64>(group: &Group<T>) -> usize {
    group.pipeline.input_stride()
}

impl<T: Float + GpuScalar + FromF64> GpuBatchCore<T> {
    /// Builds a batch core for `channels` channels of `config`, with `group_chunks` FFT chunks
    /// batched into one GPU submission and a fixed ring of `ring_slots` pre-allocated group
    /// buffer-sets (floored to `MIN_RING_SLOTS`, see the type's own doc for why picking this is
    /// the caller's call, not something this constructor derives).
    ///
    /// Returns [`GpuError::DecimationUnsupported`] if `config` requests pre-decimation,
    /// [`GpuError::DdFftUnsupportedOnGpu`] if it requests `dd_fft`, or
    /// [`GpuError::Fp64Unsupported`] if `T = f64` and `context`'s device cannot run
    /// `shaderFloat64`.
    pub fn new(context: Arc<GpuContext>, config: Config, channels: usize, group_chunks: usize, ring_slots: usize) -> Result<Self, GpuError> {
        if T::scalar_type() == vkfft_rs::ScalarType::F64 {
            context.require_f64()?;
        }
        let group_chunks = group_chunks.max(1);
        let ring_slots = ring_slots.max(MIN_RING_SLOTS);

        let derived = config
            .derive_config::<T>()
            .map_err(|err| GpuError::InvalidConfig(err.to_string()))?;
        if derived.decimation_stages > 0 {
            return Err(GpuError::DecimationUnsupported);
        }
        if derived.dd_fft {
            return Err(GpuError::DdFftUnsupportedOnGpu);
        }

        let steady_state_modes = normal_window_modes(group_chunks);
        let mut free_groups = Vec::with_capacity(ring_slots);
        for _ in 0..ring_slots {
            free_groups.push(build_group(&context, &derived, channels, &steady_state_modes)?);
        }

        Ok(Self {
            context,
            channels,
            pre: vec![None; channels],
            post: vec![None; channels],
            group_chunks,
            ring_slots,
            free_groups,
            filling: None,
            queued: VecDeque::new(),
            executing: None,
            prev_completed: None,
            prev_input_window: vec![vec![T::zero(); derived.input_chunk_frames * 2]; channels],
            window_scratch: vec![T::zero(); derived.input_fft_size],
            chunk_output_scratch: vec![T::zero(); channels * group_chunks * derived.output_chunk_frames],
            overlap_output_scratch: vec![T::zero(); channels * derived.output_chunk_frames],
            ready_output: vec![Vec::new(); channels],
            started: false,
            final_input_seen: false,
            finalized: false,
            trim_remaining: derived.output_offset,
            input_sample_count: 0,
            output_sample_count: 0,
            derived,
        })
    }

    /// Number of channels this core was built for.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of FFT chunks batched into one GPU submission.
    pub fn group_chunks(&self) -> usize {
        self.group_chunks
    }

    /// Number of pre-allocated group buffer-sets in the fixed ring.
    pub fn ring_slots(&self) -> usize {
        self.ring_slots
    }

    /// Sets per-channel previous-track tail context (see `CpuCore::pre`). Must be called before
    /// the first [`GpuBatchCore::push`].
    pub fn pre(&mut self, pre: Vec<Vec<T>>) {
        for (slot, context) in self.pre.iter_mut().zip(pre) {
            *slot = (!context.is_empty()).then_some(context);
        }
    }

    /// Sets per-channel next-track head context (see `CpuCore::post`). Must be called before
    /// [`GpuBatchCore::finalize`].
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

    /// Non-blocking check for whether the currently-executing GPU submission (if any) has
    /// finished. Callers that want to avoid ever blocking in [`GpuBatchCore::push`] can poll
    /// this and only push more once it returns `true` (or `executing` is already empty).
    pub fn pending_ready(&self) -> Result<bool, GpuError> {
        match &self.executing {
            Some((submission, _)) => self.context.is_submission_ready(submission),
            None => Ok(true),
        }
    }

    /// Pushes one chunk of per-channel input samples through the pipeline.
    ///
    /// `input` accepts any [`AsChannels`] shape -- a `&[&[T]]`, a `&Vec<Vec<T>>`/`&[Vec<T>]`, or
    /// a `&PlanarVecs<T>` -- so a caller already holding one of those doesn't need to collect a
    /// temporary `Vec<&[T]>` just to call this.
    ///
    /// `input` must be a *fixed*-size chunk (exactly [`GpuBatchCore::input_chunk_frames`]
    /// samples per channel), matching `CpuCore::process_chunk`/`GpuStreamingCore::process_chunk`'s
    /// own contract -- the only exception is the final call (`is_final = true`), which may be
    /// shorter (or empty). A caller that wants to feed arbitrary-sized reads assembles them into
    /// fixed-size chunks itself (the same split `PlanarResampler`/`InterleavedResampler` already
    /// do on top of the fixed-size `CpuCore::process_chunk`), rather than this core buffering
    /// arbitrary sizes internally.
    ///
    /// Blocks only when the fixed ring has no free buffer-set left and one is needed -- i.e.
    /// when GPU work is fully backlogged relative to `ring_slots`.
    pub fn push<'a>(&mut self, input: impl AsChannels<'a, T>, is_final: bool) -> Result<(), GpuError> {
        if self.finalized || self.final_input_seen {
            return Err(GpuError::InvalidSubmissionState("stream has already been finalized".to_string()));
        }
        let input_samples = if input.channel_count() == 0 { 0 } else { input.channel(0).len() };
        let chunk_frames = self.derived.input_chunk_frames;
        if is_final {
            self.final_input_seen = true;
            if input_samples > chunk_frames {
                return Err(GpuError::InvalidConfig(format!(
                    "final chunk must be at most {chunk_frames} samples per channel, got {input_samples}"
                )));
            }
        } else if input_samples != chunk_frames {
            return Err(GpuError::InvalidConfig(format!("expected exactly {chunk_frames} samples per channel, got {input_samples}")));
        }
        if input_samples == 0 {
            return Ok(());
        }
        self.input_sample_count += input_samples;
        self.add_chunk(input, is_final)
    }

    /// Required per-channel input length for a non-final [`GpuBatchCore::push`] call.
    pub fn input_chunk_frames(&self) -> usize {
        self.derived.input_chunk_frames
    }

    /// Copies as many output samples as fit into `output` (one channel of equal length per
    /// channel) from what's been produced so far by GPU submissions that have completed since
    /// the last call, and returns how many samples per channel were written. Never blocks.
    ///
    /// `output` accepts any [`AsChannelsMut`] shape -- a `&mut [&mut [T]]`, a `&mut
    /// Vec<Vec<T>>`/`&mut [Vec<T>]`, or a `&mut PlanarVecs<T>` -- so a caller already holding one
    /// of those doesn't need to collect a temporary `Vec<&mut [T]>` just to call this.
    ///
    /// If more samples are ready than fit in `output`, the remainder stays buffered internally
    /// for the next call -- call this in a loop (checking the returned count) to fully drain
    /// what's ready, the same way you would size a `read()`-style call against a fixed buffer. A
    /// return value less than `output`'s length (`0` included) means the internal buffer was
    /// fully drained on this call, not just that this one call happened to run out of room.
    ///
    /// After [`GpuBatchCore::finalize`] has returned, every sample the stream will ever produce
    /// is already sitting in that internal buffer (`finalize` itself blocks until all
    /// outstanding GPU work is done and processed) -- so looping `pull_output` until it returns
    /// `0` at that point is a reliable, terminating "have I drained everything" check, with
    /// nothing else to wait for or poll.
    ///
    /// Copying out of a persistent internal buffer and shrinking it in place (rather than
    /// returning a fresh, owned collection every call) is what keeps this allocation-free.
    pub fn pull_output<'a>(&mut self, output: impl AsChannelsMut<'a, T>) -> Result<usize, GpuError> {
        self.advance(false)?;
        self.drain_ready_output(output)
    }

    /// Copies up to `min(output channel lengths, buffered sample counts)` samples per channel out
    /// of `self.ready_output` into `output`, then removes exactly that many samples from the
    /// front of each `self.ready_output[c]` (`Vec::drain` shifts the remainder down in place,
    /// keeping the buffer's capacity -- no allocation).
    fn drain_ready_output<'a>(&mut self, mut output: impl AsChannelsMut<'a, T>) -> Result<usize, GpuError> {
        if output.channel_count() != self.channels {
            return Err(GpuError::InvalidConfig(format!(
                "expected {} channel output slices, got {}",
                self.channels,
                output.channel_count()
            )));
        }
        let available = self.ready_output.iter().map(Vec::len).min().unwrap_or(0);
        let requested = (0..self.channels).map(|c| output.channel_mut(c).len()).min().unwrap_or(0);
        let written = available.min(requested);
        for c in 0..self.channels {
            let dst = output.channel_mut(c);
            dst[..written].copy_from_slice(&self.ready_output[c][..written]);
            self.ready_output[c].drain(..written);
        }
        Ok(written)
    }

    /// Terminal call: takes no new input (the final chunk, if any, must already have been given
    /// to [`GpuBatchCore::push`] with `is_final = true`). If the last chunk `push` saw was
    /// exactly full-length -- so its own forward-tail wasn't already folded into a short-final
    /// window -- appends the finalize-tail window here, exactly mirroring
    /// `CpuCore::add_synthetic_finalize_tail_to_overlap`'s own skip condition. Blocks until every
    /// outstanding GPU submission completes, then marks the stream finalized. Does not return
    /// output itself -- call [`GpuBatchCore::pull_output`] in a loop afterward, same as
    /// mid-stream, until it returns `0`; see that method's doc for why that loop is guaranteed
    /// to terminate once this has returned.
    pub fn finalize(&mut self) -> Result<(), GpuError> {
        if self.finalized {
            return Err(GpuError::InvalidSubmissionState("stream has already been finalized".to_string()));
        }
        self.final_input_seen = true;

        if self.input_sample_count > 0 && self.input_sample_count.is_multiple_of(self.derived.input_chunk_frames) {
            self.close_out_full_finish()?;
        }

        if let Some(filling) = self.filling.take() {
            self.enqueue_filled(filling)?;
        }
        while !self.queued.is_empty() || self.executing.is_some() {
            self.advance(true)?;
        }
        if let Some(prev) = self.prev_completed.take()
            && prev.recyclable
        {
            self.free_groups.push(prev.group);
        }

        self.finalized = true;
        Ok(())
    }

    /// Appends one real chunk (`chunk_frames` samples, or shorter only when `is_final_chunk`) to
    /// whichever group is currently being filled, building a new one (including the true-stream
    /// [`OverlapMode::Start`] window, if this is the very first chunk ever) if needed, and
    /// submitting it once full or once closed out by `is_final_chunk`.
    fn add_chunk<'a>(&mut self, chunk: impl AsChannels<'a, T>, is_final_chunk: bool) -> Result<(), GpuError> {
        let chunk_frames = self.derived.input_chunk_frames;
        let is_short_final = is_final_chunk && chunk.channel(0).len() < chunk_frames;

        if self.filling.is_none() {
            if !self.started {
                self.started = true;
                let mut start_group = build_group(&self.context, &self.derived, self.channels, &[OverlapMode::Start])?;
                for c in 0..self.channels {
                    window::write_start_window(
                        &mut self.window_scratch,
                        chunk_frames,
                        self.derived.input_offset,
                        self.pre[c].as_deref(),
                        chunk.channel(c),
                        self.derived.extrapolation,
                    );
                    let input_stride = window_input_stride(&start_group);
                    pack_complex_batch(&mut start_group.upload_staging, c, input_stride, &self.window_scratch);
                }
                start_group.pipeline.input_buffer().upload(&start_group.upload_staging)?;
                let context = &self.context;
                let submission = context.submit_async(|command_buffer| {
                    start_group.pipeline.record(context.device(), command_buffer);
                    start_group.overlap.record(context.device(), command_buffer);
                })?;
                // Not waited on here: this becomes the initial `executing` entry, so `advance`
                // only waits for it once the first real group actually needs its overlap state
                // as a seed (`submit_group` reads `prev_completed`), not unconditionally here.
                self.executing = Some((
                    submission,
                    Filled {
                        group: start_group,
                        num_real_chunks: 0,
                        is_final: false,
                        recyclable: false,
                    },
                ));
            }

            while self.free_groups.is_empty() {
                if self.executing.is_none() && self.queued.is_empty() {
                    return Err(GpuError::ExecutionFailed("GPU batch ring exhausted".to_string()));
                }
                self.advance(true)?;
            }
            let free = self.free_groups.pop().expect("checked non-empty above");
            self.filling = Some(Filling {
                group: free,
                window_modes: Vec::with_capacity(self.group_chunks),
                windows_filled: 0,
                is_final: false,
                recyclable: true,
            });
        }

        let filling = self.filling.as_mut().expect("just ensured");
        let window_index = filling.windows_filled;
        let input_stride = window_input_stride(&filling.group);
        if is_short_final {
            for c in 0..self.channels {
                window::write_short_final_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    chunk.channel(c),
                    chunk_frames,
                    self.derived.input_offset,
                    self.post[c].as_deref(),
                    self.derived.extrapolation,
                );
                pack_complex_batch(&mut filling.group.upload_staging, window_index * self.channels + c, input_stride, &self.window_scratch);
            }
        } else {
            for c in 0..self.channels {
                window::write_normal_window(&mut self.window_scratch, self.derived.input_offset, chunk.channel(c));
                pack_complex_batch(&mut filling.group.upload_staging, window_index * self.channels + c, input_stride, &self.window_scratch);
                window::save_current_window(&mut self.prev_input_window[c], &self.window_scratch, self.derived.input_offset, chunk_frames);
            }
        }
        filling.window_modes.push(OverlapMode::Normal);
        filling.windows_filled += 1;

        // `filling.is_final` means "this group's ending overlap state is already the correctly
        // scaled finalize tail, safe to read directly" (mirroring `CpuCore::finalize`, which
        // always reads `self.overlap` as the tail). That's true for a short final chunk -- its
        // own forward tail prediction is already folded into the window itself via
        // `write_short_final_window`, so its own second-half *is* the tail -- but NOT for an
        // exactly full-length final chunk, whose group is closed out as a plain `Normal` window
        // above and still needs a separate finalize-tail `End` window (appended to a fresh
        // one-off group by `finalize()` calling `close_out_full_finish`) before its overlap state
        // is the real tail. Marking this group `is_final` in that case would make
        // `download_and_process` read (and budget-consume) its not-yet-finalized overlap state
        // as if it were the tail, corrupting the real tail the `End` group produces afterward.
        //
        // Either way, a final chunk (short or full) still needs to close out and submit its
        // group immediately rather than waiting for more input that will never come -- that part
        // applies unconditionally.
        if is_short_final {
            filling.is_final = true;
        }

        if filling.windows_filled >= self.group_chunks || is_final_chunk {
            let filling = self.filling.take().expect("just used");
            self.enqueue_filled(filling)?;
        }
        Ok(())
    }

    /// True-stream-end special case: the previous chunk was the last one and it was full-length
    /// (`is_multiple_of(chunk_frames)`), so `finalize()` found no leftover partial chunk to
    /// process -- but a finalize-tail window still needs to be appended and submitted to recover
    /// the last chunk's held-back overlap contribution.
    fn close_out_full_finish(&mut self) -> Result<(), GpuError> {
        let chunk_frames = self.derived.input_chunk_frames;
        if let Some(filling) = self.filling.as_mut() {
            let window_index = filling.windows_filled;
            let input_stride = window_input_stride(&filling.group);
            for c in 0..self.channels {
                window::write_finalize_tail_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    chunk_frames,
                    self.derived.input_offset,
                    self.post[c].as_deref(),
                    self.derived.extrapolation,
                );
                pack_complex_batch(&mut filling.group.upload_staging, window_index * self.channels + c, input_stride, &self.window_scratch);
            }
            filling.window_modes.push(OverlapMode::End);
            filling.windows_filled += 1;
            filling.is_final = true;
        } else {
            // Nothing is currently being filled (the last group was already submitted exactly
            // full): build a one-off, single-window `End` group.
            let mut end_group = build_group(&self.context, &self.derived, self.channels, &[OverlapMode::End])?;
            let input_stride = window_input_stride(&end_group);
            for c in 0..self.channels {
                window::write_finalize_tail_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    chunk_frames,
                    self.derived.input_offset,
                    self.post[c].as_deref(),
                    self.derived.extrapolation,
                );
                pack_complex_batch(&mut end_group.upload_staging, c, input_stride, &self.window_scratch);
            }
            self.filling = Some(Filling {
                group: end_group,
                window_modes: vec![OverlapMode::End],
                windows_filled: 1,
                is_final: true,
                recyclable: false,
            });
        }
        Ok(())
    }

    fn enqueue_filled(&mut self, filling: Filling<T>) -> Result<(), GpuError> {
        let Filling { mut group, window_modes, windows_filled, is_final, recyclable } = filling;
        let num_real_chunks = window_modes.iter().filter(|mode| **mode == OverlapMode::Normal).count();

        // A group built with fewer windows than the ring's steady-state shape (the true-final,
        // possibly-short group) needs its own differently-shaped `BatchOverlapShader`; rebuild
        // it in place rather than trying to reuse the pre-built uniform one. Steady-state groups
        // (exactly `group_chunks` `Normal` windows) already match and skip this.
        if windows_filled != self.group_chunks || window_modes.iter().any(|mode| *mode != OverlapMode::Normal) {
            group.overlap = BatchOverlapShader::<T>::build(
                &self.context,
                T::scalar_type(),
                group.pipeline.output_buffer(),
                group.pipeline.output_stride(),
                self.derived.output_chunk_frames,
                self.derived.input_chunk_frames,
                self.channels,
                &window_modes,
            )?;
        }

        group.pipeline.input_buffer().upload(&group.upload_staging)?;
        self.queued.push_back(Filled { group, num_real_chunks, is_final, recyclable });
        self.advance(false)
    }

    /// One pipeline pump step: if a group is currently executing, either waits for it (`block`)
    /// or polls it once (non-blocking); if it (or nothing) was executing and a queued group is
    /// waiting, submits it. Callers that want to fully drain the pipeline (`finalize`) or wait
    /// for at least one free ring slot (`add_chunk`) loop this externally, checking their own
    /// termination condition between calls -- looping *inside* this method would over-drain the
    /// whole backlog just to free a single slot, defeating the read-ahead this ring exists for.
    fn advance(&mut self, block: bool) -> Result<(), GpuError> {
        if let Some((submission, _)) = &self.executing {
            let ready = if block {
                self.context.wait_submission(submission)?;
                true
            } else {
                self.context.is_submission_ready(submission)?
            };
            if ready {
                let (submission, filled) = self.executing.take().expect("checked above");
                self.context.destroy_submission(submission);
                self.download_and_process(&filled)?;
                if let Some(prev) = self.prev_completed.replace(filled)
                    && prev.recyclable
                {
                    self.free_groups.push(prev.group);
                }
            }
        }

        if self.executing.is_none()
            && let Some(next) = self.queued.pop_front()
        {
            let submission = self.submit_group(&next)?;
            self.executing = Some((submission, next));
        }
        Ok(())
    }

    fn submit_group(&self, next: &Filled<T>) -> Result<GpuSubmission, GpuError> {
        let seed_source = self.prev_completed.as_ref().map(|prev| &prev.group.overlap);
        let context = &self.context;
        context.submit_async(|command_buffer| {
            if let Some(seed) = seed_source {
                next.group.overlap.record_seed_copy(context.device(), command_buffer, seed.overlap_buffer());
            }
            next.group.pipeline.record(context.device(), command_buffer);
            next.group.overlap.record(context.device(), command_buffer);
        })
    }

    /// Downloads `filled`'s per-chunk outputs, applies the same trim/output-budget accounting
    /// `CpuCore`/`GpuStreamingCore` apply per chunk, and appends the result to `ready_output`; if
    /// `filled.is_final`, also downloads and appends the finalize-tail from its ending overlap
    /// state.
    fn download_and_process(&mut self, filled: &Filled<T>) -> Result<(), GpuError> {
        let output_chunk_frames = self.derived.output_chunk_frames;
        let chunk_output_len = self.channels * filled.num_real_chunks * output_chunk_frames;
        let chunk_output = &mut self.chunk_output_scratch[..chunk_output_len];
        if filled.num_real_chunks > 0 {
            filled.group.overlap.output_buffer().download(chunk_output)?;
        }
        let expected_total = self.expected_total_output_samples();

        for k in 0..filled.num_real_chunks {
            let skip = self.trim_remaining.min(output_chunk_frames);
            self.trim_remaining -= skip;
            let after_trim = output_chunk_frames - skip;
            let budget = expected_total.map_or(after_trim, |total| after_trim.min(total.saturating_sub(self.output_sample_count)));
            self.output_sample_count += budget;
            for c in 0..self.channels {
                let base = k * self.channels * output_chunk_frames + c * output_chunk_frames + skip;
                self.ready_output[c].extend_from_slice(&self.chunk_output_scratch[base..base + budget]);
            }
        }

        if filled.is_final {
            filled.group.overlap.overlap_buffer().download(&mut self.overlap_output_scratch)?;
            let scale = T::from(output_chunk_frames).unwrap_or_else(T::one) / T::from(self.derived.input_chunk_frames).unwrap_or_else(T::one);
            let tail_written = expected_total.map_or(output_chunk_frames, |total| total.saturating_sub(self.output_sample_count)).min(output_chunk_frames);
            self.output_sample_count += tail_written;
            for c in 0..self.channels {
                let base = c * output_chunk_frames;
                self.ready_output[c].extend(self.overlap_output_scratch[base..base + tail_written].iter().map(|&value| value * scale));
            }
        }
        Ok(())
    }
}

fn build_group<T: Float + GpuScalar + FromF64>(
    context: &Arc<GpuContext>,
    derived: &DerivedConfig<T>,
    channels: usize,
    window_modes: &[OverlapMode],
) -> Result<Group<T>, GpuError> {
    let windows = window_modes.len();
    let geometry = RemapGeometry {
        direction_up: derived.input_chunk_frames < derived.output_chunk_frames,
        n: derived.spectral.geometry.lower_nyquist_bin,
        r0: derived.spectral.geometry.reflect_start_bin(),
        nyquist_fold: if derived.input_chunk_frames > derived.output_chunk_frames { 2.0 } else { 1.0 },
        gain: &derived.spectral.gain,
        phase: derived.spectral.phase_enabled.then_some(derived.spectral.phase.as_slice()),
    };
    let pipeline = GpuTransformPipeline::<T>::build(
        context,
        T::precision(),
        T::scalar_type(),
        derived.input_fft_size,
        derived.output_fft_size,
        channels * windows,
        &geometry,
    )?;
    let overlap = BatchOverlapShader::<T>::build(
        context,
        T::scalar_type(),
        pipeline.output_buffer(),
        pipeline.output_stride(),
        derived.output_chunk_frames,
        derived.input_chunk_frames,
        channels,
        window_modes,
    )?;
    let input_stride = pipeline.input_stride();
    let upload_staging = vec![T::zero(); channels * windows * input_stride * 2];
    Ok(Group { pipeline, overlap, upload_staging })
}

fn normal_window_modes(windows: usize) -> Vec<OverlapMode> {
    vec![OverlapMode::Normal; windows]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_core::CpuCore;

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

    /// Drains everything currently ready from `core` (looping `pull_output` until it returns
    /// `0`, since more may be ready than fits in one fixed-size scratch buffer) and appends it
    /// onto `dst`, one `Vec<f32>` per channel.
    fn drain_all_output(core: &mut GpuBatchCore<f32>, channels: usize, dst: &mut [Vec<f32>]) {
        let mut scratch = vec![vec![0.0f32; 16_384]; channels];
        loop {
            let written = core.pull_output(&mut scratch).expect("gpu pull_output");
            for (c, samples) in scratch.iter().enumerate() {
                dst[c].extend_from_slice(&samples[..written]);
            }
            if written == 0 {
                break;
            }
        }
    }

    /// Streams `total_frames` per-channel input through both `CpuCore` (one instance per
    /// channel) and the pipelined `GpuBatchCore`, both fed one fixed-size FFT chunk at a time
    /// (the last one possibly short, with `is_final = true`) -- `GpuBatchCore::push` requires
    /// this exact contract, matching `CpuCore::process_chunk`/`GpuStreamingCore::process_chunk`.
    /// Uses real `pre`/`post` context on both ends (see `gpu::streaming_core::tests` for why:
    /// `Extrapolation::Lpc` has a known, separately tracked GPU divergence without real context,
    /// unrelated to what this test checks).
    fn assert_batch_matches_cpu(input_rate: usize, output_rate: usize, channels: usize, total_frames: usize, group_chunks: usize) {
        let context = match GpuContext::new() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping GPU batch test: {err}");
                return;
            }
        };

        let config = Config::new(input_rate, output_rate, channels);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();

        let mut cpu_cores: Vec<CpuCore<f32>> = (0..channels).map(|_| CpuCore::new(derived.clone())).collect();
        let mut gpu_core = GpuBatchCore::<f32>::new(context, config, channels, group_chunks, MIN_RING_SLOTS).expect("build GpuBatchCore");
        assert_eq!(gpu_core.input_chunk_frames(), chunk_frames);

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
                let out = core.process_chunk(&input[c][offset..offset + this_chunk], is_final).expect("cpu process_chunk");
                cpu_output[c].extend_from_slice(out);
            }
            let refs: Vec<&[f32]> = (0..channels).map(|c| &input[c][offset..offset + this_chunk]).collect();
            gpu_core.push(&refs[..], is_final).expect("gpu push");
            drain_all_output(&mut gpu_core, channels, &mut gpu_output);
            offset += this_chunk;
            if is_final {
                break;
            }
        }
        for (c, core) in cpu_cores.iter_mut().enumerate() {
            cpu_output[c].extend_from_slice(core.finalize().expect("cpu finalize"));
        }
        gpu_core.finalize().expect("gpu finalize");
        drain_all_output(&mut gpu_core, channels, &mut gpu_output);

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
                "{input_rate} -> {output_rate} channel {c}: GPU batch/CPU mismatch, max abs error {max_abs_error}"
            );
        }
    }

    #[test]
    fn batch_matches_cpu_mono_44100_to_48000() {
        assert_batch_matches_cpu(44_100, 48_000, 1, 44_100 * 3, DEFAULT_GROUP_CHUNKS);
    }

    #[test]
    fn batch_matches_cpu_stereo_48000_to_44100() {
        assert_batch_matches_cpu(48_000, 44_100, 2, 48_100 * 2, DEFAULT_GROUP_CHUNKS);
    }

    #[test]
    fn batch_matches_cpu_mono_44100_to_96000_short_final() {
        assert_batch_matches_cpu(44_100, 96_000, 1, 44_100 + 777, DEFAULT_GROUP_CHUNKS);
    }

    #[test]
    fn batch_matches_cpu_with_group_size_one() {
        // group_chunks = 1 forces every single FFT chunk through its own GPU submission --
        // exercises the ring/seed-carry machinery maximally.
        assert_batch_matches_cpu(44_100, 48_000, 1, 44_100, 1);
    }

    #[test]
    fn batch_matches_cpu_single_short_chunk() {
        assert_batch_matches_cpu(44_100, 48_000, 1, 500, DEFAULT_GROUP_CHUNKS);
    }

    #[test]
    fn batch_matches_cpu_exact_multiple_of_group_and_chunk() {
        let config = Config::new(44_100, 48_000, 1);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();
        assert_batch_matches_cpu(44_100, 48_000, 1, chunk_frames * DEFAULT_GROUP_CHUNKS * 2, DEFAULT_GROUP_CHUNKS);
    }

    #[test]
    fn pull_output_partial_drain_across_multiple_calls() {
        let context = match GpuContext::new() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let channels = 1;
        let config = Config::new(44_100, 48_000, channels);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();
        let mut cpu_core = CpuCore::<f32>::new(derived);
        let mut gpu_core = GpuBatchCore::<f32>::new(context, config, channels, DEFAULT_GROUP_CHUNKS, MIN_RING_SLOTS).expect("build");

        let total_frames = chunk_frames * DEFAULT_GROUP_CHUNKS + 137;
        // Real `pre`/`post` context on both ends, same as `assert_batch_matches_cpu` -- avoids
        // `Extrapolation::Lpc`'s known, separately tracked GPU divergence without real context
        // (unrelated to what this test checks).
        let full = tone_channels(channels, chunk_frames + total_frames + chunk_frames, 44_100);
        let pre = full[0][..chunk_frames].to_vec();
        let input = vec![full[0][chunk_frames..chunk_frames + total_frames].to_vec()];
        let post = full[0][chunk_frames + total_frames..].to_vec();
        cpu_core.pre(pre.clone());
        cpu_core.post(post.clone());
        gpu_core.pre(vec![pre]);
        gpu_core.post(vec![post]);

        let mut cpu_output = Vec::new();
        let mut gpu_output = vec![Vec::new(); channels];
        let mut offset = 0;
        loop {
            let remaining = total_frames - offset;
            let is_final = remaining <= chunk_frames;
            let this_chunk = if is_final { remaining } else { chunk_frames };
            let out = cpu_core.process_chunk(&input[0][offset..offset + this_chunk], is_final).expect("cpu process_chunk");
            cpu_output.extend_from_slice(out);
            let refs: [&[f32]; 1] = [&input[0][offset..offset + this_chunk]];
            gpu_core.push(&refs[..], is_final).expect("gpu push");

            // Deliberately tiny relative to a group's real output size, to force multiple
            // `pull_output` calls to drain whatever a single completed group produced.
            let mut tiny = [0.0f32; 37];
            loop {
                let mut refs: [&mut [f32]; 1] = [&mut tiny];
                let written = gpu_core.pull_output(&mut refs[..]).expect("gpu pull_output");
                gpu_output[0].extend_from_slice(&tiny[..written]);
                if written < tiny.len() {
                    break;
                }
            }
            offset += this_chunk;
            if is_final {
                break;
            }
        }
        cpu_output.extend_from_slice(cpu_core.finalize().expect("cpu finalize"));
        gpu_core.finalize().expect("gpu finalize");
        drain_all_output(&mut gpu_core, channels, &mut gpu_output);

        assert_eq!(cpu_output.len(), gpu_output[0].len());
        let max_abs_error = cpu_output.iter().zip(gpu_output[0].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs_error < 1e-3, "max abs error {max_abs_error}");
    }

    #[test]
    fn push_rejects_wrong_chunk_size() {
        let context = match GpuContext::new() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let mut core = GpuBatchCore::<f32>::new(context, Config::new(44_100, 48_000, 1), 1, DEFAULT_GROUP_CHUNKS, MIN_RING_SLOTS).expect("build");
        let wrong_size = vec![0.0f32; core.input_chunk_frames() + 1];
        assert!(core.push(&[&wrong_size[..]][..], false).is_err());
    }

    #[test]
    fn ring_slots_and_group_chunks_are_floored_not_derived() {
        let context = match GpuContext::new() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        // Requesting fewer than `MIN_RING_SLOTS`/1 must be floored, not honored or rejected --
        // there is no other "sizing" logic left in this constructor to second-guess the caller.
        let core = GpuBatchCore::<f32>::new(context, Config::new(44_100, 48_000, 2), 2, 0, 1).expect("build");
        assert_eq!(core.ring_slots(), MIN_RING_SLOTS);
        assert_eq!(core.group_chunks(), 1);
    }
}
