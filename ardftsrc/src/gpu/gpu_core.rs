use std::collections::VecDeque;
use std::sync::Arc;

use num_traits::Float;

use crate::window;
use crate::{AsChannels, AsChannelsMut};

use super::batch_overlap_shader::BatchOverlapShader;
use super::buffer::GpuScalar;
use super::context::{GpuContext, GpuSubmission};
use super::error::GpuError;
use super::fft_program::{FromF64, pack_complex_batch};
use super::overlap_shader::OverlapMode;
use super::shaders::{GpuShaders, remap_geometry};
use super::transform_pipeline::GpuTransformPipeline;

/// Minimum number of ring slots needed for the ring to function properly.
const MIN_RING_SLOTS: usize = 4;

/// GPU-accelerated streaming resampler.
///
/// Pass one fixed-size chunk at a time to [`GpuCore::push_input`]. Each channel must
/// contain [`GpuCore::input_chunk_frames`] samples, except for the final chunk.
///
/// Chunks are processed using the chunk-group geometry compiled into [`GpuContext`], and several
/// groups may be buffered while the GPU is busy. Memory use is bounded by the `ring_slots`
/// supplied to [`GpuCore::new`]. `push_input` may block when that buffer is full; use
/// [`GpuCore::push_input_ready`] to check GPU progress without blocking.
pub struct GpuCore<T> {
    context: Arc<GpuContext<T>>,
    pre: Vec<Option<Vec<T>>>,
    post: Vec<Option<Vec<T>>>,

    ring_slots: usize,

    /// Idle, reusable group buffer-sets (all built for `[Normal; group_chunks]` windows), ready
    /// to be filled with the next group's chunks.
    free_groups: Vec<Group<T>>,
    /// The group currently being assembled (host-side only; not yet uploaded/submitted).
    filling: Option<Filling<T>>,
    /// Groups fully filled and uploaded, waiting their turn to be submitted (strictly FIFO: a
    /// group's overlap seed comes from whichever group finished immediately before it).
    queued: VecDeque<Filled<T>>,
    /// The one group currently submitted to the GPU, together with every buffer it may read.
    executing: Option<GpuSubmission<Executing<T>>>,
    /// The most recently completed group, moved into the next submission when used as its
    /// overlap seed.
    prev_completed: Option<Filled<T>>,

    /// Per-channel window history for real start/short-final/finalize-tail synthesis (mirrors
    /// `CpuCore::prev_input_window`); irrelevant to the steady-state group machinery above.
    prev_input_window: Vec<Vec<T>>,
    /// Reused scratch window buffer (`input_fft_size` samples), overwritten once per channel per
    /// chunk. Never reallocated after construction.
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
    /// Drainable via [`GpuCore::pull_output`].
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
    overlap: BatchOverlapShader<T>,
    pipeline: GpuTransformPipeline<T>,
    upload_staging: Vec<T>,
    transform_batch_count: usize,
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

/// Resources owned by one GPU submission. `seed` is included because the command stream may
/// copy from its overlap buffer before processing `current`.
struct Executing<T> {
    current: Filled<T>,
    seed: Option<Filled<T>>,
}

fn window_input_stride<T: GpuScalar + FromF64>(group: &Group<T>) -> usize {
    group.pipeline.input_stride()
}

#[allow(private_bounds)]
impl<T: Float + GpuScalar + FromF64> GpuCore<T> {
    /// Builds a streaming core using the context's fixed shader geometry and a ring of
    /// `ring_slots` runtime buffer sets. The core takes ownership of `context` and shares it
    /// internally among its GPU resources.
    pub fn new(context: GpuContext<T>, ring_slots: usize) -> Result<Self, GpuError> {
        let context = Arc::new(context);
        let ring_slots = ring_slots.max(MIN_RING_SLOTS);
        let channels = context.config().channels;
        let group_chunks = context.group_chunks();
        let derived = context.derived();
        let input_chunk_frames = derived.input_chunk_frames;
        let input_fft_size = derived.input_fft_size;
        let output_chunk_frames = derived.output_chunk_frames;
        let output_offset = derived.output_offset;
        let shaders = context.compiled_shaders()?;
        let steady_state_modes = normal_window_modes(group_chunks);
        let mut free_groups = Vec::with_capacity(ring_slots);
        for _ in 0..ring_slots {
            free_groups.push(build_group(&context, &steady_state_modes, &shaders)?);
        }

        Ok(Self {
            context,
            pre: vec![None; channels],
            post: vec![None; channels],
            ring_slots,
            free_groups,
            filling: None,
            queued: VecDeque::new(),
            executing: None,
            prev_completed: None,
            prev_input_window: vec![vec![T::zero(); input_chunk_frames * 2]; channels],
            window_scratch: vec![T::zero(); input_fft_size],
            chunk_output_scratch: vec![T::zero(); channels * group_chunks * output_chunk_frames],
            overlap_output_scratch: vec![T::zero(); channels * output_chunk_frames],
            ready_output: vec![Vec::new(); channels],
            started: false,
            final_input_seen: false,
            finalized: false,
            trim_remaining: output_offset,
            input_sample_count: 0,
            output_sample_count: 0,
        })
    }

    /// Number of channels this core was built for.
    pub fn channels(&self) -> usize {
        self.context.config().channels
    }

    /// Number of FFT chunks batched into one GPU submission.
    pub fn group_chunks(&self) -> usize {
        self.context.group_chunks()
    }

    /// Number of pre-allocated group buffer-sets in the fixed ring.
    pub fn ring_slots(&self) -> usize {
        self.ring_slots
    }

    /// The context this core was built from -- exposed so callers that need an independent
    /// sibling core (for example, one per track in a batch) can clone its geometry/device/shaders
    /// via [`GpuContext::clone_shared`] without recompiling.
    pub(crate) fn context(&self) -> &GpuContext<T> {
        &self.context
    }

    /// Sets per-channel previous-track tail context (see `CpuCore::pre`). Must be called before
    /// the first [`GpuCore::push_input`].
    pub fn pre(&mut self, pre: Vec<Vec<T>>) {
        for (slot, context) in self.pre.iter_mut().zip(pre) {
            *slot = (!context.is_empty()).then_some(context);
        }
    }

    /// Sets per-channel next-track head context (see `CpuCore::post`). Must be called before
    /// [`GpuCore::finalize`].
    pub fn post(&mut self, post: Vec<Vec<T>>) {
        for (slot, context) in self.post.iter_mut().zip(post) {
            *slot = (!context.is_empty()).then_some(context);
        }
    }

    /// Returns the total expected output samples for `input_samples` per-channel input samples.
    pub fn output_sample_count_for_input(&self, input_samples: usize) -> usize {
        (input_samples * self.context.derived().output_sample_rate).div_ceil(self.context.derived().input_sample_rate)
    }

    #[inline]
    fn expected_total_output_samples(&self) -> Option<usize> {
        self.final_input_seen
            .then(|| self.output_sample_count_for_input(self.input_sample_count))
    }

    /// Non-blocking check for whether the currently-executing GPU submission (if any) has
    /// finished. Callers that want to avoid ever blocking in [`GpuCore::push_input`] can poll
    /// this and only push more once it returns `true` (or `executing` is already empty).
    ///
    /// This check is cheap: it performs no allocation, submission, or data transfer, and at
    /// most queries the status of one Vulkan fence.
    pub fn push_input_ready(&self) -> Result<bool, GpuError> {
        match &self.executing {
            Some(submission) => submission.is_ready(),
            None => Ok(true),
        }
    }

    /// Pushes one chunk of per-channel input samples through the pipeline.
    ///
    /// `input` accepts any [`AsChannels`] shape -- a `&[&[T]]`, a `&Vec<Vec<T>>`/`&[Vec<T>]`, or
    /// a `&PlanarVecs<T>` -- so a caller already holding one of those doesn't need to collect a
    /// temporary `Vec<&[T]>` just to call this.
    ///
    /// `input` must be a *fixed*-size chunk (exactly [`GpuCore::input_chunk_frames`]
    /// samples per channel), matching `CpuCore::process_chunk`'s own contract -- the only
    /// exception is the final call (`is_final = true`), which may be
    /// shorter (or empty). A caller that wants to feed arbitrary-sized reads assembles them into
    /// fixed-size chunks itself (the same split `PlanarResampler`/`InterleavedResampler` already
    /// do on top of the fixed-size `CpuCore::process_chunk`), rather than this core buffering
    /// arbitrary sizes internally.
    ///
    /// Blocks only when the fixed ring has no free buffer-set left and one is needed -- i.e.
    /// when GPU work is fully backlogged relative to `ring_slots`.
    pub fn push_input<'a>(&mut self, input: impl AsChannels<'a, T>, is_final: bool) -> Result<(), GpuError> {
        if self.finalized || self.final_input_seen {
            return Err(GpuError::InvalidSubmissionState(
                "stream has already been finalized".to_string(),
            ));
        }
        let input_samples = if input.channel_count() == 0 {
            0
        } else {
            input.channel(0).len()
        };
        let chunk_frames = self.context.derived().input_chunk_frames;
        if is_final {
            self.final_input_seen = true;
            if input_samples > chunk_frames {
                return Err(GpuError::InvalidConfig(format!(
                    "final chunk must be at most {chunk_frames} samples per channel, got {input_samples}"
                )));
            }
        } else if input_samples != chunk_frames {
            return Err(GpuError::InvalidConfig(format!(
                "expected exactly {chunk_frames} samples per channel, got {input_samples}"
            )));
        }
        if input_samples == 0 {
            return Ok(());
        }
        self.input_sample_count += input_samples;
        self.add_chunk(input, is_final)
    }

    /// Required per-channel input length for a non-final [`GpuCore::push_input`] call.
    pub fn input_chunk_frames(&self) -> usize {
        self.context.derived().input_chunk_frames
    }

    /// Per-channel output length produced by one full (non-final, non-short) input chunk.
    pub fn output_chunk_frames(&self) -> usize {
        self.context.derived().output_chunk_frames
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
    /// After [`GpuCore::finalize`] has returned, every sample the stream will ever produce
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

    /// Copies output samples into `output`, waiting without spinning when GPU work must complete
    /// before any output can be returned.
    ///
    /// Returns immediately when output is already buffered or no submitted/queued GPU work can
    /// produce more output. A partially filled input group is not submitted implicitly; call
    /// [`GpuCore::flush`] first when that group should be made available immediately.
    pub fn pull_output_blocking<'a>(&mut self, mut output: impl AsChannelsMut<'a, T>) -> Result<usize, GpuError> {
        if output.channel_count() != self.context.config().channels {
            return self.drain_ready_output(output);
        }
        let requested = (0..self.context.config().channels)
            .map(|c| output.channel_mut(c).len())
            .min()
            .unwrap_or(0);
        if requested == 0 {
            return Ok(0);
        }

        while self.ready_output_len() == 0 && (self.executing.is_some() || !self.queued.is_empty()) {
            self.advance(true)?;
        }
        self.drain_ready_output(output)
    }

    fn ready_output_len(&self) -> usize {
        self.ready_output.iter().map(Vec::len).min().unwrap_or(0)
    }

    /// Copies up to `min(output channel lengths, buffered sample counts)` samples per channel out
    /// of `self.ready_output` into `output`, then removes exactly that many samples from the
    /// front of each `self.ready_output[c]` (`Vec::drain` shifts the remainder down in place,
    /// keeping the buffer's capacity -- no allocation).
    fn drain_ready_output<'a>(&mut self, mut output: impl AsChannelsMut<'a, T>) -> Result<usize, GpuError> {
        if output.channel_count() != self.context.config().channels {
            return Err(GpuError::InvalidConfig(format!(
                "expected {} channel output slices, got {}",
                self.context.config().channels,
                output.channel_count()
            )));
        }
        let available = self.ready_output_len();
        let requested = (0..self.context.config().channels)
            .map(|c| output.channel_mut(c).len())
            .min()
            .unwrap_or(0);
        let written = available.min(requested);
        for c in 0..self.context.config().channels {
            let dst = output.channel_mut(c);
            dst[..written].copy_from_slice(&self.ready_output[c][..written]);
            self.ready_output[c].drain(..written);
        }
        Ok(written)
    }

    /// Forces any partially-filled group out as its own (possibly smaller-than-`group_chunks`)
    /// submission, then waits for every outstanding GPU submission -- that one included -- to
    /// complete, so every sample pushed so far is guaranteed visible to a following
    /// [`GpuCore::pull_output`] call. Unlike [`GpuCore::finalize`], the stream is not ended:
    /// further [`GpuCore::push_input`] calls remain valid afterward. Combined with `group_chunks(1)`,
    /// this gives a caller a synchronous, per-chunk round trip (`push_input`, `flush`, `pull_output`)
    /// at the cost of giving up the pipelining/backpressure that leaving groups unflushed allows.
    pub fn flush(&mut self) -> Result<(), GpuError> {
        if let Some(filling) = self.filling.take() {
            self.enqueue_filled(filling)?;
        }
        while !self.queued.is_empty() || self.executing.is_some() {
            self.advance(true)?;
        }
        Ok(())
    }

    /// Resets this core so the next [`GpuCore::push_input`] starts an independent new stream, without
    /// rebuilding the ring's steady-state group buffer-sets. Waits for any outstanding GPU
    /// submission to complete first (recycling what it safely can back into the ring), then
    /// clears `pre`/`post`, window history, and all stream bookkeeping. Safe to call whether or
    /// not the previous stream was finalized -- a mid-stream reset abandons whatever was in
    /// flight rather than finishing it.
    ///
    /// The one-off true-stream-start group is never kept in the reusable ring, so it is rebuilt
    /// fresh on the next `push_input` -- a `reset` therefore carries a small one-time GPU
    /// pipeline-build cost on the new stream's first `push_input`, unlike the steady-state ring
    /// slots it leaves intact.
    pub fn reset(&mut self) -> Result<(), GpuError> {
        if let Some(filling) = self.filling.take()
            && filling.recyclable
        {
            self.free_groups.push(filling.group);
        }
        while !self.queued.is_empty() || self.executing.is_some() {
            self.advance(true)?;
        }
        if let Some(prev) = self.prev_completed.take()
            && prev.recyclable
        {
            self.free_groups.push(prev.group);
        }

        for slot in self.pre.iter_mut().chain(self.post.iter_mut()) {
            *slot = None;
        }
        for window in &mut self.prev_input_window {
            window.fill(T::zero());
        }
        for channel in &mut self.ready_output {
            channel.clear();
        }
        self.started = false;
        self.final_input_seen = false;
        self.finalized = false;
        self.trim_remaining = self.context.derived().output_offset;
        self.input_sample_count = 0;
        self.output_sample_count = 0;
        Ok(())
    }

    /// Terminal call: takes no new input (the final chunk, if any, must already have been given
    /// to [`GpuCore::push_input`] with `is_final = true`). If the last chunk `push_input` saw was
    /// exactly full-length -- so its own forward-tail wasn't already folded into a short-final
    /// window -- appends the finalize-tail window here, exactly mirroring
    /// `CpuCore::add_synthetic_finalize_tail_to_overlap`'s own skip condition. Blocks until every
    /// outstanding GPU submission completes, then marks the stream finalized. Does not return
    /// output itself -- call [`GpuCore::pull_output`] in a loop afterward, same as
    /// mid-stream, until it returns `0`; see that method's doc for why that loop is guaranteed
    /// to terminate once this has returned.
    pub fn finalize(&mut self) -> Result<(), GpuError> {
        if self.finalized {
            return Err(GpuError::InvalidSubmissionState(
                "stream has already been finalized".to_string(),
            ));
        }
        self.final_input_seen = true;

        if self.input_sample_count > 0
            && self
                .input_sample_count
                .is_multiple_of(self.context.derived().input_chunk_frames)
        {
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
        let chunk_frames = self.context.derived().input_chunk_frames;
        let is_short_final = is_final_chunk && chunk.channel(0).len() < chunk_frames;

        if self.filling.is_none() {
            if !self.started {
                self.started = true;
                let shaders = self.context.compiled_shaders()?;
                let mut start_group = build_group(&self.context, &[OverlapMode::Start], &shaders)?;
                for c in 0..self.context.config().channels {
                    window::write_start_window(
                        &mut self.window_scratch,
                        chunk_frames,
                        self.context.derived().input_offset,
                        self.pre[c].as_deref(),
                        chunk.channel(c),
                        self.context.derived().extrapolation,
                    );
                    let input_stride = window_input_stride(&start_group);
                    pack_complex_batch(&mut start_group.upload_staging, c, input_stride, &self.window_scratch);
                }
                start_group
                    .pipeline
                    .input_buffer()
                    .upload(&start_group.upload_staging)?;
                let device = Arc::clone(self.context.device());
                let executing = Executing {
                    current: Filled {
                        group: start_group,
                        num_real_chunks: 0,
                        is_final: false,
                        recyclable: false,
                    },
                    seed: None,
                };
                let submission = device
                    .submit_async(executing, |command, executing| {
                        executing.current.group.pipeline.record(command);
                        executing.current.group.overlap.record(command);
                    })
                    .map_err(|(err, _)| err)?;
                // Not waited on here: this becomes the initial `executing` entry, so `advance`
                // only waits for it once the first real group actually needs its overlap state
                // as a seed (`submit_group` reads `prev_completed`), not unconditionally here.
                self.executing = Some(submission);
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
                window_modes: Vec::with_capacity(self.context.group_chunks()),
                windows_filled: 0,
                is_final: false,
                recyclable: true,
            });
        }

        let filling = self.filling.as_mut().expect("just ensured");
        let window_index = filling.windows_filled;
        let input_stride = window_input_stride(&filling.group);
        if is_short_final {
            for c in 0..self.context.config().channels {
                window::write_short_final_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    chunk.channel(c),
                    chunk_frames,
                    self.context.derived().input_offset,
                    self.post[c].as_deref(),
                    self.context.derived().extrapolation,
                );
                pack_complex_batch(
                    &mut filling.group.upload_staging,
                    window_index * self.context.config().channels + c,
                    input_stride,
                    &self.window_scratch,
                );
            }
        } else {
            for c in 0..self.context.config().channels {
                window::write_normal_window(
                    &mut self.window_scratch,
                    self.context.derived().input_offset,
                    chunk.channel(c),
                );
                pack_complex_batch(
                    &mut filling.group.upload_staging,
                    window_index * self.context.config().channels + c,
                    input_stride,
                    &self.window_scratch,
                );
                window::save_current_window(
                    &mut self.prev_input_window[c],
                    &self.window_scratch,
                    self.context.derived().input_offset,
                    chunk_frames,
                );
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

        if filling.windows_filled >= self.context.group_chunks() || is_final_chunk {
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
        let chunk_frames = self.context.derived().input_chunk_frames;
        if let Some(filling) = self.filling.as_mut() {
            let window_index = filling.windows_filled;
            let input_stride = window_input_stride(&filling.group);
            for c in 0..self.context.config().channels {
                window::write_finalize_tail_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    chunk_frames,
                    self.context.derived().input_offset,
                    self.post[c].as_deref(),
                    self.context.derived().extrapolation,
                );
                pack_complex_batch(
                    &mut filling.group.upload_staging,
                    window_index * self.context.config().channels + c,
                    input_stride,
                    &self.window_scratch,
                );
            }
            filling.window_modes.push(OverlapMode::End);
            filling.windows_filled += 1;
            filling.is_final = true;
        } else {
            // Nothing is currently being filled (the last group was already submitted exactly
            // full): build a one-off, single-window `End` group.
            let shaders = self.context.compiled_shaders()?;
            let mut end_group = build_group(&self.context, &[OverlapMode::End], &shaders)?;
            let input_stride = window_input_stride(&end_group);
            for c in 0..self.context.config().channels {
                window::write_finalize_tail_window(
                    &mut self.window_scratch,
                    &mut self.prev_input_window[c],
                    chunk_frames,
                    self.context.derived().input_offset,
                    self.post[c].as_deref(),
                    self.context.derived().extrapolation,
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
        let Filling {
            mut group,
            window_modes,
            windows_filled,
            is_final,
            recyclable,
        } = filling;
        let num_real_chunks = window_modes.iter().filter(|mode| **mode == OverlapMode::Normal).count();

        // A group built with fewer windows than the ring's steady-state shape (the true-final,
        // possibly-short group) needs its own differently-shaped `BatchOverlapShader`; rebuild
        // it in place rather than trying to reuse the pre-built uniform one. Steady-state groups
        // (exactly `group_chunks` `Normal` windows) already match and skip this.
        if windows_filled != self.context.group_chunks() || window_modes.iter().any(|mode| *mode != OverlapMode::Normal)
        {
            let shaders = self.context.compiled_shaders()?;
            let transform_shaders = if group.transform_batch_count == self.context.config().channels {
                &shaders.single
            } else {
                &shaders.grouped
            };
            group.overlap = BatchOverlapShader::<T>::build(
                self.context.device(),
                group.pipeline.output_buffer(),
                group.pipeline.output_stride(),
                self.context.derived().output_chunk_frames,
                self.context.config().channels,
                &window_modes,
                &transform_shaders.overlap,
            )?;
        }

        group.pipeline.input_buffer().upload(&group.upload_staging)?;
        self.queued.push_back(Filled {
            group,
            num_real_chunks,
            is_final,
            recyclable,
        });
        self.advance(false)
    }

    /// One pipeline pump step: if a group is currently executing, either waits for it (`block`)
    /// or polls it once (non-blocking); if it (or nothing) was executing and a queued group is
    /// waiting, submits it. Callers that want to fully drain the pipeline (`finalize`) or wait
    /// for at least one free ring slot (`add_chunk`) loop this externally, checking their own
    /// termination condition between calls -- looping *inside* this method would over-drain the
    /// whole backlog just to free a single slot, defeating the read-ahead this ring exists for.
    fn advance(&mut self, block: bool) -> Result<(), GpuError> {
        if let Some(submission) = &self.executing {
            let ready = if block {
                // `resolve` performs the actual blocking wait below.
                true
            } else {
                submission.is_ready()?
            };
            if ready {
                let submission = self.executing.take().expect("checked above");
                let Executing { current, seed } = submission.resolve()?;
                self.download_and_process(&current)?;
                if let Some(seed) = seed
                    && seed.recyclable
                {
                    self.free_groups.push(seed.group);
                }
                self.prev_completed = Some(current);
            }
        }

        if self.executing.is_none()
            && let Some(next) = self.queued.pop_front()
        {
            self.executing = Some(self.submit_group(next)?);
        }
        Ok(())
    }

    fn submit_group(&mut self, next: Filled<T>) -> Result<GpuSubmission<Executing<T>>, GpuError> {
        let device = Arc::clone(self.context.device());
        let executing = Executing {
            current: next,
            seed: self.prev_completed.take(),
        };
        device
            .submit_async(executing, |command, executing| {
                if let Some(seed) = &executing.seed {
                    executing
                        .current
                        .group
                        .overlap
                        .record_seed_copy(command, seed.group.overlap.overlap_buffer());
                }
                executing.current.group.pipeline.record(command);
                executing.current.group.overlap.record(command);
            })
            .map_err(|(err, executing)| {
                self.prev_completed = executing.seed;
                err
            })
    }

    /// Downloads `filled`'s per-chunk outputs, applies the same trim/output-budget accounting
    /// `CpuCore` applies per chunk, and appends the result to `ready_output`; if
    /// `filled.is_final`, also downloads and appends the finalize-tail from its ending overlap
    /// state.
    fn download_and_process(&mut self, filled: &Filled<T>) -> Result<(), GpuError> {
        let output_chunk_frames = self.context.derived().output_chunk_frames;
        let chunk_output_len = self.context.config().channels * filled.num_real_chunks * output_chunk_frames;
        let chunk_output = &mut self.chunk_output_scratch[..chunk_output_len];
        if filled.num_real_chunks > 0 {
            filled.group.overlap.output_buffer().download(chunk_output)?;
        }
        let expected_total = self.expected_total_output_samples();

        for k in 0..filled.num_real_chunks {
            let skip = self.trim_remaining.min(output_chunk_frames);
            self.trim_remaining -= skip;
            let after_trim = output_chunk_frames - skip;
            let budget = expected_total.map_or(after_trim, |total| {
                after_trim.min(total.saturating_sub(self.output_sample_count))
            });
            self.output_sample_count += budget;
            for c in 0..self.context.config().channels {
                let base = k * self.context.config().channels * output_chunk_frames + c * output_chunk_frames + skip;
                self.ready_output[c].extend_from_slice(&self.chunk_output_scratch[base..base + budget]);
            }
        }

        if filled.is_final {
            filled
                .group
                .overlap
                .overlap_buffer()
                .download(&mut self.overlap_output_scratch)?;
            let scale = T::from(output_chunk_frames).unwrap_or_else(T::one)
                / T::from(self.context.derived().input_chunk_frames).unwrap_or_else(T::one);
            let tail_written = expected_total
                .map_or(output_chunk_frames, |total| {
                    total.saturating_sub(self.output_sample_count)
                })
                .min(output_chunk_frames);
            self.output_sample_count += tail_written;
            for c in 0..self.context.config().channels {
                let base = c * output_chunk_frames;
                self.ready_output[c].extend(
                    self.overlap_output_scratch[base..base + tail_written]
                        .iter()
                        .map(|&value| value * scale),
                );
            }
        }
        Ok(())
    }
}

fn build_group<T: Float + GpuScalar + FromF64>(
    context: &Arc<GpuContext<T>>,
    window_modes: &[OverlapMode],
    shaders: &GpuShaders,
) -> Result<Group<T>, GpuError> {
    let derived = context.derived();
    let channels = context.config().channels;
    let windows = window_modes.len();
    let transform_batch_count = channels * windows;
    let transform_shaders = if windows == 1 {
        &shaders.single
    } else {
        &shaders.grouped
    };
    let geometry = remap_geometry(derived);
    let pipeline = GpuTransformPipeline::<T>::build(
        context.device(),
        T::precision(),
        derived.input_fft_size,
        derived.output_fft_size,
        transform_batch_count,
        &geometry,
        transform_shaders,
    )?;
    let overlap = BatchOverlapShader::<T>::build(
        context.device(),
        pipeline.output_buffer(),
        pipeline.output_stride(),
        derived.output_chunk_frames,
        channels,
        window_modes,
        &transform_shaders.overlap,
    )?;
    let input_stride = pipeline.input_stride();
    let upload_staging = vec![T::zero(); channels * windows * input_stride * 2];
    Ok(Group {
        pipeline,
        overlap,
        upload_staging,
        transform_batch_count,
    })
}

fn normal_window_modes(windows: usize) -> Vec<OverlapMode> {
    vec![OverlapMode::Normal; windows]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_core::CpuCore;
    use crate::extrapolation::Extrapolation;
    use crate::{Config, GpuDevice};

    const DEFAULT_GROUP_CHUNKS: usize = 4;

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
    fn drain_all_output(core: &mut GpuCore<f32>, channels: usize, dst: &mut [Vec<f32>]) {
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
    /// channel) and the pipelined `GpuCore`, both fed one fixed-size FFT chunk at a time
    /// (the last one possibly short, with `is_final = true`) -- `GpuCore::push_input` requires
    /// this exact contract, matching `CpuCore::process_chunk`.
    /// Uses real `pre`/`post` context on both ends (see
    /// [`lpc_extrapolation_has_known_gpu_divergence_without_context`] for why: `Extrapolation::Lpc`
    /// has a known, separately tracked GPU divergence without real context, unrelated to what
    /// this test checks).
    fn assert_batch_matches_cpu(
        input_rate: usize,
        output_rate: usize,
        channels: usize,
        total_frames: usize,
        group_chunks: usize,
    ) {
        let context = match GpuDevice::auto_select() {
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
        let context = GpuContext::with_device(context, config, group_chunks).expect("build GpuContext");
        let mut gpu_core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build GpuCore");
        assert_eq!(gpu_core.input_chunk_frames(), chunk_frames);

        let full = tone_channels(channels, chunk_frames + total_frames + chunk_frames, input_rate);
        let pre: Vec<Vec<f32>> = full.iter().map(|c| c[..chunk_frames].to_vec()).collect();
        let input: Vec<Vec<f32>> = full
            .iter()
            .map(|c| c[chunk_frames..chunk_frames + total_frames].to_vec())
            .collect();
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
                let out = core
                    .process_chunk(&input[c][offset..offset + this_chunk], is_final)
                    .expect("cpu process_chunk");
                cpu_output[c].extend_from_slice(out);
            }
            let refs: Vec<&[f32]> = (0..channels).map(|c| &input[c][offset..offset + this_chunk]).collect();
            gpu_core.push_input(&refs[..], is_final).expect("gpu push");
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
        assert_batch_matches_cpu(
            44_100,
            48_000,
            1,
            chunk_frames * DEFAULT_GROUP_CHUNKS * 2,
            DEFAULT_GROUP_CHUNKS,
        );
    }

    #[test]
    fn pull_output_partial_drain_across_multiple_calls() {
        let context = match GpuDevice::auto_select() {
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
        let context = GpuContext::with_device(context, config, DEFAULT_GROUP_CHUNKS).expect("build context");
        let mut gpu_core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build");

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
            let out = cpu_core
                .process_chunk(&input[0][offset..offset + this_chunk], is_final)
                .expect("cpu process_chunk");
            cpu_output.extend_from_slice(out);
            let refs: [&[f32]; 1] = [&input[0][offset..offset + this_chunk]];
            gpu_core.push_input(&refs[..], is_final).expect("gpu push");

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
        let max_abs_error = cpu_output
            .iter()
            .zip(gpu_output[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_error < 1e-3, "max abs error {max_abs_error}");
    }

    #[test]
    fn push_rejects_wrong_chunk_size() {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let context = GpuContext::with_device(context, Config::new(44_100, 48_000, 1), DEFAULT_GROUP_CHUNKS)
            .expect("build context");
        let mut core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build");
        let wrong_size = vec![0.0f32; core.input_chunk_frames() + 1];
        assert!(core.push_input(&[&wrong_size[..]][..], false).is_err());
    }

    #[test]
    fn ring_slots_are_floored_and_zero_group_chunks_are_rejected() {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let config = Config::new(44_100, 48_000, 2);
        assert!(matches!(
            GpuContext::<f32>::with_device(Arc::clone(&context), config.clone(), 0),
            Err(GpuError::InvalidConfig(_))
        ));
        let context = GpuContext::with_device(context, config, 1).expect("build context");
        let core = GpuCore::<f32>::new(context, 1).expect("build");
        assert_eq!(core.ring_slots(), MIN_RING_SLOTS);
        assert_eq!(core.group_chunks(), 1);
    }

    #[test]
    fn output_chunk_frames_matches_derived_config() {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let config = Config::new(44_100, 48_000, 1);
        let derived = config.derive_config::<f32>().expect("valid config");
        let context = GpuContext::with_device(context, config, DEFAULT_GROUP_CHUNKS).expect("build context");
        let core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build");
        assert_eq!(core.output_chunk_frames(), derived.output_chunk_frames);
    }

    #[test]
    fn flush_makes_partial_group_output_available_without_ending_the_stream() {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let channels = 1;
        let group_chunks = 4;
        let config = Config::new(44_100, 48_000, channels);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();

        let mut cpu_core = CpuCore::<f32>::new(derived.clone());
        let context = GpuContext::with_device(context, config, group_chunks).expect("build context");
        let mut gpu_core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build");

        // Real `pre`/`post` context on both ends, same as `assert_batch_matches_cpu` -- avoids
        // the separately tracked `Extrapolation::Lpc`/GPU divergence unrelated to what this
        // test checks.
        let full = tone_channels(channels, chunk_frames * 6, 44_100);
        let pre = full[0][..chunk_frames].to_vec();
        let input = full[0][chunk_frames..chunk_frames * 5].to_vec();
        let post = full[0][chunk_frames * 5..].to_vec();
        cpu_core.pre(pre.clone());
        cpu_core.post(post.clone());
        gpu_core.pre(vec![pre]);
        gpu_core.post(vec![post]);

        // Two chunks into a group of four: nowhere near a natural group boundary, so nothing
        // would normally be submitted (let alone ready) without an explicit `flush`. Feed
        // `cpu_core` the same two chunks so it stays the correctness oracle for the whole stream.
        let mut cpu_output = Vec::new();
        let out = cpu_core
            .process_chunk(&input[..chunk_frames], false)
            .expect("cpu process_chunk");
        cpu_output.extend_from_slice(out);
        let out = cpu_core
            .process_chunk(&input[chunk_frames..chunk_frames * 2], false)
            .expect("cpu process_chunk");
        cpu_output.extend_from_slice(out);

        let refs: [&[f32]; 1] = [&input[..chunk_frames]];
        gpu_core.push_input(&refs[..], false).expect("push 1");
        let refs: [&[f32]; 1] = [&input[chunk_frames..chunk_frames * 2]];
        gpu_core.push_input(&refs[..], false).expect("push 2");

        let mut scratch = vec![vec![0.0f32; 16_384]; channels];
        let written = gpu_core
            .pull_output_blocking(&mut scratch)
            .expect("pull_output_blocking before flush");
        assert_eq!(
            written, 0,
            "a blocking pull shouldn't submit a partial group implicitly"
        );

        gpu_core.flush().expect("flush");
        let written = gpu_core.pull_output(&mut scratch).expect("pull_output after flush");
        assert!(
            written > 0,
            "flush should force the partial group out and make its output available"
        );

        // The stream is still usable afterward: finish it off (exercising the differently-shaped
        // overlap pipeline a partial, forced-early group builds) and confirm the result still
        // matches `CpuCore`.
        let mut gpu_output = vec![Vec::new(); channels];
        gpu_output[0].extend_from_slice(&scratch[0][..written]);
        let mut offset = chunk_frames * 2;
        loop {
            let remaining = input.len() - offset;
            let is_final = remaining <= chunk_frames;
            let this_chunk = if is_final { remaining } else { chunk_frames };
            let out = cpu_core
                .process_chunk(&input[offset..offset + this_chunk], is_final)
                .expect("cpu process_chunk");
            cpu_output.extend_from_slice(out);
            let refs: [&[f32]; 1] = [&input[offset..offset + this_chunk]];
            gpu_core.push_input(&refs[..], is_final).expect("push");
            drain_all_output(&mut gpu_core, channels, &mut gpu_output);
            offset += this_chunk;
            if is_final {
                break;
            }
        }
        cpu_output.extend_from_slice(cpu_core.finalize().expect("cpu finalize"));
        gpu_core.finalize().expect("gpu finalize");
        drain_all_output(&mut gpu_core, channels, &mut gpu_output);

        assert_eq!(cpu_output.len(), gpu_output[0].len());
        let max_abs_error = cpu_output
            .iter()
            .zip(gpu_output[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_error < 1e-3, "max abs error {max_abs_error}");
    }

    #[test]
    fn reset_allows_reuse_for_an_independent_stream() {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                return;
            }
        };
        let channels = 1;
        let input_rate = 44_100;
        let config = Config::new(input_rate, 48_000, channels);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();
        let total_frames = chunk_frames * 3 + 111;

        let context = GpuContext::with_device(context, config, DEFAULT_GROUP_CHUNKS).expect("build context");
        let mut gpu_core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build");

        // Two structurally identical streams with different content (a different synthetic tone,
        // via a different phase-generation rate) run through the *same* `GpuCore` instance, with
        // a `reset` between them -- proving `reset` leaves no state behind from the first stream.
        for phase_rate in [input_rate, input_rate + 1_234] {
            let mut cpu_core = CpuCore::<f32>::new(derived.clone());
            let full = tone_channels(channels, chunk_frames + total_frames + chunk_frames, phase_rate);
            let pre = full[0][..chunk_frames].to_vec();
            let input = full[0][chunk_frames..chunk_frames + total_frames].to_vec();
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
                let out = cpu_core
                    .process_chunk(&input[offset..offset + this_chunk], is_final)
                    .expect("cpu process_chunk");
                cpu_output.extend_from_slice(out);
                let refs: [&[f32]; 1] = [&input[offset..offset + this_chunk]];
                gpu_core.push_input(&refs[..], is_final).expect("gpu push");
                drain_all_output(&mut gpu_core, channels, &mut gpu_output);
                offset += this_chunk;
                if is_final {
                    break;
                }
            }
            cpu_output.extend_from_slice(cpu_core.finalize().expect("cpu finalize"));
            gpu_core.finalize().expect("gpu finalize");
            drain_all_output(&mut gpu_core, channels, &mut gpu_output);

            assert_eq!(
                cpu_output.len(),
                gpu_output[0].len(),
                "phase_rate {phase_rate}: output length mismatch"
            );
            let max_abs_error = cpu_output
                .iter()
                .zip(gpu_output[0].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_error < 1e-3,
                "phase_rate {phase_rate}: max abs error {max_abs_error}"
            );

            gpu_core.reset().expect("reset");
        }
    }

    /// Runs the same CPU-vs-GPU comparison as [`assert_batch_matches_cpu`], but with no
    /// `pre`/`post` context at all (forcing whichever `extrapolation` strategy `config` selects)
    /// and returns the max abs error instead of asserting, so different strategies can be
    /// compared against each other by the tests below.
    fn max_batch_error_no_context(
        input_rate: usize,
        output_rate: usize,
        total_frames: usize,
        extrapolation: Extrapolation,
    ) -> f32 {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping GPU batch investigation: {err}");
                return 0.0;
            }
        };
        let config = Config::new(input_rate, output_rate, 1).with_extrapolation(extrapolation);
        let derived = config.derive_config::<f32>().expect("valid config");
        let chunk_frames = derived.raw_input_chunk_frames();

        let mut cpu_core = CpuCore::<f32>::new(derived.clone());
        let context = GpuContext::with_device(context, config, DEFAULT_GROUP_CHUNKS).expect("build GpuContext");
        let mut gpu_core = GpuCore::<f32>::new(context, MIN_RING_SLOTS).expect("build GpuCore");

        let input: Vec<f32> = tone_channels(1, total_frames, input_rate).remove(0);

        let mut cpu_output = Vec::new();
        let mut gpu_output = vec![Vec::new(); 1];
        let mut offset = 0;
        loop {
            let remaining = total_frames - offset;
            let is_final = remaining <= chunk_frames;
            let this_chunk = if is_final { remaining } else { chunk_frames };

            let out = cpu_core
                .process_chunk(&input[offset..offset + this_chunk], is_final)
                .expect("cpu process_chunk");
            cpu_output.extend_from_slice(out);

            let refs: [&[f32]; 1] = [&input[offset..offset + this_chunk]];
            gpu_core.push_input(&refs[..], is_final).expect("gpu push");
            drain_all_output(&mut gpu_core, 1, &mut gpu_output);

            offset += this_chunk;
            if is_final {
                break;
            }
        }
        cpu_output.extend_from_slice(cpu_core.finalize().expect("cpu finalize"));
        gpu_core.finalize().expect("gpu finalize");
        drain_all_output(&mut gpu_core, 1, &mut gpu_output);

        assert_eq!(cpu_output.len(), gpu_output[0].len());
        cpu_output
            .iter()
            .zip(gpu_output[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn mirror_and_zero_extrapolation_avoid_lpc_gpu_divergence() {
        for strategy in [Extrapolation::Mirror, Extrapolation::Zero] {
            let err = max_batch_error_no_context(44_100, 96_000, 44_100 + 777, strategy);
            assert!(
                err < 1e-3,
                "{strategy:?}: unexpectedly large GPU/CPU divergence with no pre/post context: {err}"
            );
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
        let err = max_batch_error_no_context(44_100, 96_000, 44_100 + 777, Extrapolation::Lpc);
        assert!(
            err > 0.05,
            "expected the known Extrapolation::Lpc/GPU divergence to still reproduce (got err={err}); \
             if this now passes, the underlying issue may be fixed -- update this test and project memory"
        );
    }
}
