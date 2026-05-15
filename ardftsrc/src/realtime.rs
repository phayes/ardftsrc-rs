use crate::{panic_err, panic_msg};
use num_traits::Float;
use realfft::FftNum;
use std::collections::VecDeque;

use crate::{Config, Error, InterleavedResampler};

// This is sized to cover at least two chunks during priming:
// - First partial chunk
// - Second full chunk
const BUFFER_SIZE_MULTIPLIER: usize = 2;

/// Number of concurrent spans that can be playing at the same time before we allocate.
const DEFAULT_CONCURRENT_SPANS: usize = 4;

/// Realtime reasampler for live audio streams. If you're looking for **rodio** support, see [`RodioResampler`](crate::RodioResampler).
///
/// The real-time resampler allow you to plug ardftsrc into your own realtime audio pipeline. It accepts interleaved samples one-at-a-time.
///
/// 1. Call `write_sample(...)` with each incoming interleaved sample and `read_sample(...)` at your output cadence.
/// 2. For multichannel streams, samples must be written interleaved.
/// 3. Call `new_span(input_sample_rate, channels)` when the input sample rate or channel count changes.
/// 4. Call [`finalize()`](Self::finalize) at end-of-stream, then keep calling [`read_sample(...)`](Self::read_sample) until it returns `None`.
///
/// # Startup Delay
///
/// [`RealtimeResampler`] has some startup delay and will emit negative-zero silence until the resampler is primed and producing samples.
/// You can check [`RealtimeResampler::is_primed()`] to see if the resampler is ready to produce real samples.
/// You can also check [`RealtimeResampler::sample_is_underrun()`] to see if a produced sample is negative-zero silence emitted during initial delay.
///
/// If the upstream source can handle it, on stream startup it is recommended to prime the resampler by pulling samples from the upstream source rapidly, then fast-forwarding until [`RealtimeResampler::is_primed()`] returns `true`.
/// See the [RodioResampler::fast_start() source code](https://github.com/phayes/ardftsrc-rs/blob/master/ardftsrc/src/rodio.rs) for an example on how to do this.
///
/// # Pacing
///
/// If you are wiring [`RealtimeResampler`] into your own realtime audio pipeline, you'll want to keep proper pacing ratios between input and output samples.
/// See the [rodio source](https://github.com/phayes/ardftsrc-rs/blob/master/ardftsrc/src/rodio.rs) for an example on how to do this. If you notice crackling with slow playback,
/// or a very slow response to seeking, those are both symptoms of bad pacing.
///
/// # Spans
///
/// Streaming sources sometimes change format while they are still producing samples.
/// For example, a playlist-like source may play one file at 44.1 kHz stereo and then another at 48 kHz mono.
/// The realtime resampler models those format regions as spans. You can start a new span with [`new_span()`](Self::new_span).
/// When a new span starts, writes go to the new span immediately, and reads continue draining the previous span first before switching to the next.
///
/// Input spans and output spans are non-synchronous.
/// After calling [`new_span()`](Self::new_span), query [`samples_left_in_span()`](Self::samples_left_in_span) to see how many samples are left on the output side before the output will switch to a new span.
///
/// #### Potential allocations on span boundaries
///
/// [`RealtimeResampler`] uses a span pool to avoid allocations. After a span has played, its allocations are returned to the pool to be re-used.
/// Under ideal conditions (playing a single album back to back with no format changes between spans) the resampler will not allocate during playback.
/// However, the resampler may still perform transient allocations at span boundaries under the following conditions:
///   - Rapidly cycling through spans (eg. a user pressing next, next, next, next...)
///   - Playing a heterogeneous playlist with many different sample-rates and channel counts.
///
/// In both of these allocation conditions, the allocation is transient at span boundary and will eventually "settle down" and avoid allocations as the span pool is populated.
///
/// # Buffer Size
///
/// It is recommended to use a buffer before sending output samples to your DAC. Use a buffer size that is at least 2048 to 4096 frames.
/// If you experience crackling, try increasing the buffer size. Marginal buffer capacity first shows up as small glitches on seek.
///
/// # Example
/// ```rust
/// fn resample_streaming(span_1_input: Vec<f32>, span_2_input: Vec<f32>) -> Result<Vec<f32>, ardftsrc::Error> {
///     use ardftsrc::{PRESET_GOOD, RealtimeResampler};
///
///     // Span 1 is 44.1 kHz stereo. Span 2 is 48 kHz mono.
///     // Both spans are resampled to the same 48 kHz output rate.
///     assert!(span_1_input.len().is_multiple_of(2));
///
///     let config = PRESET_GOOD
///         .with_input_rate(44_100)
///         .with_output_rate(48_000)
///         .with_channels(2);
///
///     let mut resampler = RealtimeResampler::<f64>::new(config)?;
///     let mut output = Vec::<f32>::new();
///
///     // This intentionally writes one sample at a time. Larger slices are more efficient,
///     // but single-sample writes are valid.
///     for sample in span_1_input {
///         resampler.write_sample(sample as f64);
///
///         if let Some(sample) = resampler.read_sample() {
///             output.push(sample as f32);
///         }
///
///         if resampler.samples_left_in_span() == Some(0) {
///             // New span detected, maybe switch channel count in output.
///         }
///     }
///
///     resampler.new_span(48_000, 1);
///
///     for sample in span_2_input {
///         resampler.write_sample(sample as f64);
///
///         if let Some(sample) = resampler.read_sample() {
///             output.push(sample as f32);
///         }
///
///         if resampler.samples_left_in_span() == Some(0) {
///             // New span detected, maybe switch channel count in output.
///         }
///     }
///
///     // Finalization can produce delayed tail output, so keep reading until the stream is drained.
///     resampler.finalize();
///     while let Some(sample) = resampler.read_sample() {
///         output.push(sample as f32);
///     }
///
///     Ok(output)
/// }
/// ```
pub struct RealtimeResampler<T = f64>
where
    T: Float + FftNum,
{
    spans: SpanPool<T>,
    single_sample_read_buffer: [T; 1],
    is_primed: bool,
}

impl<T> RealtimeResampler<T>
where
    T: Float + FftNum,
{
    /// Constructs a sample-streaming resampler from [`config`](Config).
    pub fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            spans: SpanPool::new(config, DEFAULT_CONCURRENT_SPANS)?,
            single_sample_read_buffer: [T::zero(); 1],
            is_primed: false,
        })
    }

    #[must_use]
    #[inline]
    /// Returns the number of buffered output samples ready to read.
    pub fn num_samples_ready(&self) -> usize {
        self.active_output_span().samples_pending_output.len()
    }

    /// Returns true when the output buffer has enough samples for realtime [`read_sample()`](Self::read_sample) pulls.
    #[must_use]
    #[inline]
    pub fn is_primed(&mut self) -> bool {
        if self.is_primed {
            return true;
        }

        if self.is_finalized() {
            #[cfg(feature = "tracing")]
            tracing::trace!("RealtimeResampler finalized before bring fully primed");
            self.is_primed = true;
            return true;
        }

        let chunks_processed: usize = self.spans.spans.iter().map(|span| span.chunks_processed).sum();

        self.is_primed = chunks_processed >= 2;
        self.is_primed
    }

    /// Estimates the number of input samples required to prime the resampler.
    ///
    /// This can be inaccurate if there is a span transition during the priming process.
    #[must_use]
    pub fn estimate_priming_samples(&self) -> usize {
        self.active_input_span().input_buffer_size() * 2
    }

    /// Estimates the duration required to prime the resampler.
    ///
    /// This can be inaccurate if there is a span transition during the priming process.
    #[must_use]
    pub fn estimate_priming_duration(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(
            self.estimate_priming_samples() as f64 / self.input_sample_rate() as f64,
        )
    }

    /// Resets internal streaming state so the next input is treated as a new, independent stream.
    ///
    /// Note: this allocates
    pub fn reset(&mut self) {
        let config = self.active_input_span().config().clone();
        self.spans = SpanPool::new(config, DEFAULT_CONCURRENT_SPANS)
            .unwrap_or_else(|err| panic_err("Existing stream config became invalid", err));
        self.is_primed = false;
    }

    /// Starts a new input span while preserving output rate and quality settings.
    ///
    /// Writes will write to the new span immediately. Reads will drain the previous span before moving to the new span.
    ///
    /// If `input_sample_rate` and `channels` match the current input-active span, this is a no-op.
    pub fn new_span(&mut self, input_sample_rate: usize, channels: usize) -> Result<(), Error> {
        let current_config = self.active_input_span().config();
        if current_config.input_sample_rate == input_sample_rate && current_config.channels == channels {
            return Ok(());
        }

        let mut next_config = current_config.clone();
        next_config.input_sample_rate = input_sample_rate;
        next_config.channels = channels;

        // Validate the config to catch any invalid settings.
        next_config.validate()?;

        let active_span = self.active_input_span_mut();
        if !active_span.samples_finalized {
            active_span.finalize_samples()?;
        }

        self.spans.new_span(next_config);
        Ok(())
    }

    /// Returns samples left before reads cross from the output-active span into the next queued span.
    ///
    /// `None` means there is no pending span transition. `Some(0)` means a transition is queued and
    /// the next read can enter the next span immediately.
    #[must_use]
    pub fn samples_left_in_span(&self) -> Option<usize> {
        (self.spans.spans.len() > 1).then(|| {
            self.spans
                .spans
                .front()
                .unwrap_or_else(|| panic_msg("StreamingResampler always has at least one span"))
                .samples_pending_output
                .len()
        })
    }

    /// Returns the output channel count for the next samples that [`read_samples()`](Self::read_samples) would emit.
    ///
    /// When `samples_left_in_span() == Some(0)`, the current output-active span is already drained, so this
    /// reports the queued next span's channel count.
    #[must_use]
    #[inline]
    pub fn output_channels(&self) -> usize {
        self.active_output_span().config().channels
    }

    #[must_use]
    #[inline]
    /// Returns the input channel count for samples currently being written.
    pub fn input_channels(&self) -> usize {
        self.active_input_span().config().channels
    }

    #[must_use]
    #[inline]
    /// Returns the input sample rate for samples currently being written.
    pub fn input_sample_rate(&self) -> usize {
        self.active_input_span().config().input_sample_rate
    }

    #[must_use]
    #[inline]
    /// Returns the output sample rate for samples currently being read.
    pub fn output_sample_rate(&self) -> usize {
        self.active_output_span().config().output_sample_rate
    }

    /// Returns true when the stream has fully completed.
    ///
    /// A stream is considered done when:
    /// - the output-active span has been finalized,
    /// - there is no queued next span, and
    /// - all buffered input/output samples have been drained.
    #[must_use]
    #[inline]
    pub fn is_done(&self) -> bool {
        self.spans.spans.len() == 1 && self.spans.spans.front().is_some_and(RealtimeSpan::is_drained)
    }

    /// Check if this resampler has been finalized.
    ///
    /// A finalized resampler will not accept any more input, but will still
    /// continue to produce output until all buffered input/output samples have been drained.
    ///
    /// Call [`is_done()`](Self::is_done) to check if the resampler is fully drained.
    #[must_use]
    #[inline]
    pub fn is_finalized(&self) -> bool {
        self.spans.is_finalized()
    }

    /// Accepts a single sample.
    #[inline]
    pub fn write_sample(&mut self, sample: T) -> Result<(), Error> {
        self.write_samples(&[sample])
    }

    /// Accepts interleaved streaming samples of any length.
    ///
    /// Input is internally buffered and converted into fixed-size chunks. This method does not
    /// return produced output directly; call `read_samples()` to drain available samples.
    #[inline]
    pub fn write_samples(&mut self, input: &[T]) -> Result<(), Error> {
        self.active_input_span_mut().write_samples(input)
    }

    /// Reads a single sample from the resampler.
    ///
    /// Returns `None` if the resampler is done and no more output will be produced.
    ///
    /// Returns Some(T::neg_zero()) if the resampler is not primed and is producing negative-zero silence during initial priming delay.
    /// Query [`sample_is_underrun()`](Self::sample_is_underrun) to check if a sample is negative-zero silence emitted during initial delay.s
    #[inline]
    pub fn read_sample(&mut self) -> Option<T> {
        if self.is_done() {
            return None;
        }

        if !self.is_primed() {
            return Some(T::neg_zero());
        }

        // Deconstruct self to avoid borrowing issues.
        let (single_sample_read_buffer, span_pool) = (&mut self.single_sample_read_buffer, &mut self.spans);

        let total_read = span_pool.read_samples(single_sample_read_buffer);
        if total_read == 0 {
            return Some(T::neg_zero());
        } else {
            Some(single_sample_read_buffer[0])
        }
    }

    /// Reads up to `output.len()` interleaved samples from internally buffered output.
    ///
    /// Returns one of:
    ///    - Some(written_samples) if samples were written to `output`. Read the inner value to see how many samples were written.
    ///    - Some(0) if the resampler is not primed, the output buffer is zero-length, or the resampler is input starved and is underrunning.
    ///    - None if the resampler is done and no more output will be produced.
    #[must_use]
    pub fn read_samples(&mut self, output: &mut [T]) -> Option<usize> {
        if self.is_done() {
            return None;
        } else if output.len() == 0 || !self.is_primed() {
            return Some(0);
        }

        let total_read = self.spans.read_samples(output);
        Some(total_read)
    }

    /// Marks the input-active span as finalized and flushes delayed output into pending samples.
    ///
    /// After this call, no new input should be written for the current stream. Keep calling
    /// [`read_samples()`](Self::read_samples) until it returns zero to drain all finalized output.
    ///
    /// If your streaming pipeline does not need delayed tail output at end-of-stream, call
    /// [`reset()`](Self::reset) directly instead of `finalize_samples()`. This is specifically for abrupt
    /// switching cases where you intentionally discard the previous stream tail. For normal
    /// endings where tail output is desired, use [`finalize()`](Self::finalize).
    ///
    /// For multi-channel streams, callers must provide a complete interleaved frame via
    /// [`write_samples()`](Self::write_samples) before finalizing. If a dangling partial frame remains buffered, this
    /// method returns [`Error::DanglingPartialFrame`].
    pub fn finalize(&mut self) -> Result<(), Error> {
        self.active_input_span_mut().finalize_samples()
    }

    /// Returns the input-active span (write side).
    ///
    /// This is always the newest queued span (`back`).
    fn active_input_span(&self) -> &RealtimeSpan<T> {
        self.spans
            .spans
            .back()
            .unwrap_or_else(|| panic_msg("StreamingResampler always has at least one span"))
    }

    /// Returns a mutable reference to the input-active span (write side).
    ///
    /// This is always the newest queued span (`back`).
    fn active_input_span_mut(&mut self) -> &mut RealtimeSpan<T> {
        self.spans
            .spans
            .back_mut()
            .unwrap_or_else(|| panic_msg("StreamingResampler always has at least one span"))
    }

    /// Returns the output-active span (read side).
    ///
    /// This is normally the front span. If a queued transition is ready (`Some(0)`),
    /// reads are about to enter the next span and this reports that next span instead.
    fn active_output_span(&self) -> &RealtimeSpan<T> {
        let output_span_index = usize::from(self.samples_left_in_span() == Some(0));
        self.spans
            .spans
            .get(output_span_index)
            .or_else(|| self.spans.spans.front())
            .unwrap_or_else(|| panic_msg("StreamingResampler always has at least one span"))
    }

    /// Returns true when the sample is negative-zero silence emitted during initial delay.
    pub fn sample_is_underrun(sample: T) -> bool {
        sample.is_zero() && sample.is_sign_negative()
    }

    // Test Utility methods
    // ------------------------------------------------------------

    #[cfg(test)]
    /// Returns the number of queued output samples in the current output-active span.
    fn samples_pending_in_output_span(&self) -> usize {
        self.active_output_span().samples_pending_output.len()
    }

    /// Returns the configuration for the input-active span (write side).
    #[cfg(test)]
    fn input_config(&self) -> &Config {
        self.active_input_span().config()
    }

    #[cfg(test)]
    /// Returns the input chunk size in samples for the current write-side span.
    fn input_buffer_size(&self) -> usize {
        self.active_input_span().input_buffer_size()
    }

    #[cfg(test)]
    /// Returns the output chunk size in samples for the current read-side span.
    fn output_buffer_size(&self) -> usize {
        self.active_output_span().output_buffer_size()
    }

    #[cfg(test)]
    fn input_sample_processed(&self) -> usize {
        self.active_input_span().inner.input_sample_processed()
    }
}

/// RealtimeSpan is a single span of audio.
struct RealtimeSpan<T = f64>
where
    T: Float + FftNum,
{
    inner: InterleavedResampler<T>,
    samples_pending_input: VecDeque<T>,
    samples_pending_output: VecDeque<T>,
    samples_input_chunk_buffer: Vec<T>,
    samples_output_chunk_buffer: Vec<T>,
    samples_finalized: bool,
    chunks_processed: usize,
}

impl<T> RealtimeSpan<T>
where
    T: Float + FftNum,
{
    fn new(config: Config) -> Result<Self, Error> {
        let inner = InterleavedResampler::new(config)?;

        let input_buffer_size = inner.input_buffer_size();
        let output_buffer_size = inner.output_buffer_size();

        Ok(Self {
            inner,
            samples_pending_input: VecDeque::with_capacity(input_buffer_size * BUFFER_SIZE_MULTIPLIER),
            samples_pending_output: VecDeque::with_capacity(output_buffer_size * BUFFER_SIZE_MULTIPLIER),
            samples_input_chunk_buffer: Vec::with_capacity(input_buffer_size),
            samples_output_chunk_buffer: vec![T::zero(); output_buffer_size],
            samples_finalized: false,
            chunks_processed: 0,
        })
    }

    /// Re-initializes internal buffers and counters for span reuse.
    pub fn re_initialize(&mut self) {
        self.reset();

        if self.samples_input_chunk_buffer.capacity() < self.inner.input_buffer_size() {
            #[cfg(feature = "tracing")]
            tracing::trace!("RealtimeSpan input chunk buffer is too small, resizing (allocates)");

            self.samples_input_chunk_buffer
                .resize(self.inner.input_buffer_size(), T::zero());
        }

        if self.samples_output_chunk_buffer.capacity() < self.inner.output_buffer_size() {
            #[cfg(feature = "tracing")]
            tracing::trace!("RealtimeSpan output chunk buffer is too small, resizing (allocates)");

            self.samples_output_chunk_buffer
                .resize(self.inner.output_buffer_size(), T::zero());
        }
    }

    #[must_use]
    #[inline]
    fn config(&self) -> &Config {
        self.inner.config()
    }

    #[must_use]
    #[inline]
    fn input_buffer_size(&self) -> usize {
        self.inner.input_buffer_size()
    }

    #[cfg(test)]
    fn output_buffer_size(&self) -> usize {
        self.inner.output_buffer_size()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.samples_pending_input.clear();
        self.samples_pending_output.clear();
        self.samples_finalized = false;
        self.chunks_processed = 0;
    }

    fn write_samples(&mut self, input: &[T]) -> Result<(), Error> {
        // A write after sample finalization starts a new independent stream.
        if self.samples_finalized {
            self.reset();
        }

        self.samples_pending_input.extend(input.iter().copied());
        self.process_pending_samples(false)
    }

    fn read_samples(&mut self, output: &mut [T]) -> usize {
        let drain_count = output.len().min(self.samples_pending_output.len());

        for (dst, src) in output[..drain_count]
            .iter_mut()
            .zip(self.samples_pending_output.drain(..drain_count))
        {
            *dst = src;
        }

        drain_count
    }

    fn finalize_samples(&mut self) -> Result<(), Error> {
        if self.samples_finalized {
            return Err(Error::AlreadyFinalized);
        }

        if !self.samples_pending_input.len().is_multiple_of(self.config().channels) {
            return Err(Error::DanglingPartialFrame {
                channels: self.config().channels,
                samples: self.samples_pending_input.len(),
            });
        }

        self.process_pending_samples(true)?;
        self.samples_finalized = true;
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.samples_finalized && self.samples_pending_input.is_empty() && self.samples_pending_output.is_empty()
    }

    fn process_pending_samples(&mut self, finalize: bool) -> Result<(), Error> {
        let channels = self.config().channels;
        let input_chunk_size = self.input_buffer_size();

        while self.samples_pending_input.len() >= input_chunk_size {
            self.samples_input_chunk_buffer.clear();
            self.samples_input_chunk_buffer
                .extend(self.samples_pending_input.drain(..input_chunk_size));

            let samples_written = self
                .inner
                .process_chunk(&self.samples_input_chunk_buffer, &mut self.samples_output_chunk_buffer)?;
            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
            self.chunks_processed += 1;
        }

        if finalize && !self.samples_pending_input.is_empty() {
            let remaining_samples = self.samples_pending_input.len();
            debug_assert!(remaining_samples % channels == 0);

            self.samples_input_chunk_buffer.clear();
            self.samples_input_chunk_buffer
                .extend(self.samples_pending_input.drain(..));

            let samples_written = self
                .inner
                .process_chunk_final(&self.samples_input_chunk_buffer, &mut self.samples_output_chunk_buffer)?;

            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
            self.chunks_processed += 1;
        }

        if finalize {
            let samples_written = self.inner.finalize(&mut self.samples_output_chunk_buffer)?;
            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
            self.chunks_processed += 1;
        }

        Ok(())
    }
}

/// SpanPool is a pool of spans that can be re-used, avoiding allocations.
struct SpanPool<T = f64>
where
    T: Float + FftNum,
{
    spans: VecDeque<RealtimeSpan<T>>,
    pool: Vec<Option<RealtimeSpan<T>>>,
}

impl<T> SpanPool<T>
where
    T: Float + FftNum,
{
    /// Creates a new span pool with the given config and number of spans.
    fn new(config: Config, num_spans: usize) -> Result<Self, Error> {
        let active_span = RealtimeSpan::new(config.clone())?;

        Ok(Self {
            spans: VecDeque::from([active_span]),
            pool: (0..num_spans)
                .map(|_| Some(RealtimeSpan::new(config.clone()).unwrap()))
                .collect(), // unwrap is fine because we tested the config with active_span creation.
        })
    }

    /// Reads up to `output.len()` interleaved samples from internally buffered output.
    ///
    /// Returns the number of samples copied into `output`.
    ///
    /// This reads from the front of the queue, and will cross span boundaries as needed.
    pub fn read_samples(&mut self, output: &mut [T]) -> usize {
        let mut total_read = 0;

        while total_read < output.len() {
            self.drop_drained_front_spans();

            let span = self
                .spans
                .front_mut()
                .unwrap_or_else(|| panic_msg("StreamingResampler always has at least one span"));
            let read = span.read_samples(&mut output[total_read..]);
            total_read += read;

            if read == 0 {
                break;
            }
        }

        total_read
    }

    /// Drops drained spans from the front of the queue and adds them back to the pool to be re-used.
    fn drop_drained_front_spans(&mut self) {
        while self.spans.len() > 1 && self.spans.front().is_some_and(RealtimeSpan::is_drained) {
            let mut drained_span = self.spans.pop_front().unwrap();
            drained_span.reset();
            self.add_span_to_pool(drained_span);
        }
    }

    /// Adds a span back into the reuse pool, preferring to fill an empty slot before growing the pool.
    fn add_span_to_pool(&mut self, span: RealtimeSpan<T>) {
        if let Some(empty_slot_index) = self.pool.iter().position(Option::is_none) {
            self.pool[empty_slot_index] = Some(span);
        } else {
            #[cfg(feature = "tracing")]
            tracing::trace!("Span pool is full, allocating more pool slots (allocates)");

            self.pool.push(Some(span));
        }
    }

    /// Check if all spans are finalized.
    fn is_finalized(&self) -> bool {
        self.spans.iter().all(|span| span.inner.is_finalized())
    }

    /// Creates a new span and adds it to the end of the queue.
    ///
    /// Note: this may allocate if the span reserve pool is empty, or we cannot find a compatible span in the pool.
    fn new_span(&mut self, config: Config) -> &mut RealtimeSpan<T> {
        // Take a span from the pool if available, otherwise create a new one (allocates).
        // This shouldn't panic as we've already validated the config before we ever got here.
        let span = match self.find_compatible_span_in_pool(&config) {
            Some(span) => {
                let mut span = span;
                span.re_initialize();
                span
            }
            None => {
                #[cfg(feature = "tracing")]
                tracing::trace!("Could not find compatible span in pool, creating a new one (allocates)");

                RealtimeSpan::new(config)
                    .unwrap_or_else(|e| panic_err("Failed to create new span in RealtimeResampler::new_span", e))
            }
        };

        self.spans.push_back(span);
        self.spans
            .back_mut()
            .expect("New span should always be available, we just pushed it to the back of the queue.")
    }

    // Finds a compatible span in the pool that we can re-use.
    fn find_compatible_span_in_pool(&mut self, config: &Config) -> Option<RealtimeSpan<T>> {
        self.pool.iter_mut().find_map(|slot| {
            let span = slot.as_ref()?;
            if span.config().input_sample_rate == config.input_sample_rate
                && span.config().output_sample_rate == config.output_sample_rate
                && span.config().channels == config.channels
            {
                slot.take()
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TaperType,
        test_utils::{assert_no_nans, process_all_samples},
    };

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

    fn input_chunk_frames(resampler: &RealtimeResampler<f32>) -> usize {
        resampler.input_buffer_size() / resampler.input_config().channels
    }

    fn resample_stream_with_sample_api(
        config: Config,
        input: &[f32],
        write_block_size: usize,
        read_block_size: usize,
    ) -> Vec<f32> {
        let mut resampler = RealtimeResampler::new(config).unwrap();
        let mut output = Vec::new();
        let mut read_buffer = vec![0.0; read_block_size.max(1)];
        let channels = resampler.input_config().channels;
        let mut write_step = write_block_size.max(1);
        write_step -= write_step % channels;
        if write_step == 0 {
            write_step = channels;
        }

        let mut offset = 0;
        while offset < input.len() {
            let end = (offset + write_step).min(input.len());
            resampler.write_samples(&input[offset..end]).unwrap();
            offset = end;

            loop {
                let written = resampler.read_samples(&mut read_buffer).unwrap();
                if written == 0 {
                    break;
                }
                output.extend_from_slice(&read_buffer[..written]);
            }
        }

        resampler.finalize().unwrap();

        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written.is_none() {
                break;
            }
            output.extend_from_slice(&read_buffer[..written.unwrap()]);
        }

        assert_no_nans(&output, "streaming::resample_stream_with_sample_api output");
        output
    }

    fn drain_stream(resampler: &mut RealtimeResampler<f32>, read_block_size: usize) -> Vec<f32> {
        let mut output = Vec::new();
        let mut read_buffer = vec![0.0; read_block_size.max(1)];
        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written.is_none() {
                break;
            }
            output.extend_from_slice(&read_buffer[..written.unwrap()]);
        }
        assert_no_nans(&output, "streaming::drain_stream output");
        output
    }

    #[test]
    fn write_samples_accepts_non_channel_aligned_input() {
        let mut resampler = RealtimeResampler::new(stereo_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; 3];
        assert!(resampler.write_samples(&input).is_ok());
    }

    #[test]
    fn finalize_samples_rejects_dangling_partial_frame() {
        let mut resampler = RealtimeResampler::new(stereo_config(44_100, 48_000)).unwrap();
        resampler.write_samples(&[0.0]).unwrap();
        assert!(matches!(
            resampler.finalize(),
            Err(Error::DanglingPartialFrame {
                channels: 2,
                samples: 1
            })
        ));
    }

    #[test]
    fn new_span_matching_format_is_no_op() {
        let mut resampler = RealtimeResampler::new(stereo_config(44_100, 48_000)).unwrap();
        resampler.write_samples(&[0.0]).unwrap();

        resampler.new_span(44_100, 2).unwrap();

        assert_eq!(resampler.spans.spans.len(), 1);
        assert_eq!(resampler.samples_left_in_span(), None);
        assert!(matches!(
            resampler.finalize(),
            Err(Error::DanglingPartialFrame {
                channels: 2,
                samples: 1
            })
        ));
    }

    #[test]
    fn new_span_preserves_output_rate_and_quality_settings() {
        let config = Config {
            quality: 128,
            bandwidth: 0.91,
            taper_type: TaperType::Cosine(2.75),
            ..mono_config(44_100, 48_000)
        };
        let mut resampler = RealtimeResampler::<f32>::new(config).unwrap();

        resampler.new_span(32_000, 2).unwrap();

        assert_eq!(resampler.input_config().input_sample_rate, 32_000);
        assert_eq!(resampler.input_config().output_sample_rate, 48_000);
        assert_eq!(resampler.input_config().channels, 2);
        assert_eq!(resampler.input_config().quality, 128);
        assert_eq!(resampler.input_config().bandwidth, 0.91);
        assert_eq!(resampler.input_config().taper_type, TaperType::Cosine(2.75));
    }

    #[test]
    fn reads_drain_old_span_before_new_span() {
        let first_config = mono_config(44_100, 48_000);
        let mut first_offline = InterleavedResampler::new(first_config.clone()).unwrap();
        let first_len = first_offline.input_buffer_size() + 7;
        let first_input: Vec<f32> = (0..first_len)
            .map(|frame| (frame as f32 * 0.019).sin() * 0.25)
            .collect();
        let first_expected = process_all_samples(&mut first_offline, &first_input).unwrap();

        let second_config = Config {
            input_sample_rate: 32_000,
            ..first_config.clone()
        };
        let mut second_offline = InterleavedResampler::<f32>::new(second_config).unwrap();
        let second_len = second_offline.input_buffer_size() + 5;
        let second_input: Vec<f32> = (0..second_len)
            .map(|frame| (frame as f32 * 0.023).cos() * 0.2)
            .collect();
        let second_expected = process_all_samples(&mut second_offline, &second_input).unwrap();

        let mut resampler = RealtimeResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(32_000, 1).unwrap();
        assert_eq!(resampler.samples_left_in_span(), Some(first_expected.len()));

        resampler.write_samples(&second_input).unwrap();
        resampler.finalize().unwrap();

        let actual = drain_stream(&mut resampler, 13);
        let expected = first_expected
            .iter()
            .chain(second_expected.iter())
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn samples_left_in_span_tracks_channel_change_boundary() {
        let first_config = mono_config(44_100, 48_000);
        let mut first_offline = InterleavedResampler::new(first_config.clone()).unwrap();
        let first_input: Vec<f32> = (0..(first_offline.input_buffer_size() + 3))
            .map(|frame| (frame as f32 * 0.011).sin() * 0.3)
            .collect();
        let first_expected = process_all_samples(&mut first_offline, &first_input).unwrap();

        let second_config = stereo_config(44_100, 48_000);
        let mut second_offline = InterleavedResampler::<f32>::new(second_config).unwrap();
        let second_frames = second_offline.input_buffer_size() / 2 + 3;
        let mut second_input = Vec::with_capacity(second_frames * 2);
        for frame in 0..second_frames {
            second_input.push((frame as f32 * 0.013).sin() * 0.2);
            second_input.push((frame as f32 * 0.017).cos() * 0.2);
        }
        let second_expected = process_all_samples(&mut second_offline, &second_input).unwrap();

        let mut resampler = RealtimeResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(44_100, 2).unwrap();
        resampler.write_samples(&second_input).unwrap();
        resampler.finalize().unwrap();

        assert_eq!(resampler.input_config().channels, 2);
        assert_eq!(resampler.output_channels(), 1);
        assert_eq!(resampler.samples_left_in_span(), Some(first_expected.len()));

        let mut first_actual = vec![0.0; first_expected.len()];
        assert_eq!(resampler.read_samples(&mut first_actual).unwrap(), first_expected.len());
        assert_eq!(resampler.samples_left_in_span(), Some(0));
        assert_eq!(resampler.output_channels(), 2);

        let mut second_actual = vec![0.0; second_expected.len()];
        assert_eq!(
            resampler.read_samples(&mut second_actual).unwrap(),
            second_expected.len()
        );
        assert_eq!(resampler.samples_left_in_span(), None);
        assert_eq!(resampler.output_channels(), 2);

        for (left, right) in first_actual.iter().zip(first_expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
        for (left, right) in second_actual.iter().zip(second_expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn output_buffer_size_tracks_output_active_span_across_transition() {
        let first_config = mono_config(44_100, 48_000);
        let mut first_offline = InterleavedResampler::new(first_config.clone()).unwrap();
        let first_output_buffer_size = first_offline.output_buffer_size();
        let first_input: Vec<f32> = (0..(first_offline.input_buffer_size() + 3))
            .map(|frame| (frame as f32 * 0.009).sin() * 0.2)
            .collect();

        let second_config = stereo_config(32_000, 48_000);
        let second_offline = InterleavedResampler::<f32>::new(second_config).unwrap();
        let second_output_buffer_size = second_offline.output_buffer_size();
        let second_frames = second_offline.input_buffer_size() / 2 + 3;
        let mut second_input = Vec::with_capacity(second_frames * 2);
        for frame in 0..second_frames {
            second_input.push((frame as f32 * 0.013).sin() * 0.2);
            second_input.push((frame as f32 * 0.017).cos() * 0.2);
        }

        assert_ne!(first_output_buffer_size, second_output_buffer_size);

        let mut resampler = RealtimeResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(32_000, 2).unwrap();
        resampler.write_samples(&second_input).unwrap();
        resampler.finalize().unwrap();

        assert_eq!(
            resampler.samples_left_in_span(),
            Some(process_all_samples(&mut first_offline, &first_input).unwrap().len())
        );
        assert_eq!(resampler.output_buffer_size(), first_output_buffer_size);

        let first_read_len = resampler.samples_left_in_span().unwrap();
        let mut first_read = vec![0.0; first_read_len];
        assert_eq!(resampler.read_samples(&mut first_read).unwrap(), first_read_len);
        assert_eq!(resampler.samples_left_in_span(), Some(0));
        assert_eq!(resampler.output_buffer_size(), second_output_buffer_size);
    }

    #[test]
    fn samples_left_in_span_reports_zero_at_exact_span_boundary() {
        let first_config = mono_config(44_100, 48_000);
        let mut first_offline = InterleavedResampler::new(first_config.clone()).unwrap();
        let first_input: Vec<f32> = (0..(first_offline.input_buffer_size() + 3))
            .map(|frame| (frame as f32 * 0.011).sin() * 0.3)
            .collect();
        let first_expected = process_all_samples(&mut first_offline, &first_input).unwrap();
        let first_output_buffer_size = first_offline.output_buffer_size();

        let second_config = stereo_config(44_100, 48_000);
        let mut second_offline = InterleavedResampler::<f32>::new(second_config).unwrap();
        let second_output_buffer_size = second_offline.output_buffer_size();
        let second_frames = second_offline.input_buffer_size() / 2 + 3;
        let mut second_input = Vec::with_capacity(second_frames * 2);
        for frame in 0..second_frames {
            second_input.push((frame as f32 * 0.013).sin() * 0.2);
            second_input.push((frame as f32 * 0.017).cos() * 0.2);
        }
        let second_expected = process_all_samples(&mut second_offline, &second_input).unwrap();

        let mut resampler = RealtimeResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(44_100, 2).unwrap();
        resampler.write_samples(&second_input).unwrap();
        resampler.finalize().unwrap();
        assert_eq!(resampler.output_channels(), 1);
        assert_eq!(resampler.output_buffer_size(), first_output_buffer_size);

        let mut actual = Vec::with_capacity(first_expected.len() + second_expected.len());
        while resampler.samples_left_in_span().unwrap() > 1 {
            let left = resampler.samples_left_in_span().unwrap();
            let chunk_len = left.min(7).min(left - 1);
            let mut chunk = vec![0.0; chunk_len];
            let written = resampler.read_samples(&mut chunk).unwrap();
            assert_eq!(written, chunk_len);
            actual.extend_from_slice(&chunk);
        }

        assert_eq!(resampler.samples_left_in_span(), Some(1));

        let mut boundary_sample = [0.0];
        assert_eq!(resampler.read_samples(&mut boundary_sample), Some(1));
        actual.extend_from_slice(&boundary_sample);

        // The docs guarantee we can observe Some(0) exactly at the boundary.
        assert_eq!(resampler.samples_left_in_span(), Some(0));
        assert_eq!(resampler.output_channels(), 2);
        assert_eq!(resampler.output_buffer_size(), second_output_buffer_size);

        let mut next_span_sample = [0.0];
        assert_eq!(resampler.read_samples(&mut next_span_sample), Some(1));
        actual.extend_from_slice(&next_span_sample);
        assert_eq!(resampler.samples_left_in_span(), None);

        let mut tail = vec![0.0; 19];
        while let Some(written) = resampler.read_samples(&mut tail) {
            actual.extend_from_slice(&tail[..written]);
        }

        let expected = first_expected
            .iter()
            .chain(second_expected.iter())
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn samples_pending_in_output_span_tracks_output_active_span() {
        let first_config = mono_config(44_100, 48_000);
        let mut first_offline = InterleavedResampler::new(first_config.clone()).unwrap();
        let first_input: Vec<f32> = (0..(first_offline.input_buffer_size() + 3))
            .map(|frame| (frame as f32 * 0.011).sin() * 0.3)
            .collect();
        let first_expected = process_all_samples(&mut first_offline, &first_input).unwrap();

        let second_config = stereo_config(44_100, 48_000);
        let mut second_offline = InterleavedResampler::<f32>::new(second_config).unwrap();
        let second_frames = second_offline.input_buffer_size() / 2 + 3;
        let mut second_input = Vec::with_capacity(second_frames * 2);
        for frame in 0..second_frames {
            second_input.push((frame as f32 * 0.013).sin() * 0.2);
            second_input.push((frame as f32 * 0.017).cos() * 0.2);
        }
        let second_expected = process_all_samples(&mut second_offline, &second_input).unwrap();

        let mut resampler = RealtimeResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(44_100, 2).unwrap();
        resampler.write_samples(&second_input).unwrap();
        resampler.finalize().unwrap();

        assert_eq!(resampler.samples_pending_in_output_span(), first_expected.len());
        assert_eq!(resampler.samples_left_in_span(), Some(first_expected.len()));

        let mut first_actual = vec![0.0; first_expected.len()];
        assert_eq!(resampler.read_samples(&mut first_actual).unwrap(), first_expected.len());

        assert_eq!(resampler.samples_left_in_span(), Some(0));
        assert_eq!(resampler.samples_pending_in_output_span(), second_expected.len());
    }

    #[test]
    fn finalize_samples_rejects_dangling_partial_frame_for_active_span() {
        let mut resampler = RealtimeResampler::new(mono_config(44_100, 48_000)).unwrap();
        resampler.new_span(44_100, 2).unwrap();
        resampler.write_samples(&[0.0]).unwrap();

        assert!(matches!(
            resampler.finalize(),
            Err(Error::DanglingPartialFrame {
                channels: 2,
                samples: 1
            })
        ));
    }

    #[test]
    fn sample_api_finalize_matches_process_all_total_output() {
        let config = mono_config(44_100, 48_000);
        let mut offline = InterleavedResampler::new(config.clone()).unwrap();
        let driver = RealtimeResampler::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&driver) * 2 + input_chunk_frames(&driver) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.008).sin() * 0.25)
            .collect();

        let expected = process_all_samples(&mut offline, &input).unwrap();
        let actual = resample_stream_with_sample_api(config, &input, 7, 11);

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn sample_api_stereo_matches_process_all_total_output() {
        let config = stereo_config(44_100, 48_000);
        let mut offline = InterleavedResampler::new(config.clone()).unwrap();
        let driver = RealtimeResampler::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&driver) * 2 + 17;
        let mut input = Vec::with_capacity(input_frames * 2);
        for frame in 0..input_frames {
            input.push((frame as f32 * 0.01).sin() * 0.25);
            input.push((frame as f32 * 0.017).cos() * 0.2);
        }

        let expected = process_all_samples(&mut offline, &input).unwrap();
        let actual = resample_stream_with_sample_api(config, &input, 9, 13);

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn finalize_samples_read_until_zero_drains_stream() {
        let config = mono_config(44_100, 48_000);
        let mut offline = InterleavedResampler::new(config.clone()).unwrap();
        let mut stream = RealtimeResampler::new(config).unwrap();
        let input_frames = input_chunk_frames(&stream) + input_chunk_frames(&stream) / 4;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.011).sin() * 0.2)
            .collect();

        let expected = process_all_samples(&mut offline, &input).unwrap();

        stream.write_samples(&input).unwrap();
        stream.finalize().unwrap();

        let mut actual = Vec::new();
        let mut read_buffer = vec![0.0; 5];
        loop {
            let written = stream.read_samples(&mut read_buffer);
            if written.is_none() {
                break;
            }
            actual.extend_from_slice(&read_buffer[..written.unwrap()]);
        }

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn write_samples_after_finalize_samples_starts_new_stream() {
        let mut stream = RealtimeResampler::new(mono_config(44_100, 48_000)).unwrap();
        let chunk = stream.input_buffer_size();
        let first = vec![0.1f32; chunk];
        let second = vec![0.2f32; chunk];

        stream.write_samples(&first).unwrap();
        stream.finalize().unwrap();

        // Start a new stream; this should reset core history and sample counters.
        stream.write_samples(&second).unwrap();

        assert_eq!(stream.input_sample_processed(), second.len());
    }

    #[test]
    fn is_done_requires_finalize_and_drain() {
        let mut stream = RealtimeResampler::new(mono_config(44_100, 48_000)).unwrap();
        let input_frames = input_chunk_frames(&stream) + 5;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.01).sin() * 0.2)
            .collect();

        assert!(!stream.is_done());
        stream.write_samples(&input).unwrap();
        assert!(!stream.is_done());

        stream.finalize().unwrap();
        assert!(!stream.is_done());

        let _ = drain_stream(&mut stream, 11);
        assert!(stream.is_done());
    }

    #[test]
    fn is_done_false_while_next_span_is_queued() {
        let first_config = mono_config(44_100, 48_000);
        let mut stream = RealtimeResampler::new(first_config).unwrap();
        let input = vec![0.1f32; stream.input_buffer_size() + 3];

        stream.write_samples(&input).unwrap();
        stream.new_span(32_000, 1).unwrap();
        assert!(!stream.is_done());
        assert!(stream.samples_left_in_span().is_some());
    }
}
