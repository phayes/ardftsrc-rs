use std::collections::VecDeque;

use num_traits::Float;
use realfft::FftNum;

use crate::{Config, Error, InterleavedResampler};

pub(crate) const BUFFER_SIZE_MULTIPLIER: usize = 2;

pub struct RealtimeResampler<T = f64>
where
    T: Float + FftNum,
{
    spans: SpanPool<T>,
    single_sample_read_buffer: [T; 1],

    #[cfg(feature = "tracing")]
    underrun_count: usize,
}

struct StreamingSpan<T = f64>
where
    T: Float + FftNum,
{
    inner: InterleavedResampler<T>,
    samples_pending_input: VecDeque<T>,
    samples_pending_output: VecDeque<T>,
    samples_input_chunk_buffer: Vec<T>,
    samples_output_chunk_buffer: Vec<T>,
    samples_finalized: bool,
}

// TODO: Move various methods from RealtimeResampler to here
// TODO: Make SpanPool re-use spans instead of discarding them and creating new ones
struct SpanPool<T = f64>
where
    T: Float + FftNum,
{
    spans: VecDeque<StreamingSpan<T>>,
}

impl<T> SpanPool<T>
where
    T: Float + FftNum,
{
    fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            spans: VecDeque::from([StreamingSpan::new(config)?]),
        })
    }

    /// Reads up to `output.len()` interleaved samples from internally buffered output.
    ///
    /// Returns the number of samples copied into `output`.
    pub fn read_samples(&mut self, output: &mut [T]) -> usize {
        let mut total_read = 0;

        while total_read < output.len() {
            self.drop_drained_front_spans();

            let span = self
                .spans
                .front_mut()
                .expect("ardftsrc: StreamingResampler always has at least one span");
            let read = span.read_samples(&mut output[total_read..]);
            total_read += read;

            if read == 0 {
                break;
            }
        }

        total_read
    }

    fn drop_drained_front_spans(&mut self) {
        while self.spans.len() > 1 && self.spans.front().is_some_and(StreamingSpan::is_drained) {
            self.spans.pop_front();
        }
    }
}

impl<T> RealtimeResampler<T>
where
    T: Float + FftNum,
{
    /// Constructs a sample-streaming resampler from `config`.
    pub fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            spans: SpanPool::new(config)?,
            single_sample_read_buffer: [T::zero(); 1],
            #[cfg(feature = "tracing")]
            underrun_count: 0,
        })
    }

    /// Returns the configuration for the input-active span (write side).
    #[must_use]
    #[cfg(test)]
    pub fn config(&self) -> &Config {
        self.active_input_span().config()
    }

    #[cfg(test)]
    fn input_sample_processed(&self) -> usize {
        self.active_input_span().inner.input_sample_processed()
    }

    #[must_use]
    #[inline]
    pub fn input_buffer_size(&self) -> usize {
        self.active_input_span().input_buffer_size()
    }

    #[must_use]
    #[inline]
    pub fn output_buffer_size(&self) -> usize {
        self.active_output_span().output_buffer_size()
    }

    #[must_use]
    #[inline]
    pub fn num_samples_ready(&self) -> usize {
        self.active_output_span().samples_pending_output.len()
    }

    /// Resets internal streaming state so the next input is treated as a new, independent stream.
    pub fn reset(&mut self) {
        let config = self.active_input_span().config().clone();
        self.spans = SpanPool::new(config)
            .expect("ardftsrc: Existing stream config became invalid. This is a bug in the ardftsrc crate.");
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

        let active_span = self.active_input_span_mut();
        if !active_span.samples_finalized {
            active_span.finalize_samples()?;
        }

        self.spans.spans.push_back(StreamingSpan::new(next_config)?);
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
                .expect("ardftsrc: StreamingResampler always has at least one span")
                .samples_pending_output
                .len()
        })
    }

    /// Returns the output channel count for the next samples that `read_samples()` would emit.
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
    pub fn input_channels(&self) -> usize {
        self.active_input_span().config().channels
    }

    #[must_use]
    #[inline]
    pub fn input_sample_rate(&self) -> usize {
        self.active_input_span().config().input_sample_rate
    }

    #[must_use]
    #[inline]
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
    pub fn is_done(&self) -> bool {
        self.spans.spans.len() == 1 && self.spans.spans.front().is_some_and(StreamingSpan::is_drained)
    }

    /// Accepts interleaved streaming samples of any length.
    ///
    /// Input is internally buffered and converted into fixed-size chunks. This method does not
    /// return produced output directly; call `read_samples()` to drain available samples.
    pub fn write_samples(&mut self, input: &[T]) -> Result<(), Error> {
        // A write after sample finalization starts a new independent stream.
        if self.active_input_span().samples_finalized {
            self.reset();
        }

        self.active_input_span_mut().write_samples(input)
    }

    pub fn read_sample(&mut self) -> Option<T> {
        // Deconstruct self to avoid borrowing issues.
        let (single_sample_read_buffer, span_pool) = (&mut self.single_sample_read_buffer, &mut self.spans);

        let total_read = span_pool.read_samples(single_sample_read_buffer);
        if total_read == 0 {
            #[cfg(feature = "tracing")]
            if self.underrun_count == 0 {
                tracing::warn!("ardftsrc: RealtimeResampler underrun");
                self.underrun_count += 1;
            }

            Some(T::neg_zero())
        } else {
            #[cfg(feature = "tracing")]
            if self.underrun_count > 0 {
                tracing::info!(
                    "ardftsrc: RealtimeResampler underrun recovered after {} samples",
                    self.underrun_count
                );
                self.underrun_count = 0;
            }

            Some(single_sample_read_buffer[0])
        }
    }

    /// Reads up to `output.len()` interleaved samples from internally buffered output.
    ///
    /// Returns the number of samples copied into `output`.
    pub fn read_samples(&mut self, output: &mut [T]) -> usize {
        self.spans.read_samples(output)
    }

    /// Marks the input-active span as finalized and flushes delayed output into pending samples.
    ///
    /// After this call, no new input should be written for the current stream. Keep calling
    /// `read_samples()` until it returns zero to drain all finalized output.
    ///
    /// If your streaming pipeline does not need delayed tail output at end-of-stream, call
    /// `reset()` directly instead of `finalize_samples()`. This is specifically for abrupt
    /// switching cases where you intentionally discard the previous stream tail. For normal
    /// endings where tail output is desired, use `finalize_samples()`.
    ///
    /// For multi-channel streams, callers must provide a complete interleaved frame via
    /// `write_samples()` before finalizing. If a dangling partial frame remains buffered, this
    /// method returns `Error::DanglingPartialFrame`.
    pub fn finalize(&mut self) -> Result<(), Error> {
        self.active_input_span_mut().finalize_samples()
    }

    #[must_use]
    pub fn samples_pending_in_output_span(&self) -> usize {
        self.active_output_span().samples_pending_output.len()
    }

    /// Returns the input-active span (write side).
    ///
    /// This is always the newest queued span (`back`).
    fn active_input_span(&self) -> &StreamingSpan<T> {
        self.spans
            .spans
            .back()
            .expect("ardftsrc: StreamingResampler always has at least one span")
    }

    /// Returns a mutable reference to the input-active span (write side).
    ///
    /// This is always the newest queued span (`back`).
    fn active_input_span_mut(&mut self) -> &mut StreamingSpan<T> {
        self.spans
            .spans
            .back_mut()
            .expect("ardftsrc: StreamingResampler always has at least one span")
    }

    /// Returns the output-active span (read side).
    ///
    /// This is normally the front span. If a queued transition is ready (`Some(0)`),
    /// reads are about to enter the next span and this reports that next span instead.
    fn active_output_span(&self) -> &StreamingSpan<T> {
        let output_span_index = usize::from(self.samples_left_in_span() == Some(0));
        self.spans
            .spans
            .get(output_span_index)
            .or_else(|| self.spans.spans.front())
            .expect("ardftsrc: StreamingResampler always has at least one span")
    }
}

impl<T> StreamingSpan<T>
where
    T: Float + FftNum,
{
    fn new(config: Config) -> Result<Self, Error> {
        let inner = InterleavedResampler::new(config)?;
        let input_chunk_size = inner.input_buffer_size();
        let output_chunk_size = inner.output_buffer_size();

        Ok(Self {
            inner,
            samples_pending_input: VecDeque::with_capacity(input_chunk_size * BUFFER_SIZE_MULTIPLIER),
            samples_pending_output: VecDeque::with_capacity(output_chunk_size * BUFFER_SIZE_MULTIPLIER),
            samples_input_chunk_buffer: Vec::with_capacity(input_chunk_size),
            samples_output_chunk_buffer: vec![T::zero(); output_chunk_size],
            samples_finalized: false,
        })
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

    #[must_use]
    #[inline]
    fn output_buffer_size(&self) -> usize {
        self.inner.output_buffer_size()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.samples_pending_input.clear();
        self.samples_pending_output.clear();
        self.samples_finalized = false;
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
        }

        if finalize {
            let samples_written = self.inner.finalize(&mut self.samples_output_chunk_buffer)?;

            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
        }

        Ok(())
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
        resampler.input_buffer_size() / resampler.config().channels
    }

    fn process_chunk_samples(
        resampler: &mut InterleavedResampler<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, Error> {
        let written = resampler.process_chunk(input, output)?;
        assert_no_nans(&output[..written], "streaming::process_chunk_samples output");
        Ok(written)
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
        let channels = resampler.config().channels;
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
                let written = resampler.read_samples(&mut read_buffer);
                if written == 0 {
                    break;
                }
                output.extend_from_slice(&read_buffer[..written]);
            }
        }

        resampler.finalize().unwrap();

        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&read_buffer[..written]);
        }

        assert_no_nans(&output, "streaming::resample_stream_with_sample_api output");
        output
    }

    fn drain_stream(resampler: &mut RealtimeResampler<f32>, read_block_size: usize) -> Vec<f32> {
        let mut output = Vec::new();
        let mut read_buffer = vec![0.0; read_block_size.max(1)];
        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&read_buffer[..written]);
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

        assert_eq!(resampler.config().input_sample_rate, 32_000);
        assert_eq!(resampler.config().output_sample_rate, 48_000);
        assert_eq!(resampler.config().channels, 2);
        assert_eq!(resampler.config().quality, 128);
        assert_eq!(resampler.config().bandwidth, 0.91);
        assert_eq!(resampler.config().taper_type, TaperType::Cosine(2.75));
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

        assert_eq!(resampler.config().channels, 2);
        assert_eq!(resampler.output_channels(), 1);
        assert_eq!(resampler.samples_left_in_span(), Some(first_expected.len()));

        let mut first_actual = vec![0.0; first_expected.len()];
        assert_eq!(resampler.read_samples(&mut first_actual), first_expected.len());
        assert_eq!(resampler.samples_left_in_span(), Some(0));
        assert_eq!(resampler.output_channels(), 2);

        let mut second_actual = vec![0.0; second_expected.len()];
        assert_eq!(resampler.read_samples(&mut second_actual), second_expected.len());
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
        assert_eq!(resampler.read_samples(&mut first_read), first_read_len);
        assert_eq!(resampler.samples_left_in_span(), Some(0));
        assert_eq!(resampler.output_buffer_size(), second_output_buffer_size);
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
        assert_eq!(resampler.read_samples(&mut first_actual), first_expected.len());

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
    fn write_samples_and_read_samples_match_chunk_output() {
        let config = mono_config(44_100, 48_000);
        let mut chunk_resampler = InterleavedResampler::new(config.clone()).unwrap();
        let mut sample_resampler = RealtimeResampler::new(config).unwrap();
        let input: Vec<f32> = (0..chunk_resampler.input_buffer_size())
            .map(|frame| (frame as f32 * 0.013).sin() * 0.3)
            .collect();

        let split = input.len() / 2;
        sample_resampler.write_samples(&input[..split]).unwrap();
        sample_resampler.write_samples(&input[split..]).unwrap();

        let mut sample_output = vec![0.0; sample_resampler.output_buffer_size()];
        let sample_written = sample_resampler.read_samples(&mut sample_output);

        let mut chunk_output = vec![0.0; chunk_resampler.output_buffer_size()];
        let chunk_written = process_chunk_samples(&mut chunk_resampler, &input, &mut chunk_output).unwrap();

        assert_eq!(sample_written, chunk_written);
        for (left, right) in sample_output[..sample_written]
            .iter()
            .zip(chunk_output[..chunk_written].iter())
        {
            assert!((*left - *right).abs() < 1e-5);
        }
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
            if written == 0 {
                break;
            }
            actual.extend_from_slice(&read_buffer[..written]);
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
