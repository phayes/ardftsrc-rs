use std::collections::VecDeque;

use audio_core::Sample;
use audioadapter_buffers::direct::InterleavedSlice;
use num_traits::Float;
use realfft::FftNum;

use crate::{ChunkResampler, Config, Error};

pub struct StreamingResampler<T = f64>
where
    T: Float + FftNum + Sample,
{
    spans: VecDeque<StreamingSpan<T>>,
}

struct StreamingSpan<T = f64>
where
    T: Float + FftNum + Sample,
{
    inner: ChunkResampler<T>,
    samples_pending_input: VecDeque<T>,
    samples_pending_output: VecDeque<T>,
    samples_input_chunk_buffer: Vec<T>,
    samples_output_chunk_buffer: Vec<T>,
    samples_finalized: bool,
}

impl<T> StreamingResampler<T>
where
    T: Float + FftNum + Sample,
{
    /// Constructs a sample-streaming resampler from `config`.
    pub fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            spans: VecDeque::from([StreamingSpan::new(config)?]),
        })
    }

    /// Returns the configuration for the active writable span.
    #[must_use]
    pub fn config(&self) -> &Config {
        self.active_span().config()
    }

    #[cfg(test)]
    fn input_sample_processed(&self) -> usize {
        self.active_span().inner.input_sample_processed()
    }

    #[cfg(test)]
    #[must_use]
    #[inline]
    fn input_chunk_size(&self) -> usize {
        self.active_span().input_chunk_size()
    }

    #[cfg(test)]
    #[must_use]
    fn output_chunk_size(&self) -> usize {
        self.active_span().output_chunk_size()
    }

    /// Returns the active writable span's algorithmic latency to trim/flush.
    #[must_use]
    pub fn output_delay_frames(&self) -> usize {
        self.active_span().output_delay_frames()
    }

    /// Resets internal streaming state so the next input is treated as a new, independent stream.
    pub fn reset(&mut self) {
        let config = self.active_span().config().clone();
        self.spans = VecDeque::from([StreamingSpan::new(config)
            .expect("ardftsrc: Existing stream config became invalid. This is a bug in the ardftsrc crate.")]);
    }

    /// Starts a new input span while preserving output rate and quality settings.
    ///
    /// Writes will write to the new span immediately. Reads will drain the previous span before moving to the new span.
    ///
    /// If `input_sample_rate` and `channels` match the current writable span, this is a no-op.
    pub fn new_span(&mut self, input_sample_rate: usize, channels: usize) -> Result<(), Error> {
        let current_config = self.active_span().config();
        if current_config.input_sample_rate == input_sample_rate && current_config.channels == channels {
            return Ok(());
        }

        let mut next_config = current_config.clone();
        next_config.input_sample_rate = input_sample_rate;
        next_config.channels = channels;

        let active_span = self.active_span_mut();
        if !active_span.samples_finalized {
            active_span.finalize_samples()?;
        }

        self.spans.push_back(StreamingSpan::new(next_config)?);
        Ok(())
    }

    /// Returns samples left before reads cross into the next queued span.
    ///
    /// `None` means there is no pending span transition. `Some(0)` means a transition is queued and
    /// the next read can enter the next span immediately.
    #[must_use]
    pub fn samples_left_in_span(&self) -> Option<usize> {
        (self.spans.len() > 1).then(|| {
            self.spans
                .front()
                .expect("ardftsrc: StreamingResampler always has at least one span")
                .samples_pending_output
                .len()
        })
    }

    /// Accepts interleaved streaming samples of any length.
    ///
    /// Input is internally buffered and converted into fixed-size chunks. This method does not
    /// return produced output directly; call `read_samples()` to drain available samples.
    pub fn write_samples(&mut self, input: &[T]) -> Result<(), Error> {
        // A write after sample finalization starts a new independent stream.
        if self.active_span().samples_finalized {
            self.reset();
        }

        self.active_span_mut().write_samples(input)
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

    /// Marks the active writable span as finalized and flushes delayed output into pending samples.
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
    pub fn finalize_samples(&mut self) -> Result<(), Error> {
        self.active_span_mut().finalize_samples()
    }

    fn active_span(&self) -> &StreamingSpan<T> {
        self.spans
            .back()
            .expect("ardftsrc: StreamingResampler always has at least one span")
    }

    fn active_span_mut(&mut self) -> &mut StreamingSpan<T> {
        self.spans
            .back_mut()
            .expect("ardftsrc: StreamingResampler always has at least one span")
    }

    fn drop_drained_front_spans(&mut self) {
        while self.spans.len() > 1 && self.spans.front().is_some_and(StreamingSpan::is_drained) {
            self.spans.pop_front();
        }
    }
}

impl<T> StreamingSpan<T>
where
    T: Float + FftNum + Sample,
{
    fn new(config: Config) -> Result<Self, Error> {
        let inner = ChunkResampler::new(config)?;
        let input_chunk_size = inner.input_chunk_size();
        let output_chunk_size = inner.output_chunk_size();

        Ok(Self {
            inner,
            samples_pending_input: VecDeque::with_capacity(input_chunk_size * 2),
            samples_pending_output: VecDeque::with_capacity(output_chunk_size * 2),
            samples_input_chunk_buffer: Vec::with_capacity(input_chunk_size),
            samples_output_chunk_buffer: vec![T::zero(); output_chunk_size],
            samples_finalized: false,
        })
    }

    #[must_use]
    fn config(&self) -> &Config {
        self.inner.config()
    }

    #[must_use]
    #[inline]
    fn input_chunk_size(&self) -> usize {
        self.inner.input_chunk_size()
    }

    #[must_use]
    fn output_chunk_size(&self) -> usize {
        self.inner.output_chunk_size()
    }

    #[must_use]
    fn output_delay_frames(&self) -> usize {
        self.inner.output_delay_frames()
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
            return Err(Error::StreamAlreadyFinalized);
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
        let input_chunk_size = self.input_chunk_size();
        let input_chunk_frames = input_chunk_size / channels;
        let output_chunk_frames = self.output_chunk_size() / channels;

        while self.samples_pending_input.len() >= input_chunk_size {
            self.samples_input_chunk_buffer.clear();
            self.samples_input_chunk_buffer
                .extend(self.samples_pending_input.drain(..input_chunk_size));

            let input_adapter = InterleavedSlice::new(&self.samples_input_chunk_buffer, channels, input_chunk_frames)
                .expect("ardftsrc: Invalid input chunk size. This is a bug in the ardftsrc crate.");
            let mut output_adapter =
                InterleavedSlice::new_mut(&mut self.samples_output_chunk_buffer, channels, output_chunk_frames)
                    .expect("ardftsrc: Invalid output chunk size. This is a bug in the ardftsrc crate.");
            let samples_written = self.inner.process_chunk(&input_adapter, &mut output_adapter)?;

            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
        }

        if finalize && !self.samples_pending_input.is_empty() {
            let remaining_samples = self.samples_pending_input.len();
            debug_assert!(remaining_samples % channels == 0);
            let remaining_frames = remaining_samples / channels;

            self.samples_input_chunk_buffer.clear();
            self.samples_input_chunk_buffer
                .extend(self.samples_pending_input.drain(..));

            let input_adapter = InterleavedSlice::new(&self.samples_input_chunk_buffer, channels, remaining_frames)
                .expect("ardftsrc: Invalid final input chunk size. This is a bug in the ardftsrc crate.");
            let mut output_adapter =
                InterleavedSlice::new_mut(&mut self.samples_output_chunk_buffer, channels, output_chunk_frames)
                    .expect("ardftsrc: Invalid output chunk size. This is a bug in the ardftsrc crate.");
            let samples_written = self.inner.process_chunk_final(&input_adapter, &mut output_adapter)?;

            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
        }

        if finalize {
            let mut output_adapter =
                InterleavedSlice::new_mut(&mut self.samples_output_chunk_buffer, channels, output_chunk_frames)
                    .expect("ardftsrc: Invalid output chunk size. This is a bug in the ardftsrc crate.");
            let samples_written = self.inner.finalize(&mut output_adapter)?;

            self.samples_pending_output
                .extend(self.samples_output_chunk_buffer.iter().copied().take(samples_written));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaperType;

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

    fn input_chunk_frames(resampler: &StreamingResampler<f32>) -> usize {
        resampler.input_chunk_size() / resampler.config().channels
    }

    fn process_chunk_samples(
        resampler: &mut ChunkResampler<f32>,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, Error> {
        let channels = resampler.config().channels;
        let output_len = output.len();
        let input_adapter = InterleavedSlice::new(input, channels, input.len() / channels).map_err(|_| {
            Error::MalformedInputLength {
                channels,
                samples: input.len(),
            }
        })?;
        let mut output_adapter = InterleavedSlice::new_mut(output, channels, output_len / channels).map_err(|_| {
            Error::MalformedInputLength {
                channels,
                samples: output_len,
            }
        })?;
        resampler.process_chunk(&input_adapter, &mut output_adapter)
    }

    fn resample_stream_with_sample_api(
        config: Config,
        input: &[f32],
        write_block_size: usize,
        read_block_size: usize,
    ) -> Vec<f32> {
        let mut resampler = StreamingResampler::new(config).unwrap();
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

        resampler.finalize_samples().unwrap();

        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&read_buffer[..written]);
        }

        output
    }

    fn drain_stream(resampler: &mut StreamingResampler<f32>, read_block_size: usize) -> Vec<f32> {
        let mut output = Vec::new();
        let mut read_buffer = vec![0.0; read_block_size.max(1)];
        loop {
            let written = resampler.read_samples(&mut read_buffer);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&read_buffer[..written]);
        }
        output
    }

    #[test]
    fn write_samples_accepts_non_channel_aligned_input() {
        let mut resampler = StreamingResampler::new(stereo_config(44_100, 48_000)).unwrap();
        let input = vec![0.0; 3];
        assert!(resampler.write_samples(&input).is_ok());
    }

    #[test]
    fn finalize_samples_rejects_dangling_partial_frame() {
        let mut resampler = StreamingResampler::new(stereo_config(44_100, 48_000)).unwrap();
        resampler.write_samples(&[0.0]).unwrap();
        assert!(matches!(
            resampler.finalize_samples(),
            Err(Error::DanglingPartialFrame {
                channels: 2,
                samples: 1
            })
        ));
    }

    #[test]
    fn new_span_matching_format_is_no_op() {
        let mut resampler = StreamingResampler::new(stereo_config(44_100, 48_000)).unwrap();
        resampler.write_samples(&[0.0]).unwrap();

        resampler.new_span(44_100, 2).unwrap();

        assert_eq!(resampler.spans.len(), 1);
        assert_eq!(resampler.samples_left_in_span(), None);
        assert!(matches!(
            resampler.finalize_samples(),
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
        let mut resampler = StreamingResampler::<f32>::new(config).unwrap();

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
        let mut first_offline = ChunkResampler::new(first_config.clone()).unwrap();
        let first_len = first_offline.input_chunk_size() + 7;
        let first_input: Vec<f32> = (0..first_len)
            .map(|frame| (frame as f32 * 0.019).sin() * 0.25)
            .collect();
        let first_expected = first_offline.process_all(&first_input).unwrap();

        let second_config = Config {
            input_sample_rate: 32_000,
            ..first_config.clone()
        };
        let mut second_offline = ChunkResampler::new(second_config).unwrap();
        let second_len = second_offline.input_chunk_size() + 5;
        let second_input: Vec<f32> = (0..second_len)
            .map(|frame| (frame as f32 * 0.023).cos() * 0.2)
            .collect();
        let second_expected = second_offline.process_all(&second_input).unwrap();

        let mut resampler = StreamingResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(32_000, 1).unwrap();
        assert_eq!(resampler.samples_left_in_span(), Some(first_expected.len()));

        resampler.write_samples(&second_input).unwrap();
        resampler.finalize_samples().unwrap();

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
        let mut first_offline = ChunkResampler::new(first_config.clone()).unwrap();
        let first_input: Vec<f32> = (0..(first_offline.input_chunk_size() + 3))
            .map(|frame| (frame as f32 * 0.011).sin() * 0.3)
            .collect();
        let first_expected = first_offline.process_all(&first_input).unwrap();

        let second_config = stereo_config(44_100, 48_000);
        let mut second_offline = ChunkResampler::new(second_config).unwrap();
        let second_frames = second_offline.input_chunk_size() / 2 + 3;
        let mut second_input = Vec::with_capacity(second_frames * 2);
        for frame in 0..second_frames {
            second_input.push((frame as f32 * 0.013).sin() * 0.2);
            second_input.push((frame as f32 * 0.017).cos() * 0.2);
        }
        let second_expected = second_offline.process_all(&second_input).unwrap();

        let mut resampler = StreamingResampler::new(first_config).unwrap();
        resampler.write_samples(&first_input).unwrap();
        resampler.new_span(44_100, 2).unwrap();
        resampler.write_samples(&second_input).unwrap();
        resampler.finalize_samples().unwrap();

        assert_eq!(resampler.config().channels, 2);
        assert_eq!(resampler.samples_left_in_span(), Some(first_expected.len()));

        let mut first_actual = vec![0.0; first_expected.len()];
        assert_eq!(resampler.read_samples(&mut first_actual), first_expected.len());
        assert_eq!(resampler.samples_left_in_span(), Some(0));

        let mut second_actual = vec![0.0; second_expected.len()];
        assert_eq!(resampler.read_samples(&mut second_actual), second_expected.len());
        assert_eq!(resampler.samples_left_in_span(), None);

        for (left, right) in first_actual.iter().zip(first_expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
        for (left, right) in second_actual.iter().zip(second_expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn finalize_samples_rejects_dangling_partial_frame_for_active_span() {
        let mut resampler = StreamingResampler::new(mono_config(44_100, 48_000)).unwrap();
        resampler.new_span(44_100, 2).unwrap();
        resampler.write_samples(&[0.0]).unwrap();

        assert!(matches!(
            resampler.finalize_samples(),
            Err(Error::DanglingPartialFrame {
                channels: 2,
                samples: 1
            })
        ));
    }

    #[test]
    fn write_samples_and_read_samples_match_chunk_output() {
        let config = mono_config(44_100, 48_000);
        let mut chunk_resampler = ChunkResampler::new(config.clone()).unwrap();
        let mut sample_resampler = StreamingResampler::new(config).unwrap();
        let input: Vec<f32> = (0..chunk_resampler.input_chunk_size())
            .map(|frame| (frame as f32 * 0.013).sin() * 0.3)
            .collect();

        let split = input.len() / 2;
        sample_resampler.write_samples(&input[..split]).unwrap();
        sample_resampler.write_samples(&input[split..]).unwrap();

        let mut sample_output = vec![0.0; sample_resampler.output_chunk_size()];
        let sample_written = sample_resampler.read_samples(&mut sample_output);

        let mut chunk_output = vec![0.0; chunk_resampler.output_chunk_size()];
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
        let mut offline = ChunkResampler::new(config.clone()).unwrap();
        let driver = StreamingResampler::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&driver) * 2 + input_chunk_frames(&driver) / 3;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.008).sin() * 0.25)
            .collect();

        let expected = offline.process_all(&input).unwrap();
        let actual = resample_stream_with_sample_api(config, &input, 7, 11);

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn sample_api_stereo_matches_process_all_total_output() {
        let config = stereo_config(44_100, 48_000);
        let mut offline = ChunkResampler::new(config.clone()).unwrap();
        let driver = StreamingResampler::<f32>::new(config.clone()).unwrap();
        let input_frames = input_chunk_frames(&driver) * 2 + 17;
        let mut input = Vec::with_capacity(input_frames * 2);
        for frame in 0..input_frames {
            input.push((frame as f32 * 0.01).sin() * 0.25);
            input.push((frame as f32 * 0.017).cos() * 0.2);
        }

        let expected = offline.process_all(&input).unwrap();
        let actual = resample_stream_with_sample_api(config, &input, 9, 13);

        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((*left - *right).abs() < 1e-5);
        }
    }

    #[test]
    fn finalize_samples_read_until_zero_drains_stream() {
        let config = mono_config(44_100, 48_000);
        let mut offline = ChunkResampler::new(config.clone()).unwrap();
        let mut stream = StreamingResampler::new(config).unwrap();
        let input_frames = input_chunk_frames(&stream) + input_chunk_frames(&stream) / 4;
        let input: Vec<f32> = (0..input_frames)
            .map(|frame| (frame as f32 * 0.011).sin() * 0.2)
            .collect();

        let expected = offline.process_all(&input).unwrap();

        stream.write_samples(&input).unwrap();
        stream.finalize_samples().unwrap();

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
        let mut stream = StreamingResampler::new(mono_config(44_100, 48_000)).unwrap();
        let chunk = stream.input_chunk_size();
        let first = vec![0.1f32; chunk];
        let second = vec![0.2f32; chunk];

        stream.write_samples(&first).unwrap();
        stream.finalize_samples().unwrap();

        // Start a new stream; this should reset core history and sample counters.
        stream.write_samples(&second).unwrap();

        assert_eq!(stream.input_sample_processed(), second.len());
    }
}
