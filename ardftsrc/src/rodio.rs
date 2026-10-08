use crate::SamplesLeftInSpan;
use crate::{Config, Error, RealtimeResampler, panic_err, panic_msg};
use num_traits::Float;
use realfft::FftNum;
/// Wrap a [`rodio::Source`] and resample it in realtime in your rodio pipeline. Requires the `rodio` feature.
///
/// When playing from a buffered audio source such as a file or a buffered stream, it is recommended to use [`config.with_rodio_fast_start(true)`](Config::with_rodio_fast_start), which will
/// avoid initial output delay by pulling samples from the upstream source to prime the resampler. For very-realtime sources such as microphones or similar,
/// do not enable fast-start.
///
/// Seeking discards all audio buffered from before the seek. With fast-start enabled, the resampler is re-primed inside
/// `try_seek()` so post-seek audio plays immediately; this decodes and resamples two chunks on the audio thread, which takes
/// noticeably longer at high quality settings. Without fast-start, output is silent until the resampler is primed again.
///
/// Be aware that because RodioResampler resamples on the audio thread, your cpal buffer size should be at least 2048 to 4096.
/// If you experience crackling, try increasing the cpal buffer size. Marginal buffer capacity first shows up as small glitches on seek.
///
/// # Example:
/// ```rust
/// let stream = rodio::DeviceSinkBuilder::open_default_sink()?;
/// let mixer = stream.mixer();
///
/// let tone = rodio::source::SignalGenerator::new(
///     NonZero::new(44_100 as u32).unwrap(),
///     400, // 400 Hz
///     rodio::source::Function::Sine,
/// )
/// .take_duration(Duration::from_secs(3.0));
///
/// let config = PRESET_FAST.with_channels(1).with_input_rate(44_100).with_output_rate(48_000);
/// let resampled_tone = RodioResampler::new(tone, config)?;
///
/// mixer.add(resampled_tone);
/// thread::sleep(Duration::from_secs(4));
///```
pub struct RodioResampler<S: rodio::Source, T = f64>
where
    T: Float + FftNum,
{
    inner: S,
    resampler: RealtimeResampler<T>,
    config: Config,
    stream_input_ended: bool,
    samples_this_span: u64,
    output_samples_this_span: u64,
    span_ratio: f64,
    /// Length of the current inner span in samples, or `None` when the next pull starts a new inner span.
    /// Sources without span boundaries are tracked as `u64::MAX`.
    inner_span_len: Option<u64>,
    inner_span_samples: u64,
    /// Samples written to the resampler for the current, incomplete input frame.
    input_frame_offset: usize,
    output_frame_samples_remaining: usize,
    output_frame_channels: usize,
    output_frame_is_startup_silence: bool,
}

impl<S, T> RodioResampler<S, T>
where
    S: rodio::Source,
    T: Float + FftNum,
{
    fn new_typed(inner: S, config: Config) -> Result<Self, Error> {
        let fast_start = config.rodio_fast_start;
        let resampler = RealtimeResampler::new(config.clone())?;
        let span_ratio = resampler.input_sample_rate() as f64 / resampler.output_sample_rate() as f64;

        #[cfg(feature = "tracing")]
        tracing::trace!(
            "Creating resampler. Input rate: {}, Output rate: {} (ratio: {})",
            resampler.input_sample_rate(),
            resampler.output_sample_rate(),
            span_ratio
        );

        let mut rodio_resampler = Self {
            inner,
            resampler,
            config,
            stream_input_ended: false,
            samples_this_span: 0,
            output_samples_this_span: 0,
            span_ratio,
            inner_span_len: None,
            inner_span_samples: 0,
            input_frame_offset: 0,
            output_frame_samples_remaining: 0,
            output_frame_channels: 0,
            output_frame_is_startup_silence: false,
        };
        rodio_resampler.set_span_ratio();
        if fast_start {
            rodio_resampler.fast_start();
        }

        Ok(rodio_resampler)
    }

    fn set_span_ratio(&mut self) {
        self.span_ratio = self.resampler.input_sample_rate() as f64 / self.resampler.output_sample_rate() as f64;
    }

    fn maybe_new_input_span(&mut self) -> bool {
        let current_input_sample_rate = self.resampler.input_sample_rate();
        let current_input_channels = self.resampler.input_channels();

        let input_sample_rate = self.inner.sample_rate().get() as usize;
        let input_channels = self.inner.channels().get() as usize;

        if current_input_sample_rate != input_sample_rate || current_input_channels != input_channels {
            self.resampler
                .new_span(input_sample_rate, input_channels)
                .unwrap_or_else(|err| panic_err("failed to create new input span", err));
            self.samples_this_span = 0;
            self.output_samples_this_span = 0;
            self.input_frame_offset = 0;
            self.set_span_ratio();

            #[cfg(feature = "tracing")]
            tracing::trace!(
                "new input span started: {} -> {} (ratio: {})",
                input_sample_rate,
                input_channels,
                self.span_ratio,
            );
            true
        } else {
            false
        }
    }

    /// Estimate the number of input samples to pull this tick.
    fn calculate_inner_pulls(&mut self) -> u64 {
        // If the inner stream is ended, return zero.
        if self.stream_input_ended {
            return 0;
        }

        // Otherwise, calculate the number of input samples to pull to keep output production approximately aligned with input consumption given the current span ratio.
        self.output_samples_this_span = self.output_samples_this_span.saturating_add(1);
        let target_input_samples = (self.output_samples_this_span as f64 * self.span_ratio).ceil() as u64;
        target_input_samples.saturating_sub(self.samples_this_span)
    }

    // Pull a sample from the inner source and write it to the resampler.
    fn pull_inner_sample(&mut self, count_samples: bool) {
        // If input is none, end the stream, but keep reading until the resampler is drained.
        let Some(sample) = self.inner.next() else {
            if !self.stream_input_ended {
                self.stream_input_ended = true;
                self.resampler
                    .finalize()
                    .unwrap_or_else(|err| panic_err("failed to finalize resampler", err));
            }
            return;
        };

        // First sample of a new inner span. Sources may only expose the new span's format and length once
        // its first sample has been pulled (rodio's decoders and source-chaining adapters advance inside
        // `next()`), so read them now, before the sample is written into a resampler span.
        if self.inner_span_len.is_none() {
            self.maybe_new_input_span();
            let span_len = self.inner.current_span_len().map_or(u64::MAX, |len| len as u64);
            debug_assert!(
                span_len == u64::MAX || span_len.is_multiple_of(u64::from(self.inner.channels().get())),
                "ardftsrc: Error in inner source: current_span_len should be a multiple of channels"
            );
            self.inner_span_len = Some(span_len);
            self.inner_span_samples = 0;
        }

        self.resampler
            .write_samples(&[num_traits::cast(sample).unwrap()])
            .unwrap_or_else(|err| panic_err("failed to write sample", err));
        if count_samples {
            self.samples_this_span += 1;
        }
        self.input_frame_offset = (self.input_frame_offset + 1) % self.resampler.input_channels();

        self.inner_span_samples += 1;
        if self
            .inner_span_len
            .is_some_and(|span_len| self.inner_span_samples >= span_len)
        {
            self.inner_span_len = None;
        }
    }

    // Fast-start the resampler by pulling samples from the inner source until the resampler is primed.
    fn fast_start(&mut self) {
        while !self.resampler.is_primed() {
            if self.stream_input_ended {
                break;
            }
            self.pull_inner_sample(false);
        }
    }

    #[inline]
    fn next_sample(&mut self) -> Option<T> {
        let starts_output_frame = self.output_frame_samples_remaining == 0;
        if starts_output_frame && self.resampler.is_done() {
            return None;
        }

        // Keep input consumption approximately aligned with output production:
        // pull 0 or multiple input samples depending on span_ratio and current drift.
        let inner_pulls = self.calculate_inner_pulls();

        for _ in 0..inner_pulls {
            self.pull_inner_sample(true);
        }

        if starts_output_frame {
            if self.resampler.is_done() {
                return None;
            }

            self.output_frame_channels = self.resampler.output_channels();
            self.output_frame_samples_remaining = self.output_frame_channels;
            self.output_frame_is_startup_silence =
                !self.resampler.is_primed() || self.resampler.num_samples_ready() < self.output_frame_channels;
        }

        let sample = if self.output_frame_is_startup_silence {
            T::neg_zero()
        } else {
            self.resampler
                .read_sample()
                .unwrap_or_else(|| panic_msg("primed resampler ended before emitting a complete output frame"))
        };

        self.output_frame_samples_remaining -= 1;
        Some(sample)
    }
}

impl<S> RodioResampler<S, f64>
where
    S: rodio::Source,
{
    /// Create a new RodioResampler using `f64` as the internal resampling type.
    ///
    /// Config input sample rate and channel count can be a best-guess if you don't know the exact values at the time of construction.
    /// If they are innacuate, a new span will be created when the actual values are known.
    pub fn new(inner: S, config: Config) -> Result<Self, Error> {
        Self::new_typed(inner, config)
    }
}

impl<S> RodioResampler<S, f32>
where
    S: rodio::Source,
{
    /// Create a new RodioResampler using `f32` as the internal resampling type.
    ///
    /// Config input sample rate and channel count can be a best-guess if you don't know the exact values at the time of construction.
    /// If they are innacuate, a new span will be created when the actual values are known.
    pub fn new_f32(inner: S, config: Config) -> Result<Self, Error> {
        Self::new_typed(inner, config)
    }
}

impl<S, T> Iterator for RodioResampler<S, T>
where
    S: rodio::Source,
    T: Float + FftNum,
{
    type Item = rodio::Sample;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample().map(|sample| {
            num_traits::cast(sample)
                .unwrap_or_else(|| panic_msg("resampler sample should be representable as rodio sample"))
        })
    }
}

impl<S, T> rodio::Source for RodioResampler<S, T>
where
    S: rodio::Source,
    T: Float + FftNum,
{
    fn sample_rate(&self) -> std::num::NonZero<u32> {
        std::num::NonZero::new(self.resampler.output_sample_rate() as u32).unwrap()
    }

    fn channels(&self) -> std::num::NonZero<u16> {
        let channels = if self.output_frame_samples_remaining > 0 {
            self.output_frame_channels
        } else {
            self.resampler.output_channels()
        };
        std::num::NonZero::new(channels as u16).unwrap()
    }

    fn total_duration(&self) -> Option<core::time::Duration> {
        self.inner.total_duration().map(|inner_duration| {
            if self.config.rodio_fast_start {
                inner_duration
            } else {
                inner_duration + self.resampler.estimate_priming_duration()
            }
        })
    }

    fn current_span_len(&self) -> Option<usize> {
        if self.output_frame_samples_remaining > 0 {
            return Some(self.output_frame_channels);
        }

        let channels = self.resampler.output_channels();
        match self.resampler.samples_left_in_span() {
            SamplesLeftInSpan::Known(0) => Some(channels),
            SamplesLeftInSpan::Known(samples_left) => {
                if !samples_left.is_multiple_of(channels) {
                    panic_msg("output span ended with a partial frame");
                }
                Some(samples_left)
            }
            SamplesLeftInSpan::Unknown => {
                self.inner.current_span_len()?;

                let samples_ready = self.resampler.num_samples_ready();
                if samples_ready == 0 {
                    Some(channels)
                } else {
                    if !samples_ready.is_multiple_of(channels) {
                        panic_msg("output buffer contains a partial frame");
                    }
                    Some(samples_ready)
                }
            }
            SamplesLeftInSpan::EndOfStream => Some(0),
        }
    }

    fn try_seek(&mut self, time: core::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(time)?;

        // Discard everything buffered from before the seek. Otherwise up to two resampler chunks of stale
        // audio would play first, which is several seconds at high quality settings.
        self.resampler.reset();
        self.stream_input_ended = false;
        self.samples_this_span = 0;
        self.output_samples_this_span = 0;
        self.inner_span_len = None;

        // Rodio sources resume a seek on the channel they were positioned at, so pad the discarded
        // leading samples of a partial input frame to keep the channels aligned.
        for _ in 0..self.input_frame_offset {
            self.resampler
                .write_samples(&[T::zero()])
                .unwrap_or_else(|err| panic_err("failed to write sample", err));
        }

        // Finish an output frame that is already underway with silence so downstream channels stay aligned.
        self.output_frame_is_startup_silence = self.output_frame_samples_remaining > 0;

        if self.config.rodio_fast_start {
            self.fast_start();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::Source;
    use std::num::NonZero;
    use std::time::Duration;

    struct TestSpan {
        sample_rate: u32,
        channels: u16,
        samples: Vec<rodio::Sample>,
    }

    struct ExplicitSpanSource {
        spans: Vec<TestSpan>,
        span_index: usize,
        sample_index: usize,
    }

    impl ExplicitSpanSource {
        fn new(spans: Vec<TestSpan>) -> Self {
            assert!(!spans.is_empty(), "test source needs at least one span");
            Self {
                spans,
                span_index: 0,
                sample_index: 0,
            }
        }

        fn active_span_index(&self) -> Option<usize> {
            let mut span_index = self.span_index;
            let mut sample_index = self.sample_index;

            while let Some(span) = self.spans.get(span_index) {
                if sample_index < span.samples.len() {
                    return Some(span_index);
                }
                span_index += 1;
                sample_index = 0;
            }

            None
        }

        fn active_or_last_span(&self) -> &TestSpan {
            let span_index = self
                .active_span_index()
                .unwrap_or_else(|| self.spans.len().saturating_sub(1));
            &self.spans[span_index]
        }
    }

    impl Iterator for ExplicitSpanSource {
        type Item = rodio::Sample;

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(span) = self.spans.get(self.span_index) {
                if self.sample_index < span.samples.len() {
                    let sample = span.samples[self.sample_index];
                    self.sample_index += 1;
                    return Some(sample);
                }

                self.span_index += 1;
                self.sample_index = 0;
            }

            None
        }
    }

    impl Source for ExplicitSpanSource {
        fn current_span_len(&self) -> Option<usize> {
            let Some(span_index) = self.active_span_index() else {
                return Some(0);
            };
            Some(self.spans[span_index].samples.len())
        }

        fn channels(&self) -> NonZero<u16> {
            NonZero::new(self.active_or_last_span().channels).expect("test span channel count is non-zero")
        }

        fn sample_rate(&self) -> NonZero<u32> {
            NonZero::new(self.active_or_last_span().sample_rate).expect("test span sample rate is non-zero")
        }

        fn total_duration(&self) -> Option<Duration> {
            None
        }
    }

    fn test_config(input_sample_rate: usize, channels: usize) -> Config {
        Config {
            input_sample_rate,
            output_sample_rate: 48_000,
            channels,
            quality: 64,
            bandwidth: 0.95,
            ..Config::default()
        }
    }

    fn test_span(sample_rate: u32, channels: u16, frames: usize, phase: f32) -> TestSpan {
        let channels_usize = usize::from(channels);
        let sample_count = frames * channels_usize;
        let samples = (0..sample_count)
            .map(|sample| ((sample as f32 * 0.013) + phase).sin() * 0.25)
            .collect();

        TestSpan {
            sample_rate,
            channels,
            samples,
        }
    }

    fn consume_samples<S, T>(resampler: &mut RodioResampler<S, T>, samples: usize)
    where
        S: Source,
        T: Float + FftNum,
    {
        for _ in 0..samples {
            assert!(
                resampler.next().is_some(),
                "reported current_span_len exceeded the remaining stream"
            );
        }
    }

    #[test]
    fn startup_silence_ends_on_an_output_frame_boundary() {
        let source = ExplicitSpanSource::new(vec![test_span(44_100, 2, 1024, 0.0)]);
        let mut resampler = RodioResampler::new(source, test_config(44_100, 2)).expect("resampler should construct");

        let mut observed_silence = false;
        let mut observed_audio = false;
        while let Some(left) = resampler.next() {
            let right = resampler
                .next()
                .expect("a stereo source must not end halfway through an output frame");
            let left_is_silence = left == 0.0 && left.is_sign_negative();
            let right_is_silence = right == 0.0 && right.is_sign_negative();

            assert_eq!(
                left_is_silence, right_is_silence,
                "startup silence must not transition to resampled audio halfway through a frame"
            );
            observed_silence |= left_is_silence;
            observed_audio |= !left_is_silence;
        }

        assert!(observed_silence, "test should observe startup silence");
        assert!(observed_audio, "test should observe resampled audio");
    }

    #[test]
    fn current_span_len_is_frame_aligned_for_non_integer_resample_boundary() {
        let source = ExplicitSpanSource::new(vec![test_span(44_100, 2, 512, 0.0), test_span(32_000, 2, 512, 0.7)]);
        let mut resampler = RodioResampler::new(source, test_config(44_100, 2)).expect("resampler should construct");

        let mut observed_samples = 0usize;
        const MAX_OUTPUT_SAMPLES: usize = 20_000;
        loop {
            let span_len = resampler
                .current_span_len()
                .expect("finite explicit spans should report finite output spans");
            if span_len == 0 {
                assert!(
                    resampler.next().is_none(),
                    "Some(0) should only be reported at end-of-stream"
                );
                break;
            }

            let channels = usize::from(resampler.channels().get());
            assert_eq!(
                span_len % channels,
                0,
                "current_span_len must stay aligned to complete output frames"
            );

            consume_samples(&mut resampler, span_len);
            observed_samples += span_len;
            assert!(
                observed_samples <= MAX_OUTPUT_SAMPLES,
                "resampler did not drain the finite span source"
            );
        }

        assert!(observed_samples > 0, "finite spans should produce output");
    }

    #[test]
    fn current_span_len_exposes_boundary_before_output_format_change() {
        let source = ExplicitSpanSource::new(vec![test_span(44_100, 1, 512, 0.0), test_span(44_100, 2, 512, 0.7)]);
        let mut resampler = RodioResampler::new(source, test_config(44_100, 1)).expect("resampler should construct");

        let mut observed_channel_change = false;
        let mut previous_channels = usize::from(resampler.channels().get());
        let mut observed_samples = 0usize;
        const MAX_OUTPUT_SAMPLES: usize = 20_000;

        loop {
            let span_len = resampler
                .current_span_len()
                .expect("finite explicit spans should report finite output spans");
            if span_len == 0 {
                break;
            }

            let chunk_channels = usize::from(resampler.channels().get());
            for sample_in_chunk in 0..span_len {
                assert_eq!(
                    usize::from(resampler.channels().get()),
                    chunk_channels,
                    "output channels changed inside a reported stable span at sample {sample_in_chunk} of {span_len}"
                );
                assert!(
                    resampler.next().is_some(),
                    "reported current_span_len exceeded the remaining stream"
                );
            }

            let next_channels = usize::from(resampler.channels().get());
            if next_channels != previous_channels {
                assert_eq!(
                    next_channels, 2,
                    "test source should only transition from mono to stereo"
                );
                observed_channel_change = true;
            }
            previous_channels = next_channels;
            observed_samples += span_len;
            assert!(
                observed_samples <= MAX_OUTPUT_SAMPLES,
                "resampler did not drain the finite span source"
            );
        }

        assert!(
            observed_channel_change,
            "resampler should expose the queued stereo output span"
        );
    }

    #[test]
    fn current_span_len_handles_exact_boundary_without_zero_stall() {
        let source = ExplicitSpanSource::new(vec![test_span(44_100, 1, 512, 0.0), test_span(44_100, 2, 512, 0.7)]);
        let mut resampler = RodioResampler::new(source, test_config(44_100, 1)).expect("resampler should construct");

        let mut previous_channels = usize::from(resampler.channels().get());
        let mut observed_boundary = false;
        let mut observed_samples = 0usize;
        const MAX_OUTPUT_SAMPLES: usize = 20_000;

        loop {
            let span_len = resampler
                .current_span_len()
                .expect("finite explicit spans should report finite output spans");
            let channels = usize::from(resampler.channels().get());

            if channels != previous_channels {
                assert_eq!(channels, 2, "test source should only transition from mono to stereo");
                assert_eq!(
                    span_len, channels,
                    "exact output span boundaries should report one stable frame, not zero or a partial frame"
                );
                observed_boundary = true;
            }

            if span_len == 0 {
                break;
            }

            consume_samples(&mut resampler, span_len);
            previous_channels = channels;
            observed_samples += span_len;
            assert!(
                observed_samples <= MAX_OUTPUT_SAMPLES,
                "resampler did not drain the finite span source"
            );
        }

        assert!(
            observed_boundary,
            "test should observe the exact mono-to-stereo boundary"
        );
        assert!(
            observed_samples > 0,
            "resampler should continue producing output after the boundary"
        );
    }

    #[test]
    fn no_underrun_across_delayed_span_transition() {
        let first_span = rodio::source::SignalGenerator::new(
            NonZero::new(44_100).expect("constant non-zero sample rate"),
            440.0,
            rodio::source::Function::Sine,
        )
        .take_duration(Duration::from_secs(2));

        let second_span = rodio::source::SignalGenerator::new(
            NonZero::new(48_000).expect("constant non-zero sample rate"),
            660.0,
            rodio::source::Function::Sine,
        )
        .take_duration(Duration::from_secs(2));

        let source = rodio::source::from_iter([first_span, second_span]);
        let config = Config {
            input_sample_rate: 44_100,
            output_sample_rate: 48_000,
            channels: 1,
            ..Config::default()
        };

        let mut resampler = RodioResampler::new(source, config).expect("resampler should construct");

        let mut output_samples = 0usize;
        const MAX_OUTPUT_SAMPLES: usize = 1_000_000;
        while let Some(sample) = resampler.next() {
            assert!(!sample.is_nan(), "resampler output should be finite");
            output_samples += 1;
            assert!(
                output_samples <= MAX_OUTPUT_SAMPLES,
                "resampler did not drain after delayed span transition"
            );
        }

        assert!(
            output_samples > 0,
            "resampler should produce output for finite two-span input"
        );
    }

    /// One run of fixed-size packets sharing a sample rate and channel count.
    struct PacketSegment {
        sample_rate: u32,
        channels: u16,
        packet_frames: usize,
        packets: usize,
    }

    /// Mimics rodio's symphonia decoder: each decoded packet is its own span, the next packet is only
    /// decoded inside `next()`, and `current_span_len()`/format describe the packet most recently
    /// decoded. Channel `c` carries DC at `level * (1 - 2 * (c % 2))`, so swapped channels are visible.
    /// Seeking flips the sign of `level` and forces a fresh packet decode.
    struct PacketSource {
        segments: Vec<PacketSegment>,
        segment_index: usize,
        packets_left: usize,
        packet_len: usize,
        packet_offset: usize,
        packet_sample_rate: u32,
        packet_channels: u16,
        level: f32,
        unbounded_span: bool,
        samples_pulled: u64,
    }

    impl PacketSource {
        fn new(segments: Vec<PacketSegment>) -> Self {
            let mut source = Self {
                segments,
                segment_index: 0,
                packets_left: 0,
                packet_len: 0,
                packet_offset: 0,
                packet_sample_rate: 0,
                packet_channels: 0,
                level: 0.5,
                unbounded_span: false,
                samples_pulled: 0,
            };
            source.packets_left = source.segments[0].packets;
            assert!(source.decode_packet(), "test source needs at least one packet");
            source
        }

        fn decode_packet(&mut self) -> bool {
            while self.packets_left == 0 {
                self.segment_index += 1;
                let Some(segment) = self.segments.get(self.segment_index) else {
                    return false;
                };
                self.packets_left = segment.packets;
            }
            let segment = &self.segments[self.segment_index];
            self.packets_left -= 1;
            self.packet_sample_rate = segment.sample_rate;
            self.packet_channels = segment.channels;
            self.packet_len = segment.packet_frames * usize::from(segment.channels);
            self.packet_offset = 0;
            true
        }
    }

    impl Iterator for PacketSource {
        type Item = rodio::Sample;

        fn next(&mut self) -> Option<Self::Item> {
            if self.packet_offset >= self.packet_len && !self.decode_packet() {
                return None;
            }
            let channel = self.packet_offset % usize::from(self.packet_channels);
            self.packet_offset += 1;
            self.samples_pulled += 1;
            Some(if channel.is_multiple_of(2) {
                self.level
            } else {
                -self.level
            })
        }
    }

    impl Source for PacketSource {
        fn current_span_len(&self) -> Option<usize> {
            (!self.unbounded_span).then_some(self.packet_len)
        }

        fn channels(&self) -> NonZero<u16> {
            NonZero::new(self.packet_channels).expect("test packet channel count is non-zero")
        }

        fn sample_rate(&self) -> NonZero<u32> {
            NonZero::new(self.packet_sample_rate).expect("test packet sample rate is non-zero")
        }

        fn total_duration(&self) -> Option<Duration> {
            None
        }

        fn try_seek(&mut self, _: Duration) -> Result<(), rodio::source::SeekError> {
            self.level = -self.level;
            self.packet_offset = usize::MAX;
            Ok(())
        }
    }

    fn endless_stereo_packets(sample_rate: u32) -> PacketSource {
        PacketSource::new(vec![PacketSegment {
            sample_rate,
            channels: 2,
            packet_frames: 1152,
            packets: usize::MAX,
        }])
    }

    /// Plays `output_frames` frames and returns how many input frames were consumed beyond the ideal
    /// `output_frames * input_rate / output_rate`.
    fn excess_input_frames<T: Float + FftNum>(
        resampler: &mut RodioResampler<PacketSource, T>,
        output_frames: usize,
    ) -> i64 {
        consume_samples(resampler, output_frames * 2);
        let ratio = resampler.span_ratio;
        let input_frames = resampler.inner.samples_pulled as i64 / 2;
        input_frames - (output_frames as f64 * ratio).round() as i64
    }

    #[test]
    fn packetized_upsampling_consumes_input_at_the_resampling_ratio() {
        let mut resampler = RodioResampler::new(endless_stereo_packets(44_100), test_config(44_100, 2))
            .expect("resampler should construct");
        let lead = resampler.resampler.estimate_priming_samples() as i64 / 2;

        let excess = excess_input_frames(&mut resampler, 48_000 * 5);
        assert!(
            excess <= lead + 1152,
            "consumed {excess} input frames beyond the resampling ratio; expected at most the priming lead ({lead}) plus one packet"
        );
    }

    #[test]
    fn source_without_span_len_is_supported() {
        let mut source = endless_stereo_packets(44_100);
        source.unbounded_span = true;
        let mut resampler = RodioResampler::new(source, test_config(44_100, 2)).expect("resampler should construct");
        let lead = resampler.resampler.estimate_priming_samples() as i64 / 2;

        let excess = excess_input_frames(&mut resampler, 48_000);
        assert!(
            excess <= lead + 1,
            "consumed {excess} input frames beyond the resampling ratio"
        );
    }

    #[test]
    fn lazily_decoded_format_change_keeps_frames_in_their_own_span() {
        let segment = |sample_rate, channels| PacketSegment {
            sample_rate,
            channels,
            packet_frames: 1152,
            packets: 8,
        };
        let source = PacketSource::new(vec![segment(44_100, 2), segment(48_000, 1), segment(32_000, 2)]);
        let mut resampler = RodioResampler::new(source, test_config(44_100, 2)).expect("resampler should construct");

        let mut stereo_frames = 0usize;
        let mut mono_frames = 0usize;
        let mut misaligned_frames = 0usize;
        while resampler.current_span_len().is_some_and(|len| len > 0) {
            let channels = usize::from(resampler.channels().get());
            let frame: Vec<f32> = (0..channels)
                .map(|_| resampler.next().expect("output should not end mid-frame"))
                .collect();
            if frame.iter().all(|&sample| sample == 0.0) {
                continue;
            }
            if channels == 2 {
                stereo_frames += 1;
                misaligned_frames += usize::from(frame[0] < -0.1 || frame[1] > 0.1);
            } else {
                mono_frames += 1;
                misaligned_frames += usize::from(frame[0] < -0.1);
            }
        }

        assert!(
            stereo_frames > 0 && mono_frames > 0,
            "test should observe both stereo and mono output"
        );
        assert_eq!(misaligned_frames, 0, "channels were misaligned across a span boundary");
    }

    fn assert_seek_discards_buffered_audio(fast_start: bool) {
        let config = test_config(44_100, 2).with_rodio_fast_start(fast_start);
        let mut resampler =
            RodioResampler::new(endless_stereo_packets(44_100), config).expect("resampler should construct");
        consume_samples(&mut resampler, 48_000 * 2);

        resampler.try_seek(Duration::from_secs(1)).expect("test source seeks");

        let mut first_new_frame = None;
        for frame in 0..48_000 {
            let left = resampler.next().expect("endless source");
            let right = resampler.next().expect("endless source");
            assert!(
                left <= 0.1 && right >= -0.1,
                "pre-seek audio played {frame} frames after the seek (fast_start: {fast_start})"
            );
            if first_new_frame.is_none() && left < -0.25 && right > 0.25 {
                first_new_frame = Some(frame);
            }
        }

        let first_new_frame = first_new_frame.expect("post-seek audio should play");
        if fast_start {
            assert!(
                first_new_frame < 16,
                "fast-start should re-prime on seek, but post-seek audio started at frame {first_new_frame}"
            );
        }
    }

    #[test]
    fn seek_discards_buffered_audio() {
        assert_seek_discards_buffered_audio(false);
    }

    #[test]
    fn seek_with_fast_start_discards_buffered_audio_and_reprimes() {
        assert_seek_discards_buffered_audio(true);
    }
}
