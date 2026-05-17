use crate::SamplesLeftInSpan;
use crate::{panic_err, panic_msg, Config, Error, RealtimeResampler};
use num_traits::Float;
use realfft::FftNum;
/// Wrap a [`rodio::Source`] and resample it in realtime in your rodio pipeline. Requires the `rodio` feature.
///
/// When playing from a buffered audio source such as a file or a buffered stream, it is recommended to use [`config.with_rodio_fast_start(true)`](Config::with_rodio_fast_start), which will
/// avoid initial output delay by pulling samples from the upstream source to prime the resampler. For very-realtime sources such as microphones or similar,
/// do not enable fast-start.
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
    just_seeked: bool,
    pending_span_transition: bool,
    samples_this_span: u64,
    output_samples_this_span: u64,
    span_ratio: f64,
    inner_span_len: u64,
    inner_channel_count: u64,
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
            just_seeked: false,
            pending_span_transition: false,
            samples_this_span: 0,
            output_samples_this_span: 0,
            span_ratio,
            inner_span_len: 0,      // Zero means uninitialized here
            inner_channel_count: 0, // Zero means uninitialized here
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
            self.inner_span_len = 0; // Zero means uninitialized here
            self.inner_channel_count = 0; // Zero means uninitialized here
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
        let mut inner_pulls = target_input_samples.saturating_sub(self.samples_this_span);

        // Make sure span boundaries get at least one pull
        if self.pending_span_transition && inner_pulls == 0 {
            inner_pulls = 1;
        }

        inner_pulls
    }

    // Pull a sample from the inner source and write it to the resampler.
    fn pull_inner_sample(&mut self, count_samples: bool) {
        // Cache the inner span length for span boundary checking
        if self.inner_span_len == 0 {
            self.inner_span_len = self.inner.current_span_len().unwrap() as u64;
            self.inner_channel_count = self.inner.channels().get() as u64;

            // Debug assert that we are right on a frame boundary
            debug_assert!(self.inner_span_len % self.inner_channel_count == 0, "ardftsrc: Error in inner source: current_span_len should be a multiple of channels on a frame boundary");
        }

        // If input is none, end the stream, but keep reading until the resampler is drained.
        match self.inner.next() {
            Some(sample) => {
                self.resampler
                    .write_samples(&[num_traits::cast(sample).unwrap()])
                    .unwrap_or_else(|err| panic_err("failed to write sample", err));
                if count_samples {
                    self.samples_this_span += 1;
                }
            }
            None => {
                if !self.stream_input_ended {
                    self.stream_input_ended = true;
                    self.resampler
                        .finalize()
                        .unwrap_or_else(|err| panic_err("failed to finalize resampler", err));
                }
            }
        }

        if self.samples_this_span == self.inner_span_len {
            // Debug assert that we are right on a frame boundary
            debug_assert!(
                self.samples_this_span % self.inner_channel_count == 0,
                "samples_this_span should be a multiple of inner_channel_count on a frame boundary"
            );

            self.pending_span_transition = true;
        }

        // Some sources (for example source-chaining adapters) can switch to a new span one pull
        // after reporting `current_span_len() == Some(1)`. Keep checking after each pull while a
        // transition is pending so pacing can update as soon as the new format is visible.
        if self.pending_span_transition {
            let started_new_span = self.maybe_new_input_span(); // TODO THIS
            if started_new_span || self.stream_input_ended {
                self.pending_span_transition = false;
            }
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
        // If we just seeked, we may already be in a new span.
        if self.just_seeked {
            self.maybe_new_input_span();
            self.just_seeked = false;
        }

        // Keep input consumption approximately aligned with output production:
        // pull 0 or multiple input samples depending on span_ratio and current drift.
        let inner_pulls = self.calculate_inner_pulls();

        for _ in 0..inner_pulls {
            self.pull_inner_sample(true);
        }

        // Read the sample
        self.resampler.read_sample()
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
        std::num::NonZero::new(self.resampler.output_channels() as u16).unwrap()
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
        let input_span_len = match self.inner.current_span_len() {
            Some(len) => len,
            None => return None,
        };

        let input_sample_rate = self.inner.sample_rate().get();
        let output_sample_rate = self.config.output_sample_rate;

        // Integer upsampling (2x, 3x, etc.) - always exact and frame-aligned
        if output_sample_rate % input_sample_rate as usize == 0 {
            return Some(input_span_len * output_sample_rate / input_sample_rate as usize);
        } else {
            return match self.resampler.samples_left_in_span() {
                SamplesLeftInSpan::Known(samples_left) => {
                    let samples_left = samples_left as usize;

                    // Samples left == 0 means the span is drained and a new span is ready to be read (does NOT mean end-of-stream)
                    // Tell the caller to come back in one frame
                    if samples_left == 0 {
                        Some(self.resampler.output_channels() as usize)
                    } else {
                        Some(samples_left)
                    }
                }
                SamplesLeftInSpan::Unknown => {
                    let num_samples_ready = self.resampler.num_samples_ready();

                    // Samples ready == 0 means the output buffer is empty (does NOT mean end-of-stream)
                    // Tell the caller to come back in one frame
                    if num_samples_ready == 0 {
                        Some(self.resampler.output_channels() as usize)
                    } else {
                        Some(num_samples_ready)
                    }
                }
                SamplesLeftInSpan::EndOfStream => Some(0),
            };
        }
    }

    fn try_seek(&mut self, time: core::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(time)?;
        self.stream_input_ended = false;
        self.just_seeked = true;
        self.pending_span_transition = false;
        self.maybe_new_input_span();
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
            let span = &self.spans[span_index];
            let sample_index = if span_index == self.span_index {
                self.sample_index
            } else {
                0
            };

            Some(span.samples.len() - sample_index)
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
}
