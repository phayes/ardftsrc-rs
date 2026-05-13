use crate::{Config, Error, RealtimeResampler};
use num_traits::Float;
use realfft::FftNum;

/// Wrap a `rodio::Source` and resample it in realtime in your rodio pipeline. Requires the `rodio` feature.
///
/// When playing from a buffered audio source such as a file or a buffered stream, it is recommended to use `config.with_rodio_fast_start(true)`, which will
/// avoid initial output delay by pulling samples from the upstream source to prime the resampler. For very-realtime sources such as microphones or similar,
/// do not enable fast-start.
///
/// Additional configuration settings are `Config::with_realtime_input_range()` and `Config::with_realtime_max_channels()` which lets you tune the resampler if you
/// know the shape of the upstream sample-rate and channel counts. It is not recommended to change these settings - the default values are quite generous.
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
    samples_this_span: u64,
    output_samples_this_span: u64,
    span_ratio: f64,
}

impl<S, T> RodioResampler<S, T>
where
    S: rodio::Source,
    T: Float + FftNum,
{
    fn new_typed(inner: S, config: Config) -> Result<Self, Error> {
        let fast_start = config.rodio_fast_start;
        let resampler = RealtimeResampler::new(config.clone())?;
        let span_format_in = resampler.span_format_in();
        let span_format_out = resampler.span_format_out();
        let span_ratio = span_format_in.sample_rate as f64 / span_format_out.sample_rate as f64;

        let mut rodio_resampler = Self {
            inner,
            resampler,
            config,
            stream_input_ended: false,
            just_seeked: false,
            samples_this_span: 0,
            output_samples_this_span: 0,
            span_ratio,
        };

        if fast_start {
            rodio_resampler.fast_start();
        }

        Ok(rodio_resampler)
    }

    fn set_span_ratio(&mut self) {
        let span_format_in = self.resampler.span_format_in();
        let span_format_out = self.resampler.span_format_out();
        self.span_ratio = span_format_in.sample_rate as f64 / span_format_out.sample_rate as f64;
    }

    fn maybe_new_span(&mut self) {
        let span_format_in = self.resampler.span_format_in();
        if self.inner.sample_rate().get() != span_format_in.sample_rate
            || self.inner.channels().get() != span_format_in.channels as u16
        {
            self.resampler.new_span(
                self.inner.sample_rate().get() as usize,
                self.inner.channels().get() as usize,
            );
            self.samples_this_span = 0;
            self.output_samples_this_span = 0;
            self.set_span_ratio();
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
        let inner_pulls = target_input_samples.saturating_sub(self.samples_this_span);

        inner_pulls
    }

    fn pull_inner_sample(&mut self) {
        // Check for a new span on each pull since downsampling can consume >1 input sample per output.
        let new_span_after_next = self.inner.current_span_len() == Some(1);

        // If input is none, end the stream, but keep reading until the resampler is drained.
        match self.inner.next() {
            Some(sample) => {
                if self.resampler.write_sample(num_traits::cast(sample).unwrap()) {
                    self.samples_this_span += 1;
                }
            }
            None => {
                if !self.stream_input_ended {
                    self.stream_input_ended = true;
                    self.resampler.finalize();
                }
            }
        }

        if new_span_after_next {
            self.maybe_new_span();
        }
    }

    fn fast_start(&mut self) {
        let num_in_samples = self.resampler.initial_input_sample_delay() as u64;

        // Write the initial samples to the resampler.
        for _ in 0..num_in_samples {
            if self.stream_input_ended {
                break;
            }
            self.pull_inner_sample();
        }

        // Treat the pre-pulled input as intentional lead so normal pacing preserves it.
        self.output_samples_this_span = (self.samples_this_span as f64 / self.span_ratio).ceil() as u64;

        // Wait until the worker has produced output without consuming the startup backlog.
        let mut spins: u32 = 0;
        loop {
            if let Some(sample) = self.resampler.peek_sample() {
                if !RealtimeResampler::sample_is_underrun(sample) {
                    return;
                }
            }

            if self.stream_input_ended {
                return;
            }

            spins += 1;
            if spins > 100 {
                std::thread::yield_now();
                spins = 0;
            }
            std::hint::spin_loop();
        }
    }

    #[inline]
    fn next_sample(&mut self) -> Option<T> {
        // If we just seeked, we may already be in a new span.
        if self.just_seeked {
            self.maybe_new_span();
            self.just_seeked = false;
        }

        // Keep input consumption approximately aligned with output production:
        // pull 0 or multiple input samples depending on span_ratio and current drift.
        let inner_pulls = self.calculate_inner_pulls();

        for _ in 0..inner_pulls {
            self.pull_inner_sample();
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
    pub fn new(inner: S, config: Config) -> Result<Self, Error> {
        Self::new_typed(inner, config)
    }
}

impl<S> RodioResampler<S, f32>
where
    S: rodio::Source,
{
    /// Create a new RodioResampler using `f32` as the internal resampling type.
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
        self.next_sample()
            .map(|sample| num_traits::cast(sample).expect("resampler sample should be representable as rodio sample"))
    }
}

impl<S, T> rodio::Source for RodioResampler<S, T>
where
    S: rodio::Source,
    T: Float + FftNum,
{
    fn sample_rate(&self) -> std::num::NonZero<u32> {
        std::num::NonZero::new(self.resampler.span_format_out.sample_rate as u32).unwrap()
    }

    fn channels(&self) -> std::num::NonZero<u16> {
        std::num::NonZero::new(self.resampler.span_format_out.channels as u16).unwrap()
    }

    fn total_duration(&self) -> Option<core::time::Duration> {
        self.inner.total_duration().map(|inner_duration| {
            if self.config.rodio_fast_start {
                inner_duration
            } else {
                inner_duration + self.resampler.initial_sample_delay_duration()
            }
        })
    }

    fn current_span_len(&self) -> Option<usize> {
        self.resampler.current_span_len()
    }

    fn try_seek(&mut self, time: core::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(time)?;
        self.stream_input_ended = false;
        self.just_seeked = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PRESET_GOOD;
    use std::num::NonZero;

    #[test]
    fn fast_start_preserves_pre_pulled_input_as_pacing_lead() {
        let tone = rodio::source::SignalGenerator::new(
            NonZero::new(44_100).expect("constant non-zero sample rate"),
            440.0,
            rodio::source::Function::Sine,
        );

        let config = PRESET_GOOD
            .with_channels(2)
            .with_input_rate(44_100)
            .with_output_rate(96_000)
            .with_rodio_fast_start(true);

        let mut resampler = RodioResampler::new(tone, config).expect("resampler should construct");

        assert_eq!(resampler.samples_this_span, 12_348);
        assert_eq!(resampler.output_samples_this_span, 26_880);
        assert_eq!(resampler.calculate_inner_pulls(), 1);
    }
}
