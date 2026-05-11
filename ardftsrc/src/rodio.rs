use crate::{Config, RealtimeResampler, Error};
use num_traits::Float;
use realfft::FftNum;

pub struct RodioResampler<S: rodio::Source, T = f64>
where
    T: Float + FftNum,
{
    inner: S,
    resampler: RealtimeResampler<T>,
    fast_start: bool,
    fast_start_finished: bool,
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
        let resampler = RealtimeResampler::new(config)?;
        let span_format_in = resampler.span_format_in();
        let span_format_out = resampler.span_format_out();
        let span_ratio = span_format_in.sample_rate as f64 / span_format_out.sample_rate as f64;

        Ok(Self {
            inner,
            resampler,
            fast_start,
            fast_start_finished: false,
            stream_input_ended: false,
            just_seeked: false,
            samples_this_span: 0,
            output_samples_this_span: 0,
            span_ratio,
        })
    }

    #[inline]
    pub fn set_span_ratio(&mut self) {
        let span_format_in = self.resampler.span_format_in();
        let span_format_out = self.resampler.span_format_out();
        self.span_ratio = span_format_in.sample_rate as f64 / span_format_out.sample_rate as f64;
    }

    pub fn new_span(&mut self) {
        self.resampler.new_span(
            self.inner.sample_rate().get() as usize,
            self.inner.channels().get() as usize,
        );
        self.samples_this_span = 0;
        self.output_samples_this_span = 0;
        self.set_span_ratio();
    }

    #[inline]
    fn is_underrun(&self, sample: Option<T>) -> bool {
        match sample {
            Some(sample) => sample.is_zero() && sample.is_sign_negative(),
            None => false,
        }
    }

    /// Estimate the number of input samples to pull this tick.
    #[inline]
    fn calculate_inner_pulls(&mut self) -> u64 {
        // If the inner stream is ended, return zero.
        if self.stream_input_ended {
            return 0;
        }

        // If we are in fast start mode, try to prime the resampler with initial samples to get it up to speed.
        if self.fast_start && !self.fast_start_finished {
            return self.resampler.initial_sample_delay() as u64;
        }

        // Otherwise, calculate the number of input samples to pull to keep output production approximately aligned with input consumption given the current span ratio.
        self.output_samples_this_span = self.output_samples_this_span.saturating_add(1);
        let target_input_samples = (self.output_samples_this_span as f64 * self.span_ratio).ceil() as u64;
        let inner_pulls = target_input_samples.saturating_sub(self.samples_this_span);

        inner_pulls
    }

    fn next_sample(&mut self) -> Option<T> {
        // If we just seeked, we may already be in a new span.
        if self.just_seeked {
            let span_format_in = self.resampler.span_format_in();
            if self.inner.sample_rate().get() != span_format_in.sample_rate
                || self.inner.channels().get() != span_format_in.channels as u16
            {
                self.new_span();
            }
            self.just_seeked = false;
        }

        // Keep input consumption approximately aligned with output production:
        // pull 0 or multiple input samples depending on span_ratio and current drift.
        let inner_pulls = self.calculate_inner_pulls();

        for _ in 0..inner_pulls {
            // Check for a new span on each pull since downsampling can consume >1 input sample per output.
            let new_span_after_next = self.inner.current_span_len() == Some(1);

            // If input is none, end the stream, but keep reading until the resampler is drained.
            match self.inner.next() {
                Some(sample) => {
                    self.resampler.write_sample(num_traits::cast(sample).unwrap());
                    self.samples_this_span += 1;
                }
                None => {
                    if !self.stream_input_ended {
                        self.stream_input_ended = true;
                        self.resampler.finalize();
                    }
                }
            }

            if new_span_after_next {
                self.new_span();
            }
        }

        // Read the sample
        let mut output_sample = self.resampler.read_sample();

        // If we are in fast start mode, read output until we stop getting underruns.
        while self.fast_start && !self.fast_start_finished && self.is_underrun(output_sample) {
            output_sample = self.resampler.read_sample();
        }
        if self.fast_start && !self.fast_start_finished {
            self.fast_start_finished = true;
            self.samples_this_span = 0;
            self.output_samples_this_span = 0;
        }

        output_sample
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
            if self.fast_start {
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

        // If we have fast_start enabled, reset the resampler to clear any pending samples.
        // Then reset the fast start flag so we pre-load inner samples into the resampler.
        if false {
            if self.resampler.reset(false).is_ok() {
                self.fast_start_finished = false;
            }
        }

        self.just_seeked = true;
        Ok(())
    }
}
