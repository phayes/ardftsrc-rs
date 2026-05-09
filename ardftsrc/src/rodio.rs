use crate::RealtimeResampler;
use num_traits::Float;
use realfft::FftNum;

pub struct RodioResampler<S: rodio::Source, T = f64>
where
    T: Float + FftNum,
{
    inner: S,
    resampler: RealtimeResampler<T>,
    stream_input_ended: bool,
    just_seeked: bool,
    samples_this_span: u64,
    output_samples_this_span: u64,
    span_ratio: f64,
}

impl<S: rodio::Source, T: Float + FftNum> RodioResampler<S, T> {
    pub fn new(inner: S, resampler: RealtimeResampler<T>) -> Self {
        let span_format_in = resampler.span_format_in();
        let span_format_out = resampler.span_format_out();
        let span_ratio = span_format_in.sample_rate as f64 / span_format_out.sample_rate as f64;

        Self {
            inner,
            resampler,
            stream_input_ended: false,
            just_seeked: false,
            samples_this_span: 0,
            output_samples_this_span: 0,
            span_ratio,
        }
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

    /// Estimate the number of input samples to pull to keep output production approximately aligned with input consumption given the current span ratio.
    #[inline]
    fn calculate_inner_pulls(&mut self) -> u64 {
        self.output_samples_this_span = self.output_samples_this_span.saturating_add(1);
        let target_input_samples = (self.output_samples_this_span as f64 * self.span_ratio).round() as u64;
        target_input_samples.saturating_sub(self.samples_this_span)
    }
}

impl<S: rodio::Source> Iterator for RodioResampler<S, f32>
where
    S::Item: Float + FftNum + Send + 'static,
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
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
                    self.resampler.write_sample(sample);
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

        self.resampler.read_sample()
    }
}

impl<S: rodio::Source> Iterator for RodioResampler<S, f64>
where
    S::Item: Float + FftNum + Send + 'static + Into<f64>,
{
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
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
                    self.resampler.write_sample(sample.into());
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

        self.resampler.read_sample().map(|sample| sample as f32)
    }
}

impl<S: rodio::Source> rodio::Source for RodioResampler<S, f64>
where
    S::Item: Float + FftNum + Send + 'static + Into<f64>,
{
    fn sample_rate(&self) -> std::num::NonZero<u32> {
        std::num::NonZero::new(self.resampler.span_format_out.sample_rate as u32).unwrap()
    }

    fn channels(&self) -> std::num::NonZero<u16> {
        std::num::NonZero::new(self.resampler.span_format_out.channels as u16).unwrap()
    }

    fn total_duration(&self) -> Option<core::time::Duration> {
        self.inner.total_duration()
    }

    fn current_span_len(&self) -> Option<usize> {
        self.resampler.current_span_len()
    }

    fn try_seek(&mut self, time: core::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(time)?;
        self.just_seeked = true;
        Ok(())
    }
}

impl<S: rodio::Source> rodio::Source for RodioResampler<S, f32>
where
    S::Item: Float + FftNum + Send + 'static + Into<f32>,
{
    fn sample_rate(&self) -> std::num::NonZero<u32> {
        std::num::NonZero::new(self.resampler.span_format_out.sample_rate as u32).unwrap()
    }

    fn channels(&self) -> std::num::NonZero<u16> {
        std::num::NonZero::new(self.resampler.span_format_out.channels as u16).unwrap()
    }

    fn total_duration(&self) -> Option<core::time::Duration> {
        self.inner.total_duration()
    }

    fn current_span_len(&self) -> Option<usize> {
        self.resampler.current_span_len()
    }

    fn try_seek(&mut self, time: core::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(time)?;
        self.just_seeked = true;
        Ok(())
    }
}
