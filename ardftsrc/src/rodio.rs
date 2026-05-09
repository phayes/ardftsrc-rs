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
}

impl<S: rodio::Source, T: Float + FftNum> RodioResampler<S, T> {
    pub fn new(inner: S, resampler: RealtimeResampler<T>) -> Self {
        Self {
            inner,
            resampler,
            stream_input_ended: false,
            just_seeked: false,
        }
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
            self.resampler.new_span(
                self.inner.sample_rate().get() as usize,
                self.inner.channels().get() as usize,
            );
            self.just_seeked = false;
        }

        // Check for a new span
        let new_span_after_next = self.inner.current_span_len() == Some(1);

        // If input is none, end the stream, but keep reading until the resampler is drained.
        match self.inner.next() {
            Some(sample) => {
                self.resampler.write_sample(sample);
            }
            None => {
                if !self.stream_input_ended {
                    self.stream_input_ended = true;
                    self.resampler.finalize();
                }
            }
        }

        if new_span_after_next {
            self.resampler.new_span(
                self.inner.sample_rate().get() as usize,
                self.inner.channels().get() as usize,
            );
        }

        self.resampler.read_sample()
    }
}

impl<S: rodio::Source> Iterator for RodioResampler<S, f64>
where
    S::Item: Float + FftNum + Send + 'static,
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // If we just seeked, we may already be in a new span.
        if self.just_seeked {
            self.resampler.new_span(
                self.inner.sample_rate().get() as usize,
                self.inner.channels().get() as usize,
            );
            self.just_seeked = false;
        }

        // Check for a new span
        let new_span_after_next = self.inner.current_span_len() == Some(1);

        // If input is none, end the stream, but keep reading until the resampler is drained.
        match self.inner.next() {
            Some(sample) => {
                self.resampler.write_sample(sample as f64);
            }
            None => {
                if !self.stream_input_ended {
                    self.stream_input_ended = true;
                    self.resampler.finalize();
                }
            }
        }

        if new_span_after_next {
            self.resampler.new_span(
                self.inner.sample_rate().get() as usize,
                self.inner.channels().get() as usize,
            );
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

    fn try_seek(&self, time: core::time::Duration) -> Result<(), rodio::SeekError> {
        self.inner.seek(time)?;
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

    fn try_seek(&self, time: core::time::Duration) -> Result<(), rodio::SeekError> {
        self.inner.seek(time)?;
        self.just_seeked = true;
        Ok(())
    }
}
