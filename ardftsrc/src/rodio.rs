use num_traits::Float;
use realfft::FftNum;
use crate::StreamingResampler;

pub struct RodioResampler<S: rodio::Source, T = f64>
where
    T: Float + FftNum,
{
    inner: S,
    resampler: StreamingResampler<T>,
}

impl<S: rodio::Source, T: Float + FftNum> RodioResampler<S, T> {
    pub fn new(inner: S, resampler: StreamingResampler<T>) -> Self {
        Self {
            inner,
            resampler,
        }
    }
}

impl<S: rodio::Source> Iterator for RodioResampler<S, f32>
where
    S::Item: Float + FftNum + Send + 'static
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {

        // TODO: Span handling
        let input_sample = self.inner.next()?;
        self.resampler.write_sample(input_sample);

        let output_sample = self.resampler.read_sample()?;

        Some(output_sample)
    }
}

impl<S: rodio::Source> Iterator for RodioResampler<S, f64>
where
    S::Item: Float + FftNum + Send + 'static
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // TODO: Span handling
        let sample = self.inner.next()?;

        self.resampler.write_sample(sample as f64);

        self.resampler.read_sample().map(|sample| sample as f32)
    }
}

impl<S: rodio::Source> rodio::Source for RodioResampler<S, f64>
where
    S::Item: Float + FftNum + Send + 'static + From<f64>,
{
    fn sample_rate(&self) -> std::num::NonZero<u32> {
        std::num::NonZero::new(self.resampler.span_format_out.sample_rate as u32).unwrap()
    }

    fn channels(&self) -> std::num::NonZero<u16> {
        std::num::NonZero::new(self.resampler.span_format_out.channels as u16).unwrap()
    }

    // TODO: Implement this.
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    // TODO: Implement this.
    fn total_duration(&self) -> Option<core::time::Duration> {
        None
    }
}

impl<S: rodio::Source> rodio::Source for RodioResampler<S, f32>
where
    S::Item: Float + FftNum + Send + 'static + From<f32>,
{
    fn sample_rate(&self) -> std::num::NonZero<u32> {
        std::num::NonZero::new(self.resampler.span_format_out.sample_rate as u32).unwrap()
    }

    fn channels(&self) -> std::num::NonZero<u16> {
        std::num::NonZero::new(self.resampler.span_format_out.channels as u16).unwrap()
    }

    // TODO: Implement this.
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    // TODO: Implement this.
    fn total_duration(&self) -> Option<core::time::Duration> {
        None
    }
}
