use std::collections::VecDeque;

use num_traits::Float;
use realfft::FftNum;
use rtrb::{Consumer, PopError, Producer, PushError, RingBuffer};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use crate::{Config, Error, offthread::OffThreadStreamingResampler};

/// Runtime audio format metadata for tapped packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanFormat {
    channels: u8,
    sample_rate: u32,
}

/// Packet emitted by the low-level ring buffer.
///
/// Invariants:
/// - The first packet for a stream is always `Format`.
/// - Whenever span format changes, a new `Format` packet is emitted before the first `Sample` in that span.
///
/// This should be sized 128bits - oh for a float-niche!
#[derive(Debug, Clone, PartialEq)]
pub enum Packet<T: Float> {
    /// Format packet announces the current `{channels, sample_rate_hz}`.
    Format(SpanFormat),
    /// A single sample (usually f64)
    Sample(T),
}

pub struct StreamingResampler<T = f64>
where
    T: Float + FftNum,
{
    thread_handle: Option<JoinHandle<Result<(), Error>>>,

    // The shape of the input span, when this changes we emit a "SpanChanged"
    span_format_in: SpanFormat,

    // The shape of the output span, we update this when we receive a "SpanChanged" even
    span_format_out: SpanFormat,

    in_producer: Producer<Packet<T>>,

    out_consumer: Consumer<Packet<T>>,

    // If we can't write to the input ring buffer, we need to ditch `span_format_in.channel` (channel -1 additional) number of samples to stay frame-aligned.
    input_alignment_ditch: usize,

    // Flip this to true to stop the thread.
    stop_thread: Arc<AtomicBool>,
}

impl<T> Drop for StreamingResampler<T>
where
    T: Float + FftNum,
{
    fn drop(&mut self) {
        self.stop_thread.store(true, Ordering::Relaxed);
    }
}

impl<T> StreamingResampler<T>
where
    T: Float + FftNum,
{
    /// Constructs a sample-streaming resampler from `config`.
    pub fn new(config: Config, streaming_config: StreamingConfig) -> Self {
        let span_format_in = SpanFormat {
            sample_rate: config.input_sample_rate as u32,
            channels: config.channels as u8,
        };

        let span_format_out = SpanFormat {
            sample_rate: config.output_sample_rate as u32,
            channels: config.channels as u8,
        };

        // TODO, make this configurable based on expected_input_range, expected_output_range, expected max channels
        let (input_buffer_size, output_buffer_size) = streaming_config.required_buffer_sizes(&config);
        let (mut in_producer, mut in_consumer) = RingBuffer::new(input_buffer_size * 2);
        let (mut out_producer, mut out_consumer) = RingBuffer::new(output_buffer_size * 2);

        let stop_thread: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let thread_should_stop = Arc::clone(&stop_thread);

        // Off-thread resampler loop thread, handles all the off-thread resampling work.
        let thread_handle = std::thread::spawn(move || {
            let mut streaming_sampler = OffThreadStreamingResampler::<T>::new(config.clone())?;
            let mut output_buffer = vec![T::zero(); streaming_sampler.output_buffer_size()];

            let mut span_format_in = SpanFormat {
                sample_rate: config.input_sample_rate as u32,
                channels: config.channels as u8,
            };

            // Loop until we get a "thread should stop" signal.
            while !thread_should_stop.load(Ordering::Relaxed) {
                // Read a single packet from the input ring buffer and process it.
                let did_input_work = match in_consumer.pop() {
                    Ok(packet) => {
                        match packet {
                            Packet::Format(format) => {
                                if span_format_in != format {
                                    streaming_sampler
                                        .new_span(format.sample_rate as usize, format.channels as usize)?;
                                    span_format_in = format;
                                }
                            }
                            Packet::Sample(sample) => {
                                streaming_sampler.write_samples(&[sample])?;
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };

                let did_output_work = {
                    // Check to see if are nearing the end of an output span, if so process end-of-span behavior.
                    if let Some(samples_remaining) = streaming_sampler.samples_left_in_span() {
                        // If there are samples remaining in the span, read them.

                        // First check the size of the output buffer, and grow it if needed.
                        if output_buffer.len() < samples_remaining {
                            output_buffer.resize(samples_remaining, T::zero());
                        }

                        let samples_read = streaming_sampler.read_samples(&mut output_buffer[..samples_remaining]);

                        for i in 0..samples_read {
                            let res = out_producer.push(Packet::Sample(output_buffer[i]));

                            // If the ring buffer is full, ditch enough samples to keep it frame aligned and park the thread.
                            // This should never happen unless the CPU is in heavy contention.
                            if let Err(_) = res {
                                let ditch_samples = config.channels - 1;
                                if ditch_samples > 0 {
                                    // TODO
                                }

                                #[cfg(debug_assertions)]
                                {
                                    eprintln!(
                                        "ardftsrc: output ring buffer full, ditching {ditch_samples} out samples to keep it frame aligned."
                                    );
                                }

                                std::thread::park();
                            }
                        }

                        // We should be right at a span boundary.
                        debug_assert!(
                            streaming_sampler.samples_left_in_span() == Some(0),
                            "ardftsrc: samples_left_in_span() is not 0 at expected span boundary."
                        );

                        // Grab the next span shape
                        let output_channels = streaming_sampler.output_channels();

                        // We *must* write this packet, so loop and park until we can.
                        while !out_producer.is_full() {
                            let output_span_format = SpanFormat {
                                sample_rate: config.output_sample_rate as u32,
                                channels: output_channels as u8,
                            };

                            let res = out_producer.push(Packet::Format(output_span_format));
                            if res.is_err() {
                                std::thread::park();
                            }
                        }

                        true
                    } else {
                        // If we are not nearing the end of an output span, just read samples into the output buffer and write them to the output ring buffer.
                        let samples_read = streaming_sampler.read_samples(&mut output_buffer);

                        for i in 0..samples_read {
                            let res = out_producer.push(Packet::Sample(output_buffer[i]));

                            // If the ring buffer is full, ditch enough samples to keep it frame aligned and park the thread.
                            // This should never happen unless the CPU is in heavy contention.
                            if let Err(_) = res {
                                let ditch_samples = config.channels - 1;
                                if ditch_samples > 0 {
                                    let _ = streaming_sampler.read_samples(&mut output_buffer[ditch_samples..]);
                                }

                                #[cfg(debug_assertions)]
                                {
                                    eprintln!(
                                        "ardftsrc: output ring buffer full, ditching {ditch_samples} out samples to keep it frame aligned."
                                    );
                                }

                                std::thread::park();
                            }
                        }

                        if samples_read > 0 { true } else { false }
                    }
                };

                // If we didn't do any work, park the thread.
                if !did_input_work && !did_output_work {
                    std::thread::park();
                }

                // TODO: There's a race condition here. This can miss a wakeup:
                // consumer: pop() sees empty
                // producer: push()
                // producer: unpark(consumer)
                // consumer: park()
            }

            Ok(())
        });

        Self {
            thread_handle: Some(thread_handle),
            span_format_in,
            span_format_out,
            in_producer,
            out_consumer,
            input_alignment_ditch: 0,
            stop_thread,
        }
    }

    // Shutdown the resampler worker thread.
    // Returns the error from the thread if it panicked or returned an error.
    pub fn shutdown(&mut self) -> Result<(), Error> {
        self.stop_thread.store(true, Ordering::Relaxed);

        if let Some(thread_handle) = self.thread_handle.take() {
            // Wake the thread in case it is currently parked.
            thread_handle.thread().unpark();

            match thread_handle.join() {
                Ok(thread_result) => thread_result?,
                Err(panic_payload) => {
                    let panic_message = if let Some(msg) = panic_payload.downcast_ref::<String>() {
                        msg.clone()
                    } else if let Some(msg) = panic_payload.downcast_ref::<&str>() {
                        (*msg).to_owned()
                    } else {
                        "unknown panic payload".to_owned()
                    };
                    return Err(Error::WorkerThreadPanic(panic_message));
                }
            }
        }

        Ok(())
    }
}

pub struct StreamingConfig {
    expected_input_range: std::ops::Range<usize>,
    expected_max_channels: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            expected_input_range: 22_050..192_000,
            expected_max_channels: 8,
        }
    }
}

impl StreamingConfig {
    // Returns the required input and output buffer sizes for the given config.
    //
    // Returns (input_buffer_size, output_buffer_size)
    pub fn required_buffer_sizes(&self, config: &Config) -> (usize, usize) {
        let quality = config.quality;
        let fixed_output_rate = config.output_sample_rate;
        let min_input_rate = self.expected_input_range.start;
        let max_input_rate = self.expected_input_range.end;

        let mut a = max_input_rate;
        let mut b = fixed_output_rate;
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        debug_assert!(a != 0, "expected non-zero gcd for input/output rates");
        let input_common_divisor = a;
        let mut input_chunk_frames = max_input_rate / input_common_divisor;
        let output_chunk_frames_for_input = fixed_output_rate / input_common_divisor;
        let input_denominator = input_chunk_frames.min(output_chunk_frames_for_input).max(1);
        let input_factor = quality.div_ceil(input_denominator).next_multiple_of(2);
        input_chunk_frames *= input_factor;

        let mut a = min_input_rate;
        let mut b = fixed_output_rate;
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        debug_assert!(a != 0, "expected non-zero gcd for input/output rates");
        let output_common_divisor = a;
        let input_chunk_frames_for_output = min_input_rate / output_common_divisor;
        let mut output_chunk_frames = fixed_output_rate / output_common_divisor;
        let output_denominator = input_chunk_frames_for_output.min(output_chunk_frames).max(1);
        let output_factor = quality.div_ceil(output_denominator).next_multiple_of(2);
        output_chunk_frames *= output_factor;

        (
            input_chunk_frames * self.expected_max_channels,
            output_chunk_frames * self.expected_max_channels,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InterleavedResampler;

    fn endpoint_chunk_sizes(input_rate: usize, output_rate: usize, quality: usize, channels: usize) -> (usize, usize) {
        let mut a = input_rate;
        let mut b = output_rate;
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        let gcd = a;
        let mut input_chunk_frames = input_rate / gcd;
        let mut output_chunk_frames = output_rate / gcd;
        let denominator = input_chunk_frames.min(output_chunk_frames).max(1);
        let factor = quality.div_ceil(denominator).next_multiple_of(2);
        input_chunk_frames *= factor;
        output_chunk_frames *= factor;
        (input_chunk_frames * channels, output_chunk_frames * channels)
    }

    #[test]
    fn required_buffer_sizes_matches_known_direct_math_case() {
        let streaming = StreamingConfig {
            expected_input_range: 22_050..192_000,
            expected_max_channels: 16,
        };
        let mut config = Config::new(48_000, 22_000, 16);
        config.quality = 1_878;

        let (input_size, output_size) = streaming.required_buffer_sizes(&config);
        assert_eq!(input_size, 264_192);
        assert_eq!(output_size, 42_240);
    }

    #[test]
    fn required_buffer_sizes_matches_endpoint_math_for_varied_configs() {
        let cases = [
            (22_050..192_000, 22_000usize, 8usize, 1_878usize),
            (16_000..96_000, 48_000usize, 2usize, 512usize),
            (32_000..192_000, 96_000usize, 6usize, 2_048usize),
        ];

        for (input_range, output_rate, channels, quality) in cases {
            let streaming = StreamingConfig {
                expected_input_range: input_range.clone(),
                expected_max_channels: channels,
            };
            let mut config = Config::new(48_000, output_rate, channels);
            config.quality = quality;

            let (actual_input, actual_output) = streaming.required_buffer_sizes(&config);

            let (expected_input, _) = endpoint_chunk_sizes(input_range.end, output_rate, quality, channels);
            let (_, expected_output) = endpoint_chunk_sizes(input_range.start, output_rate, quality, channels);

            assert_eq!(actual_input, expected_input);
            assert_eq!(actual_output, expected_output);
        }
    }

    #[test]
    fn required_input_size_matches_interleaved_at_max_input_endpoint() {
        let streaming = StreamingConfig {
            expected_input_range: 22_050..192_000,
            expected_max_channels: 8,
        };
        let mut config = Config::new(48_000, 22_000, 8);
        config.quality = 1_878;

        let (required_input_size, _) = streaming.required_buffer_sizes(&config);

        let max_endpoint_config = Config {
            input_sample_rate: streaming.expected_input_range.end,
            output_sample_rate: config.output_sample_rate,
            channels: streaming.expected_max_channels,
            quality: config.quality,
            bandwidth: config.bandwidth,
            taper_type: config.taper_type,
        };
        let interleaved = InterleavedResampler::<f32>::new(max_endpoint_config).unwrap();

        assert_eq!(required_input_size, interleaved.input_buffer_size());
    }

    #[test]
    fn required_output_size_matches_offthread_at_min_input_endpoint() {
        let streaming = StreamingConfig {
            expected_input_range: 22_050..192_000,
            expected_max_channels: 8,
        };
        let mut config = Config::new(48_000, 22_000, 8);
        config.quality = 1_878;

        let (_, required_output_size) = streaming.required_buffer_sizes(&config);

        let min_endpoint_config = Config {
            input_sample_rate: streaming.expected_input_range.start,
            output_sample_rate: config.output_sample_rate,
            channels: streaming.expected_max_channels,
            quality: config.quality,
            bandwidth: config.bandwidth,
            taper_type: config.taper_type,
        };
        let offthread = OffThreadStreamingResampler::<f32>::new(min_endpoint_config).unwrap();

        assert_eq!(required_output_size, offthread.output_buffer_size());
    }
}
