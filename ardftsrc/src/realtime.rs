use num_traits::Float;
use realfft::FftNum;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use crate::{Config, Error, offthread::OffThreadStreamingResampler};

/// Runtime audio format metadata for tapped packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpanFormat {
    pub(crate) channels: u8,
    pub(crate) sample_rate: u32,
}

/// Packet emitted by the low-level ring buffer.
///
/// Invariants:
/// - The first packet for a stream is always `Format`.
/// - Whenever span format changes, a new `Format` packet is emitted before the first `Sample` in that span.
///
/// This should be sized 128bits - my kingdom for a float-niche!
#[derive(Debug, Clone, PartialEq)]
pub enum Packet<T: Float> {
    /// Format packet announces the current `{channels, sample_rate_hz}`.
    Format(SpanFormat),
    /// New span pending in `n` samples
    NewSpanPending(usize),
    /// A single sample (usually f64)
    Sample(T),
    /// End of stream packet
    EndOfStream,
}

pub struct RealtimeResampler<T = f64>
where
    T: Float + FftNum,
{

    // The configuration for the resampler.
    // TODO: Move all this into ResamplingConfig and derive downstream Config.
    config: Config,

    thread_handle: Option<JoinHandle<Result<(), Error>>>,

    // The shape of the input span, when this changes we emit a "SpanChanged"
    pub(crate) span_format_in: SpanFormat,

    // The shape of the output span, we update this when we receive a "SpanChanged" packet.
    pub(crate) span_format_out: SpanFormat,

    // The input-samples producer for the input ring buffer.
    in_producer: Producer<Packet<T>>,

    // The output-samples consumer for the output ring buffer.
    out_consumer: Consumer<Packet<T>>,

    // Flip this to true to stop the thread.
    stop_thread: Arc<AtomicBool>,

    // The size of the current span in samples. None for unknown.
    current_span_len: Option<usize>,

    // Track the number of samples written since the last wakeup.
    samples_written_since_wake: usize,

    // The number of samples to write before waking up the thread.
    // This can change as the sample-rate ratio and channel count changes across spans.
    samples_per_wake: usize,

    // Initial sample delay to allow the resampler to prime the FFT, then prime the output ring buffer.
    // This can change as the sample-rate ratio and channel count changes across spans (unlikely in first few ms, but still possible)
    initial_sample_delay: usize,

    // Track the total number of input samples written to the resampler.
    // This is used to determine if we need to prime output with silence.
    total_input_samples_written: usize,
}

impl<T> Drop for RealtimeResampler<T>
where
    T: Float + FftNum,
{
    fn drop(&mut self) {
        self.stop_thread.store(true, Ordering::Relaxed);
    }
}

impl<T> RealtimeResampler<T>
where
    T: Float + FftNum,
{
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }

    fn input_chunk_samples(&self) -> usize {
        let input_sample_rate = self.span_format_in.sample_rate as usize;
        let output_sample_rate = self.config.output_sample_rate;
        let channels = self.span_format_in.channels as usize;

        let common_divisor = Self::gcd(input_sample_rate, output_sample_rate);
        debug_assert!(common_divisor != 0, "expected non-zero gcd for input/output rates");

        let mut input_chunk_frames = input_sample_rate / common_divisor;
        let output_chunk_frames = output_sample_rate / common_divisor;
        let denominator = input_chunk_frames.min(output_chunk_frames).max(1);
        let factor = self.config.quality.div_ceil(denominator).next_multiple_of(2);
        input_chunk_frames *= factor;

        input_chunk_frames * channels
    }

    fn set_samples_per_wake(&mut self) {
        self.samples_per_wake = self.input_chunk_samples().div_ceil(4).max(1);
    }

    fn set_initial_sample_delay(&mut self) {
        self.initial_sample_delay = self.input_chunk_samples() * 2;
    }

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
        let thread_config = config.clone();

        // TODO, make this configurable based on expected_input_range, expected_output_range, expected max channels
        let (input_buffer_size, output_buffer_size) = streaming_config.required_buffer_sizes(&config);
        let (in_producer, mut in_consumer) = RingBuffer::new(input_buffer_size * 4);
        let (mut out_producer, out_consumer) = RingBuffer::new(output_buffer_size * 4);

        let stop_thread: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let thread_should_stop = Arc::clone(&stop_thread);

        // Off-thread resampler loop thread, handles all the off-thread resampling work.
        let thread_handle = std::thread::spawn(move || {
            let mut streaming_sampler = OffThreadStreamingResampler::<T>::new(thread_config)?;
            let mut output_buffer = vec![T::zero(); streaming_sampler.output_buffer_size()];

            // Loop until we get a "thread should stop" signal.
            while !thread_should_stop.load(Ordering::Relaxed) {
                // TODO: Check if the input ring-buffer is full, if it is that means we're falling behind.
                // Read forward in a frame-aligned way until either we hit a span boundary or we've consumed enough samples to catch up.

                // Read a single packet from the input ring buffer and process it.
                let did_input_work = match in_consumer.pop() {
                    Ok(packet) => {
                        match packet {
                            Packet::EndOfStream => {
                                streaming_sampler.finalize_samples()?;

                                let pending_packet =
                                    Packet::NewSpanPending(streaming_sampler.samples_pending_in_output_span());
                                // TODO: Handle this error
                                out_producer.push(pending_packet).unwrap();
                            }
                            Packet::Format(format) => {
                                streaming_sampler.new_span(format.sample_rate as usize, format.channels as usize)?;

                                let pending_packet =
                                    Packet::NewSpanPending(streaming_sampler.samples_pending_in_output_span());
                                // TODO: Handle this error
                                out_producer.push(pending_packet).unwrap();
                            }
                            Packet::Sample(sample) => {
                                streaming_sampler.write_samples(&[sample])?;
                            }
                            Packet::NewSpanPending(_) => {
                                // Ignore - we'll handle it on Packet::Format when the new span is actually announced.
                                #[cfg(debug_assertions)]
                                panic!(
                                    "ardftsrc: NewSpanPending packet received. The input thread should not be sending this packet."
                                );
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };

                let did_output_work = {
                    // Check to see if are nearing the end of an output span, if so process end-of-span behavior.
                    // TODO: There's a bug here:
                    //   - The first time around we see streaming_sampler.samples_left_in_span() == Some(foo) (!= 0)
                    //   - The second time around we see streaming_sampler.samples_left_in_span() == Some(0), and then re-emit out_producer.push(Packet::Format(output_span_format)).unwrap();
                    //   - The issue is that we always need to emit the format packet at least once, even when streaming_sampler.samples_left_in_span() is naturally zero from the upstream resampler.
                    //   - IF samples_left_in_span() is always non-zero when we complete a span because of finalize, we are safe skip the Some(0) case.
                    //   
                    //   Bug TLDR: We spam Format packets onto the ring buffer during span transitions. 
                    //   This can be fixed when we move the buffered output on the ring buffer since we'll be able to peek.
                    if let Some(samples_remaining) = streaming_sampler.samples_left_in_span() {
                        // If there are samples remaining in the span, read them.

                        // First check the size of the output buffer, and grow it if needed.
                        if output_buffer.len() < samples_remaining {
                            output_buffer.resize(samples_remaining, T::zero());
                        }

                        let samples_read = streaming_sampler.read_samples(&mut output_buffer[..samples_remaining]);

                        // If we read no samples, check if the stream is done.
                        // If so, push the end of stream packet and return, exiting the thread.
                        if samples_read == 0 {
                            if streaming_sampler.is_done() {
                                let _ = out_producer.push(Packet::EndOfStream);
                                return Ok(());
                            }
                        }

                        let mut i = 0usize;
                        while i < samples_read {
                            let res = out_producer.push(Packet::Sample(output_buffer[i]));

                            // If the ring buffer is full, park and retry this same sample after wakeup.
                            // This should never happen unless the CPU is in heavy contention.
                            if let Err(_) = res {
                                #[cfg(debug_assertions)]
                                eprintln!("ardftsrc: output ring buffer full, parking producer thread.");

                                std::thread::park();
                                continue;
                            }
                            i += 1;
                        }

                        // We should be right at a span boundary.
                        debug_assert!(
                            streaming_sampler.samples_left_in_span() == Some(0),
                            "ardftsrc: samples_left_in_span() is not 0 at expected span boundary."
                        );

                        // Grab the next span shape
                        let output_channels = streaming_sampler.output_channels();

                        let output_span_format = SpanFormat {
                            sample_rate: config.output_sample_rate as u32,
                            channels: output_channels as u8,
                        };

                        // TODO: Handle this error
                        out_producer.push(Packet::Format(output_span_format)).unwrap();

                        true
                    } else {
                        // If we are not nearing the end of an output span, just read samples into the output buffer and write them to the output ring buffer.
                        let samples_read = streaming_sampler.read_samples(&mut output_buffer);

                        let mut i = 0usize;
                        while i < samples_read {
                            let res = out_producer.push(Packet::Sample(output_buffer[i]));

                            // If the ring buffer is full, park and retry this same sample after wakeup.
                            // This should never happen unless the CPU is in heavy contention.
                            if let Err(_) = res {
                                #[cfg(debug_assertions)]
                                eprintln!("ardftsrc: output ring buffer full, parking producer thread.");

                                std::thread::park();
                                continue;
                            }
                            i += 1;
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

        let mut instance = Self {
            thread_handle: Some(thread_handle),
            span_format_in,
            span_format_out,
            in_producer,
            out_consumer,
            current_span_len: None,
            samples_written_since_wake: 0,
            samples_per_wake: 1,
            config,
            stop_thread,
            initial_sample_delay: 0,
            total_input_samples_written: 0,
        };
        instance.set_samples_per_wake();
        instance.set_initial_sample_delay();
        instance.total_input_samples_written = 0;
        instance
    }

    pub fn read_sample(&mut self) -> Option<T> {
        // Check if the ring buffer is abandoned, if so read until the end.
        if self.out_consumer.is_abandoned() {
            return match self.out_consumer.pop() {
                Ok(Packet::EndOfStream) => None,
                Ok(Packet::Sample(sample)) => {
                    if let Some(current_span_len) = self.current_span_len.as_mut() {
                        *current_span_len -= 1;
                    }
                    Some(sample)
                }
                Ok(Packet::NewSpanPending(n)) => {
                    self.current_span_len = Some(n);
                    return self.read_sample();
                }
                Ok(Packet::Format(format)) => {
                    self.span_format_out = format;
                    self.current_span_len = None;
                    return self.read_sample();
                }
                Err(_) => None,
            };
        }

        if self.total_input_samples_written < self.initial_sample_delay {
            return Some(T::zero());
        }

        if let Ok(packet) = self.out_consumer.pop() {
            match packet {
                Packet::EndOfStream => None,
                Packet::Sample(sample) => {
                    if let Some(current_span_len) = self.current_span_len.as_mut() {
                        *current_span_len -= 1;
                    }
                    Some(sample)
                }
                Packet::NewSpanPending(n) => {
                    self.current_span_len = Some(n);
                    return self.read_sample();
                }
                Packet::Format(format) => {
                    self.span_format_out = format;
                    self.current_span_len = None;
                    self.read_sample()
                }
            }
        } else {
            // Return silence if the ring buffer is empty
            Some(T::zero())
        }
    }

    pub fn write_sample(&mut self, sample: T) {
        let packet = Packet::Sample(sample);
        let res = self.in_producer.push(packet);
        if res.is_err() {
            // TODO: Handle this
        }

        // Track the total number of input samples written to the resampler.
        self.total_input_samples_written = self.total_input_samples_written.saturating_add(1);
        
        // Track if we should wake up the thread to process the pending samples in the input ring buffer.
        self.samples_written_since_wake += 1;
        if self.samples_written_since_wake >= self.samples_per_wake {
            self.wake_up();
        }
    }

    pub fn finalize(&mut self) {
        // TODO: Handle this error condition
        let _ = self.in_producer.push(Packet::EndOfStream);

        // Wake up the thread to wake up and process the end of stream packet.
        self.wake_up();
    }

    pub fn new_span(&mut self, input_sample_rate: usize, channels: usize) {
        let span_format_in = SpanFormat {
            sample_rate: input_sample_rate as u32,
            channels: channels as u8,
        };

        if span_format_in == self.span_format_in {
            return;
        }

        self.span_format_in = span_format_in.clone();
        self.set_samples_per_wake();
        self.set_initial_sample_delay();

        let packet = Packet::Format(span_format_in);

        // TODO: Handle this error
        self.in_producer.push(packet).unwrap();

        self.wake_up();
    }

    pub fn wake_up(&mut self) {
        self.thread_handle
            .as_ref()
            .expect("ardftsrc: thread handle should be set")
            .thread()
            .unpark();

        self.samples_written_since_wake = 0;
    }

    pub fn current_span_len(&self) -> Option<usize> {
        self.current_span_len
    }

    // Shutdown the resampler worker thread.
    // Returns the error from the thread if it panicked or returned an error.
    pub fn shutdown(&mut self) -> Result<(), Error> {
        self.stop_thread.store(true, Ordering::Relaxed);

        if let Some(thread_handle) = self.thread_handle.take() {
            // Wake the thread in case it is currently parked.
            self.wake_up();

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
