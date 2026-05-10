use num_traits::Float;
use realfft::FftNum;
use rtrb::{Consumer, Producer, RingBuffer};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use crate::{Config, Error, offthread::OffThreadStreamingResampler};

// PERFORMANCE TUNING PARAMETERS
//
// TODO: tune these
const WAKES_PER_CHUNK: usize = 4;
const INITIAL_SAMPLE_DELAY_CHUNKS: usize = 3;
const INPUT_BUFFER_SIZE_MULTIPLIER: usize = 4;
const OUTPUT_BUFFER_SIZE_MULTIPLIER: usize = 8;
const LOCAL_READ_BUFFER_SIZE: usize = 16384;

// Idling parameters
//
// We enter idle mode when both the input and output ring buffers are empty.
// This usually occurs because the user pushed "pause" or the stream ran out of content.
//
// TODO: There's a bug here where once we idle, starting up again underruns and we crackle. 
// We need to tell the front-end that we're indle, then the front-end can wait for us to wake up.
// One possible answer: Add an realtime_on_idle callback to the front-end, backend emit a Idle packet when it's about to park the thread.
// If realtime_on_idle is not set, then we never idle (because we cant do it safely without underrunning)

// How long to idle before yielding the thread.
const IDLE_YIELD_THRESHOLD_MS: u64 = 10;

// How long to idle before parking the thread.
const IDLE_PARK_THRESHOLD_MS: u64 = 1000;

/// Runtime audio format metadata for tapped packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanFormat {
    pub channels: u8,
    pub sample_rate: u32,
}

/// Packet emitted by the low-level ring buffer.
///
/// Invariants:
/// - The first packet for a stream is always `Format`.
/// - Whenever span format changes, a new `Format` packet is emitted before the first `Sample` in that span.
///
/// This should be sized 128bits - my kingdom for a float-niche!
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Packet<T: Float + Copy> {
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

    // TODO:
    // There's a bug here, this should be packets
    // The bug is because we can change the sample rate ratio and channel count across spans,
    // so we need to unpack packets at read_sample() time.
    local_read_buffer: VecDeque<Packet<T>>,

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

    // Track the number of samples read since the last output-side wakeup.
    samples_read_since_wake: usize,

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
        if let Some(thread_handle) = self.thread_handle.take() {
            thread_handle.thread().unpark();
        }
    }
}

impl<T> RealtimeResampler<T>
where
    T: Float + FftNum,
{
    /// Constructs a sample-streaming resampler from `config`.
    pub fn new(config: Config) -> Result<Self, Error> {
        let span_format_in = SpanFormat {
            sample_rate: config.input_sample_rate as u32,
            channels: config.channels as u8,
        };

        let span_format_out = SpanFormat {
            sample_rate: config.output_sample_rate as u32,
            channels: config.channels as u8,
        };

        // TODO, make this configurable based on expected_input_range, expected_output_range, expected max channels
        let (input_buffer_size, output_buffer_size) = config.realtime_buffer_sizes();
        let (in_producer, in_consumer) = RingBuffer::new(input_buffer_size * INPUT_BUFFER_SIZE_MULTIPLIER);
        let (out_producer, out_consumer) = RingBuffer::new(output_buffer_size * OUTPUT_BUFFER_SIZE_MULTIPLIER);

        let stop_thread: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let thread_should_stop = Arc::clone(&stop_thread);

        let thread_handle = launch_thread(config.clone(), thread_should_stop, in_consumer, out_producer)?;

        let mut instance = Self {
            thread_handle: Some(thread_handle),
            span_format_in,
            span_format_out,
            in_producer,
            out_consumer,
            current_span_len: None,
            samples_written_since_wake: 0,
            samples_read_since_wake: 0,
            samples_per_wake: 1,
            config,
            stop_thread,
            initial_sample_delay: 0,
            total_input_samples_written: 0,
            local_read_buffer: VecDeque::with_capacity(LOCAL_READ_BUFFER_SIZE),
        };
        instance.set_samples_per_wake();
        instance.set_initial_sample_delay();
        instance.total_input_samples_written = 0;
        Ok(instance)
    }

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

    // Set the processor to wake up at least 4 times per chunk of input samples.
    fn set_samples_per_wake(&mut self) {
        self.samples_per_wake = (self.input_chunk_samples() / WAKES_PER_CHUNK).max(1);
    }

    // Set the initial sample delay to 2 chunks of input samples.
    fn set_initial_sample_delay(&mut self) {
        self.initial_sample_delay = self.input_chunk_samples() * INITIAL_SAMPLE_DELAY_CHUNKS;
    }

    /// Get the initial sample delay.
    #[inline]
    #[must_use]
    pub fn initial_sample_delay(&self) -> usize {
        self.initial_sample_delay
    }

    pub fn fill_local_read_buffer(&mut self) {
        // If the local read buffer is less than half full, fill it.
        if self.local_read_buffer.len() <= LOCAL_READ_BUFFER_SIZE / 2 {
            // Check how many samples we can read from the ring buffer.
            let want_samples = self
                .out_consumer
                .slots()
                .min(LOCAL_READ_BUFFER_SIZE - self.local_read_buffer.len());
            if want_samples > 0 {
                let packets = self
                    .out_consumer
                    .read_chunk(want_samples)
                    .expect("ardftsrc: failed to read samples despite checking for slots");

                packets.into_iter().for_each(|packet| {
                    self.local_read_buffer.push_back(packet);
                });
            }
        }
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
                    self.track_sample_read_for_wake(1);
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

        // If we are still priming the resampler, return negative-zero silence.
        if self.total_input_samples_written < self.initial_sample_delay {
            return Some(T::neg_zero());
        }

        // Fill the local read buffer with samples from the ring buffer.
        self.fill_local_read_buffer();

        // If we have samples in the local read buffer, return the next sample.
        if let Some(packet) = self.local_read_buffer.pop_front() {
            return match packet {
                Packet::EndOfStream => None,
                Packet::Sample(sample) => {
                    if let Some(current_span_len) = self.current_span_len.as_mut() {
                        *current_span_len -= 1;
                    }
                    self.track_sample_read_for_wake(1);
                    Some(sample)
                }
                Packet::NewSpanPending(n) => {
                    self.current_span_len = Some(n);
                    return self.read_sample();
                }
                Packet::Format(format) => {
                    self.span_format_out = format;
                    self.current_span_len = None;
                    return self.read_sample();
                }
            };
        } else {
            // If we don't have samples in the local read buffer, return negative-zero silence.
            return Some(T::neg_zero());
        }
    }

    fn track_sample_read_for_wake(&mut self, samples_read: usize) {
        self.samples_read_since_wake += samples_read;
        if self.samples_read_since_wake >= self.samples_per_wake {
            self.wake_up();
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
        self.samples_read_since_wake = 0;
    }

    #[inline]
    #[must_use]
    pub fn current_span_len(&self) -> Option<usize> {
        self.current_span_len
    }

    #[inline]
    #[must_use]
    pub fn span_format_in(&self) -> SpanFormat {
        self.span_format_in
    }

    #[inline]
    #[must_use]
    pub fn span_format_out(&self) -> SpanFormat {
        self.span_format_out
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

fn launch_thread<T: Float + FftNum>(
    config: Config,
    thread_should_stop: Arc<AtomicBool>,
    in_consumer: Consumer<Packet<T>>,
    out_producer: Producer<Packet<T>>,
) -> Result<JoinHandle<Result<(), Error>>, Error> {
    std::thread::Builder::new().name("ardftsrc-resampler".to_string()).spawn(move || {
            let mut streaming_sampler = OffThreadStreamingResampler::<T>::new(config.clone())?;
            let mut sample_output_buffer = vec![T::zero(); streaming_sampler.output_buffer_size()];
            let mut packet_output_buffer = Vec::with_capacity(streaming_sampler.output_buffer_size());
            let mut out_producer = out_producer;
            let mut in_consumer = in_consumer;
            let mut idle_time = None; // TODO, maybe move to tick-counter (https://github.com/sheroz/tick_counter)       
            let mut current_output_span_format = SpanFormat {
                sample_rate: config.output_sample_rate as u32,
                channels: streaming_sampler.output_channels() as u8,
            };

            let mut write_output_packets = |output_slice: &[T], out_producer: &mut Producer<Packet<T>>| {
                packet_output_buffer.clear();
                packet_output_buffer.extend(output_slice.iter().map(|s| Packet::Sample(*s)));

                let mut packet_output_buffer_slice = packet_output_buffer.as_slice();

                while !packet_output_buffer_slice.is_empty() {
                    let (_written_part, remaining_part) = out_producer.push_partial_slice(&packet_output_buffer_slice);

                    if remaining_part.is_empty() {
                        break;
                    }

                    // Output is full, yeild the thread.
                    packet_output_buffer_slice = remaining_part;
                    std::thread::yield_now();
                }
            };

            // Loop until we get a "thread should stop" signal.
            while !thread_should_stop.load(Ordering::Relaxed) {
                // Input work first
                let num_packets = in_consumer.slots();
                let mut did_input_work = false;
                if num_packets > 0 {
                    did_input_work = true;

                    for packet in in_consumer
                        .read_chunk(num_packets)
                        .expect("ardftsrc: failed to read input chunk despite checking for slots")
                    {
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
                    }
                }

                // Output work second
                let did_output_work = {
                    // Check to see if are nearing the end of an output span, if so process end-of-span behavior.
                    if let Some(samples_remaining) = streaming_sampler.samples_left_in_span() {
                        // If there are samples remaining in the span, read them.

                        // First check the size of the output buffer, and grow it if needed.
                        if sample_output_buffer.len() < samples_remaining {
                            sample_output_buffer.resize(samples_remaining, T::zero());
                        }

                        let samples_read =
                            streaming_sampler.read_samples(&mut sample_output_buffer[..samples_remaining]);

                        // If we read no samples, check if the stream is done.
                        // If so, push the end of stream packet and return, exiting the thread.
                        if samples_read == 0 {
                            if streaming_sampler.is_done() {
                                let _ = out_producer.push(Packet::EndOfStream);
                                return Ok(());
                            }
                        } else {
                            write_output_packets(&sample_output_buffer[..samples_read], &mut out_producer);
                        }

                        // We should be right at a span boundary.
                        debug_assert!(
                            streaming_sampler.samples_left_in_span() == Some(0),
                            "ardftsrc: samples_left_in_span() is not 0 at expected span boundary."
                        );

                        // Grab the next span shape and check if we should emit a Format packet.
                        let output_channels = streaming_sampler.output_channels();
                        let candidate_output_span_format = SpanFormat {
                            sample_rate: config.output_sample_rate as u32,
                            channels: output_channels as u8,
                        };
                        if current_output_span_format != candidate_output_span_format {
                            current_output_span_format = candidate_output_span_format;

                            // TODO: Handle this error
                            out_producer.push(Packet::Format(candidate_output_span_format)).unwrap();
                        }

                        samples_read > 0
                    } else {
                        // If we are not nearing the end of an output span, just read samples into the output buffer and write them to the output ring buffer.
                        let samples_read = streaming_sampler.read_samples(&mut sample_output_buffer);

                        if samples_read > 0 {
                            write_output_packets(&sample_output_buffer[..samples_read], &mut out_producer);
                        }

                        samples_read > 0
                    }
                };

                // If we didn't do any work, either busy-wait, yield, or park the thread.
                if !did_input_work && !did_output_work {
                    let output_buffer_capacity = out_producer.buffer().capacity();
                    let output_buffer_slots = out_producer.slots();
                    let current_output_samples = output_buffer_capacity - output_buffer_slots;

                    // Check if we should go idles
                    // Idle mode is entered when the output ring buffer empty and the input ring buffer is empty.
                    //
                    // It usually occurs because the user pushed "pause" or the stream ran out of content.
                    if current_output_samples == 0 && in_consumer.slots() == 0 {
                        if let Some(idle_start) = idle_time {
                            let idle_duration = std::time::Instant::now().duration_since(idle_start);

                            if idle_duration.as_millis() > IDLE_PARK_THRESHOLD_MS as u128 {
                                std::thread::park();
                            }
                            else if idle_duration.as_millis() > IDLE_YIELD_THRESHOLD_MS as u128 {
                                std::thread::yield_now();
                            }
                            else {
                                std::hint::spin_loop();
                            }
                        }
                        else {
                            // enter idle and continue
                            idle_time = Some(std::time::Instant::now());
                            continue;
                        }
                    } else {
                        // Exit idle
                        if idle_time.is_some() {
                            idle_time = None;
                        }

                        // We shouldn't be idling, but maybe we should at least yield the thread.
                        if current_output_samples < streaming_sampler.output_buffer_size() {
                            // if the output ring buffer has less than one buffer-worth of samples, busy wait so we dont underrun.
                            std::hint::spin_loop();
                        } else {
                            // We've got enough leeway to yield the thread.
                            std::thread::yield_now();
                        }
                    }
                }
            }

            Ok(())
        }).map_err(|e| Error::FailedToLaunchWorkerThread(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InterleavedResampler;
    use std::mem::size_of;

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
        let mut config = Config::new(48_000, 22_000, 16);
        config.quality = 1_878;
        config.realtime_input_range = 22_050..192_000;
        config.realtime_max_channels = 16;

        let (input_size, output_size) = config.realtime_buffer_sizes();
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
            let mut config = Config::new(48_000, output_rate, channels);
            config.quality = quality;
            config.realtime_input_range = input_range.clone();
            config.realtime_max_channels = channels;

            let (actual_input, actual_output) = config.realtime_buffer_sizes();

            let (expected_input, _) = endpoint_chunk_sizes(input_range.end, output_rate, quality, channels);
            let (_, expected_output) = endpoint_chunk_sizes(input_range.start, output_rate, quality, channels);

            assert_eq!(actual_input, expected_input);
            assert_eq!(actual_output, expected_output);
        }
    }

    #[test]
    fn default_required_buffer_sizes_match_current_memory_use() {
        let config = Config::default();

        let (input_size, output_size) = config.realtime_buffer_sizes();
        let input_capacity = input_size * INPUT_BUFFER_SIZE_MULTIPLIER;
        let output_capacity = output_size * OUTPUT_BUFFER_SIZE_MULTIPLIER;
        let packet_size = size_of::<Packet<f64>>();

        assert_eq!(input_size, 71_680);
        assert_eq!(output_size, 30_048);
        assert_eq!(packet_size, 16);
        assert_eq!(input_capacity * packet_size, 4_587_520);
        assert_eq!(output_capacity * packet_size, 3_846_144);
        assert_eq!((input_capacity + output_capacity) * packet_size, 8_433_664);
    }

    #[test]
    fn required_input_size_matches_interleaved_at_max_input_endpoint() {
        let mut config = Config::new(48_000, 22_000, 8);
        config.quality = 1_878;
        config.realtime_input_range = 22_050..192_000;
        config.realtime_max_channels = 8;

        let (required_input_size, _) = config.realtime_buffer_sizes();

        let max_endpoint_config = Config {
            input_sample_rate: config.realtime_input_range.end,
            output_sample_rate: config.output_sample_rate,
            channels: config.realtime_max_channels,
            quality: config.quality,
            bandwidth: config.bandwidth,
            taper_type: config.taper_type,
            realtime_input_range: config.realtime_input_range.clone(),
            realtime_max_channels: config.realtime_max_channels,
            rodio_fast_start: config.rodio_fast_start,
        };
        let interleaved = InterleavedResampler::<f32>::new(max_endpoint_config).unwrap();

        assert_eq!(required_input_size, interleaved.input_buffer_size());
    }

    #[test]
    fn required_output_size_matches_offthread_at_min_input_endpoint() {
        let mut config = Config::new(48_000, 22_000, 8);
        config.quality = 1_878;
        config.realtime_input_range = 22_050..192_000;
        config.realtime_max_channels = 8;

        let (_, required_output_size) = config.realtime_buffer_sizes();

        let min_endpoint_config = Config {
            input_sample_rate: config.realtime_input_range.start,
            output_sample_rate: config.output_sample_rate,
            channels: config.realtime_max_channels,
            quality: config.quality,
            bandwidth: config.bandwidth,
            taper_type: config.taper_type,
            realtime_input_range: config.realtime_input_range.clone(),
            realtime_max_channels: config.realtime_max_channels,
            rodio_fast_start: config.rodio_fast_start,
        };
        let offthread = OffThreadStreamingResampler::<f32>::new(min_endpoint_config).unwrap();

        assert_eq!(required_output_size, offthread.output_buffer_size());
    }
}
