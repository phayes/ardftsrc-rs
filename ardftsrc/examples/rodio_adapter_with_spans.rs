use ardftsrc::{Config, rodio::RodioResampler};
use rodio::Source;
use std::error::Error;
use std::num::NonZero;
use std::thread;
use std::time::Duration;

const FIRST_SPAN_FREQUENCY_HZ: f32 = 440.0;
const SECOND_SPAN_FREQUENCY_HZ: f32 = 660.0;
const AMPLITUDE: f32 = 0.20;
const SPAN_DURATION_SECS: u64 = 2;
const FIRST_SPAN_SAMPLE_RATE_HZ: u32 = 44_100;
const SECOND_SPAN_SAMPLE_RATE_HZ: u32 = 48_000;
const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;

fn main() -> Result<(), Box<dyn Error>> {
    // Keep this handle alive for the full playback duration.
    let stream = rodio::DeviceSinkBuilder::open_default_sink()?;
    let mixer = stream.mixer();

    let first_span_tone = rodio::source::SignalGenerator::new(
        NonZero::new(FIRST_SPAN_SAMPLE_RATE_HZ).expect("constant non-zero sample rate"),
        FIRST_SPAN_FREQUENCY_HZ,
        rodio::source::Function::Sine,
    )
    .amplify(AMPLITUDE)
    .take_duration(Duration::from_secs(SPAN_DURATION_SECS));

    let second_span_tone = rodio::source::SignalGenerator::new(
        NonZero::new(SECOND_SPAN_SAMPLE_RATE_HZ).expect("constant non-zero sample rate"),
        SECOND_SPAN_FREQUENCY_HZ,
        rodio::source::Function::Sine,
    )
    .amplify(AMPLITUDE)
    .take_duration(Duration::from_secs(SPAN_DURATION_SECS));

    let tone = rodio::source::from_iter([first_span_tone, second_span_tone]);

    let config = Config {
        input_sample_rate: FIRST_SPAN_SAMPLE_RATE_HZ as usize,
        output_sample_rate: OUTPUT_SAMPLE_RATE_HZ,
        channels: 1,
        ..Config::default()
    };

    let resampled_tone: RodioResampler<_, f64> = RodioResampler::new(tone, config);

    mixer.add(resampled_tone);
    thread::sleep(Duration::from_secs(SPAN_DURATION_SECS * 2));

    Ok(())
}
