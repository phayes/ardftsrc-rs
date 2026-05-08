use ardftsrc::{Config, StreamingConfig, StreamingResampler, rodio::RodioResampler};
use rodio::Source;
use std::error::Error;
use std::num::NonZero;
use std::thread;
use std::time::Duration;

const FREQUENCY_HZ: f32 = 440.0;
const AMPLITUDE: f32 = 0.20;
const DURATION_SECS: u64 = 3;
const INPUT_SAMPLE_RATE_HZ: u32 = 44_100;
const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;

fn main() -> Result<(), Box<dyn Error>> {
    // Keep this handle alive for the full playback duration.
    let stream = rodio::DeviceSinkBuilder::open_default_sink()?;
    let mixer = stream.mixer();

    let tone = rodio::source::SignalGenerator::new(
        NonZero::new(INPUT_SAMPLE_RATE_HZ).expect("constant non-zero sample rate"),
        FREQUENCY_HZ,
        rodio::source::Function::Sine,
    )
    .amplify(AMPLITUDE)
    .take_duration(Duration::from_secs(DURATION_SECS));

    let config = Config {
        input_sample_rate: INPUT_SAMPLE_RATE_HZ as usize,
        output_sample_rate: OUTPUT_SAMPLE_RATE_HZ,
        channels: 1,
        ..Config::default()
    };
    let streaming_resampler = StreamingResampler::<f32>::new(config, StreamingConfig::default());
    let resampled_tone: RodioResampler<_, f32> = RodioResampler::new(tone, streaming_resampler);

    mixer.add(resampled_tone);

    println!(
        "Playing {FREQUENCY_HZ} Hz sine wave for {DURATION_SECS} seconds ({INPUT_SAMPLE_RATE_HZ} Hz -> {OUTPUT_SAMPLE_RATE_HZ} Hz)..."
    );
    thread::sleep(Duration::from_secs(DURATION_SECS));
    println!("Done.");

    Ok(())
}
