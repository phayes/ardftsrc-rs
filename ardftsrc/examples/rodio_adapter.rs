use ardftsrc::PRESET_FAST;
use ardftsrc::rodio::RodioResampler;
use rodio::Source;
use std::error::Error;
use std::num::NonZero;
use std::thread;
use std::time::Duration;

const FREQUENCY_HZ: f32 = 440.0;
const AMPLITUDE: f32 = 0.20;
const DURATION_SECS: u64 = 3;
const INPUT_SAMPLE_RATE_HZ: usize = 44_100;
const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;

fn main() -> Result<(), Box<dyn Error>> {
    // Keep this handle alive for the full playback duration.
    let stream = rodio::DeviceSinkBuilder::open_default_sink()?;
    let mixer = stream.mixer();

    let tone = rodio::source::SignalGenerator::new(
        NonZero::new(INPUT_SAMPLE_RATE_HZ as u32).unwrap(),
        FREQUENCY_HZ,
        rodio::source::Function::Sine,
    )
    .amplify(AMPLITUDE)
    .take_duration(Duration::from_secs(DURATION_SECS));

    let config = PRESET_FAST
        .with_channels(1)
        .with_output_rate(OUTPUT_SAMPLE_RATE_HZ)
        .with_input_rate(INPUT_SAMPLE_RATE_HZ);
    let resampled_tone = RodioResampler::new(tone, config);

    mixer.add(resampled_tone);
    thread::sleep(Duration::from_secs(DURATION_SECS));

    Ok(())
}
