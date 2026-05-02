use ardftsrc::{Ardftsrc, Config};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;
const INPUT_WAVS: &[&str] = &[
    "test_wavs/example-pcm16-44k1-stereo.wav",
    "test_wavs/sweep-pcm16-22k05.wav",
    "test_wavs/gaplesstest_s-f32-96k.wav",
    "test_wavs/gaplesstest_m-f32-96k.wav",
];

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = temp_output_dir()?;
    println!("Writing converted WAVs to {}", output_dir.display());

    for input_path in INPUT_WAVS {
        let input_path = Path::new(input_path);
        let wav = read_wav_f32(input_path)?;
        let config = Config {
            input_sample_rate: wav.sample_rate_hz as usize,
            output_sample_rate: OUTPUT_SAMPLE_RATE_HZ,
            channels: wav.channels,
            ..Config::default()
        };
        let mut resampler = Ardftsrc::new(config)?;
        let converted = resampler.process_all(&wav.samples)?;

        let output_path = output_dir.join(format!(
            "{}_to_{}hz.wav",
            input_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("converted"),
            OUTPUT_SAMPLE_RATE_HZ
        ));
        write_wav_f32(&output_path, wav.channels, OUTPUT_SAMPLE_RATE_HZ as u32, &converted)?;

        println!(
            "{}: {} Hz -> {} Hz, {} frames -> {} frames",
            input_path.display(),
            wav.sample_rate_hz,
            OUTPUT_SAMPLE_RATE_HZ,
            wav.samples.len() / wav.channels,
            converted.len() / wav.channels
        );
        println!("  wrote {}", output_path.display());
    }

    Ok(())
}

struct WavData {
    samples: Vec<f32>,
    channels: usize,
    sample_rate_hz: u32,
}

fn read_wav_f32(path: &Path) -> Result<WavData, Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => read_int_samples(&mut reader, spec.bits_per_sample)?,
    };

    Ok(WavData {
        samples,
        channels: spec.channels as usize,
        sample_rate_hz: spec.sample_rate,
    })
}

fn read_int_samples<R: std::io::Read>(
    reader: &mut WavReader<R>,
    bits_per_sample: u16,
) -> Result<Vec<f32>, hound::Error> {
    let scale = (1_i64 << (bits_per_sample.saturating_sub(1) as u32)) as f32;

    if bits_per_sample <= 16 {
        reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| sample as f32 / scale))
            .collect()
    } else {
        reader
            .samples::<i32>()
            .map(|sample| sample.map(|sample| sample as f32 / scale))
            .collect()
    }
}

fn write_wav_f32(path: &Path, channels: usize, sample_rate_hz: u32, samples: &[f32]) -> Result<(), hound::Error> {
    let spec = WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate_hz,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()
}

fn temp_output_dir() -> Result<PathBuf, Box<dyn Error>> {
    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("ardftsrc_examples_{ts_millis}"));
    fs::create_dir_all(&path)?;
    Ok(path)
}
