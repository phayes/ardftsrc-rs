use ardftsrc::{Ardftsrc, Config};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use wavers::{Wav, read, write};

const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;
const INPUT_WAVS: &[&str] = &[
    "../test_wavs/example-pcm16-44k1-stereo.wav",
    "../test_wavs/sweep-pcm16-22k05.wav",
    "../test_wavs/gaplesstest_s-f32-96k.wav",
    "../test_wavs/gaplesstest_m-f32-96k.wav",
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
    let wav = Wav::<f32>::from_path(path)?;
    let channels = wav.n_channels() as usize;
    let sample_rate_hz = wav.sample_rate() as u32;
    let (samples, _) = read::<f32, _>(path)?;

    Ok(WavData {
        samples: samples.to_vec(),
        channels,
        sample_rate_hz,
    })
}

fn write_wav_f32(path: &Path, channels: usize, sample_rate_hz: u32, samples: &[f32]) -> Result<(), Box<dyn Error>> {
    write(path, samples, sample_rate_hz as i32, channels as u16)?;
    Ok(())
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
