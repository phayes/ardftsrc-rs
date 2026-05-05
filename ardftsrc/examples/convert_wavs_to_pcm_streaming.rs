use ardftsrc::{Config, StreamingResampler};
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use wavers::Wav;

const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;
const STREAM_BUFFER_FRAMES: usize = 4096;
const INPUT_WAVS: &[&str] = &[
    "../test_wavs/example-pcm16-44k1-stereo.wav",
    "../test_wavs/sweep-pcm16-22k05.wav",
    "../test_wavs/gaplesstest_s-f32-96k.wav",
    "../test_wavs/gaplesstest_m-f32-96k.wav",
];

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = temp_output_dir()?;
    println!("Streaming converted WAVs to {}", output_dir.display());

    for input_path in INPUT_WAVS {
        convert_one(Path::new(input_path), &output_dir)?;
    }

    Ok(())
}

fn convert_one(input_path: &Path, output_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut reader = Wav::<f32>::from_path(input_path)?;
    let channels = reader.n_channels() as usize;
    let input_samples = reader.n_samples();
    let input_frames = input_samples / channels;
    let config = Config {
        input_sample_rate: reader.sample_rate() as usize,
        output_sample_rate: OUTPUT_SAMPLE_RATE_HZ,
        channels,
        ..Config::default()
    };
    let mut resampler = StreamingResampler::<f32>::new(config)?;

    let output_path = output_dir.join(format!(
        "{}_streaming_to_{}hz_f32le.pcm",
        input_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("converted"),
        OUTPUT_SAMPLE_RATE_HZ
    ));
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);
    let samples_written = stream_samples(&mut reader, &mut resampler, &mut writer)?;
    writer.flush()?;
    println!(
        "{}: {} Hz -> {} Hz, {} frames -> {} frames (raw f32 LE PCM)",
        input_path.display(),
        reader.sample_rate(),
        OUTPUT_SAMPLE_RATE_HZ,
        input_frames,
        samples_written / channels
    );
    println!("  wrote {}", output_path.display());

    Ok(())
}

fn stream_samples(
    reader: &mut Wav<f32>,
    resampler: &mut StreamingResampler<f32>,
    writer: &mut BufWriter<File>,
) -> Result<usize, Box<dyn Error>> {
    let channels = resampler.config().channels;
    let buffer_samples = STREAM_BUFFER_FRAMES * channels;
    let mut output_chunk = vec![0.0; buffer_samples];
    let mut output_bytes = Vec::with_capacity(buffer_samples * std::mem::size_of::<f32>());
    let mut samples_written = 0;
    let mut samples_read = 0;
    let total_samples = reader.n_samples();

    while samples_read < total_samples {
        let read_count = (total_samples - samples_read).min(buffer_samples);
        let chunk = reader.read_samples(read_count)?;
        if chunk.is_empty() {
            break;
        }
        samples_read += chunk.len();

        resampler.write_samples(&chunk)?;
        samples_written += drain_available_samples(resampler, writer, &mut output_chunk, &mut output_bytes)?;
    }

    resampler.finalize_samples()?;
    samples_written += drain_available_samples(resampler, writer, &mut output_chunk, &mut output_bytes)?;

    Ok(samples_written)
}

fn drain_available_samples(
    resampler: &mut StreamingResampler<f32>,
    writer: &mut BufWriter<File>,
    output_chunk: &mut [f32],
    output_bytes: &mut Vec<u8>,
) -> Result<usize, Box<dyn Error>> {
    let mut samples_written = 0;
    loop {
        let written = resampler.read_samples(output_chunk);
        if written == 0 {
            break;
        }
        write_f32le_pcm(writer, output_bytes, &output_chunk[..written])?;
        samples_written += written;
    }
    Ok(samples_written)
}

fn write_f32le_pcm(
    writer: &mut BufWriter<File>,
    output_bytes: &mut Vec<u8>,
    samples: &[f32],
) -> Result<(), Box<dyn Error>> {
    output_bytes.clear();
    output_bytes.reserve(samples.len() * std::mem::size_of::<f32>());
    for sample in samples {
        output_bytes.extend_from_slice(&sample.to_le_bytes());
    }
    writer.write_all(output_bytes)?;
    Ok(())
}

fn temp_output_dir() -> Result<PathBuf, Box<dyn Error>> {
    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("ardftsrc_streaming_examples_{ts_millis}"));
    fs::create_dir_all(&path)?;
    Ok(path)
}
