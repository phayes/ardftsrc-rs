use ardftsrc::{Ardftsrc, Config};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use wavers::{Wav, write};

const OUTPUT_SAMPLE_RATE_HZ: usize = 48_000;
const INPUT_WAVS: &[&str] = &[
    "test_wavs/example-pcm16-44k1-stereo.wav",
    "test_wavs/sweep-pcm16-22k05.wav",
    "test_wavs/gaplesstest_s-f32-96k.wav",
    "test_wavs/gaplesstest_m-f32-96k.wav",
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
    let mut resampler = Ardftsrc::new(config)?;

    let output_path = output_dir.join(format!(
        "{}_streaming_to_{}hz.wav",
        input_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("converted"),
        OUTPUT_SAMPLE_RATE_HZ
    ));
    let output_samples_target = resampler.expected_output_size(input_frames) * channels;
    let (samples_written, output_samples) =
        stream_samples(&mut reader, &mut resampler, output_samples_target)?;
    write(
        &output_path,
        &output_samples,
        OUTPUT_SAMPLE_RATE_HZ as i32,
        channels as u16,
    )?;
    println!(
        "{}: {} Hz -> {} Hz, {} frames -> {} frames",
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
    resampler: &mut Ardftsrc,
    output_samples_target: usize,
) -> Result<(usize, Vec<f32>), Box<dyn Error>> {
    let input_buffer_size = resampler.input_chunk_size();
    let output_buffer_size = resampler.output_chunk_size();
    let mut input_chunk = vec![0.0; input_buffer_size];
    let mut output_chunk = vec![0.0; output_buffer_size];
    let mut input_len = 0;
    let mut samples_written = 0;
    let mut output_samples = Vec::with_capacity(output_samples_target);
    let mut samples_read = 0;
    let total_samples = reader.n_samples();

    while samples_read < total_samples {
        let read_count = (total_samples - samples_read).min(input_buffer_size - input_len);
        let chunk = reader.read_samples(read_count)?;
        if chunk.is_empty() {
            break;
        }
        input_chunk[input_len..input_len + chunk.len()].copy_from_slice(&chunk);
        input_len += chunk.len();
        samples_read += chunk.len();

        if input_len == input_buffer_size {
            let written = resampler.process_chunk(&input_chunk, &mut output_chunk)?;
            samples_written += collect_limited(
                &mut output_samples,
                &output_chunk[..written],
                output_samples_target - samples_written,
            );
            input_len = 0;
        }
    }

    if input_len > 0 {
        let channels = resampler.config().channels;
        if !input_len.is_multiple_of(channels) {
            return Err(format!(
                "input ended with incomplete frame: {input_len} trailing samples for {channels} channels"
            )
            .into());
        }
        let written = resampler.process_chunk_final(&input_chunk[..input_len], &mut output_chunk)?;
        samples_written += collect_limited(
            &mut output_samples,
            &output_chunk[..written],
            output_samples_target - samples_written,
        );
    }

    let written = resampler.finalize(&mut output_chunk)?;
    samples_written += collect_limited(
        &mut output_samples,
        &output_chunk[..written],
        output_samples_target - samples_written,
    );

    Ok((samples_written, output_samples))
}

fn collect_limited(
    output_samples: &mut Vec<f32>,
    samples: &[f32],
    remaining: usize,
) -> usize {
    let samples_to_write = samples.len().min(remaining);
    output_samples.extend_from_slice(&samples[..samples_to_write]);
    samples_to_write
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
