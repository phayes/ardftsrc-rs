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
    println!("Streaming converted WAVs to {}", output_dir.display());

    for input_path in INPUT_WAVS {
        convert_one(Path::new(input_path), &output_dir)?;
    }

    Ok(())
}

fn convert_one(input_path: &Path, output_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut reader = WavReader::open(input_path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let input_frames = reader.duration() as usize / channels;
    let config = Config {
        input_sample_rate: spec.sample_rate as usize,
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
    let mut writer = WavWriter::create(
        &output_path,
        WavSpec {
            channels: channels as u16,
            sample_rate: OUTPUT_SAMPLE_RATE_HZ as u32,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )?;

    let output_samples_target = resampler.expected_output_size(input_frames) * channels;
    let samples_written = match spec.sample_format {
        SampleFormat::Float => stream_samples(
            reader.samples::<f32>(),
            &mut resampler,
            &mut writer,
            output_samples_target,
        )?,
        SampleFormat::Int if spec.bits_per_sample <= 16 => {
            let scale = int_scale(spec.bits_per_sample);
            stream_samples(
                reader
                    .samples::<i16>()
                    .map(move |sample| sample.map(|sample| sample as f32 / scale)),
                &mut resampler,
                &mut writer,
                output_samples_target,
            )?
        }
        SampleFormat::Int => {
            let scale = int_scale(spec.bits_per_sample);
            stream_samples(
                reader
                    .samples::<i32>()
                    .map(move |sample| sample.map(|sample| sample as f32 / scale)),
                &mut resampler,
                &mut writer,
                output_samples_target,
            )?
        }
    };

    writer.finalize()?;
    println!(
        "{}: {} Hz -> {} Hz, {} frames -> {} frames",
        input_path.display(),
        spec.sample_rate,
        OUTPUT_SAMPLE_RATE_HZ,
        input_frames,
        samples_written / channels
    );
    println!("  wrote {}", output_path.display());

    Ok(())
}

fn stream_samples<I>(
    samples: I,
    resampler: &mut Ardftsrc,
    writer: &mut WavWriter<std::io::BufWriter<std::fs::File>>,
    output_samples_target: usize,
) -> Result<usize, Box<dyn Error>>
where
    I: IntoIterator<Item = Result<f32, hound::Error>>,
{
    let input_buffer_size = resampler.input_chunk_size();
    let output_buffer_size = resampler.output_chunk_size();
    let mut input_chunk = vec![0.0; input_buffer_size];
    let mut output_chunk = vec![0.0; output_buffer_size];
    let mut input_len = 0;
    let mut samples_written = 0;

    for sample in samples {
        input_chunk[input_len] = sample?;
        input_len += 1;

        if input_len == input_buffer_size {
            let written = resampler.process_chunk(&input_chunk, &mut output_chunk)?;
            samples_written += write_limited(
                writer,
                &output_chunk[..written],
                output_samples_target - samples_written,
            )?;
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
        samples_written += write_limited(
            writer,
            &output_chunk[..written],
            output_samples_target - samples_written,
        )?;
    }

    let written = resampler.finalize(&mut output_chunk)?;
    samples_written += write_limited(
        writer,
        &output_chunk[..written],
        output_samples_target - samples_written,
    )?;

    Ok(samples_written)
}

fn write_limited(
    writer: &mut WavWriter<std::io::BufWriter<std::fs::File>>,
    samples: &[f32],
    remaining: usize,
) -> Result<usize, hound::Error> {
    let samples_to_write = samples.len().min(remaining);
    for sample in samples.iter().take(samples_to_write) {
        writer.write_sample(*sample)?;
    }
    Ok(samples_to_write)
}

fn int_scale(bits_per_sample: u16) -> f32 {
    (1_i64 << (bits_per_sample.saturating_sub(1) as u32)) as f32
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
