#!/usr/bin/env rust-script

//! ```cargo
//! [dependencies]
//! clap = { version = "4.5", features = ["derive"] }
//! i24 = "1.0.1"
//! audioadapter-buffers = "3.0.0"
//! rubato = "2.0.0"
//! wavers = "1.5.1"
//! ```
//!
use audioadapter_buffers::direct::InterleavedSlice;
use clap::Parser;
use i24::i24;
use rubato::{
    Fft, FixedSync, Resampler,
};
use std::error::Error;
use std::path::PathBuf;
use wavers::{Wav, WavType, read, write};

const DEFAULT_CHUNK_SIZE: usize = 512;
const DEFAULT_SUB_CHUNKS: usize = 1;
const DEFAULT_FIXED_SYNC: FixedSync = FixedSync::Both;

#[derive(Debug, Clone, Copy)]
enum OutputEncoding {
    Int16,
    Int24,
    Int32,
    Float32,
    Float64,
}

impl OutputEncoding {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "16i" => Ok(Self::Int16),
            "24i" => Ok(Self::Int24),
            "32i" => Ok(Self::Int32),
            "32f" => Ok(Self::Float32),
            "64f" => Ok(Self::Float64),
            _ => Err(format!(
                "unsupported encoding '{value}', expected one of: 16i, 24i, 32i, 32f, 64f"
            )),
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "wav_resample")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long = "output-rate")]
    output_sample_rate: Option<usize>,
    #[arg(long)]
    encoding: Option<String>,
    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
    chunk_size: usize,
    #[arg(long = "sub-chunks", default_value_t = DEFAULT_SUB_CHUNKS)]
    sub_chunks: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let wav = read_wav(&args.input).map_err(|err| {
        std::io::Error::other(format!(
            "failed to read input WAV '{}': {err}",
            args.input.display()
        ))
    })?;

    let selected_encoding = args
        .encoding
        .as_deref()
        .map(OutputEncoding::parse)
        .transpose()
        .map_err(std::io::Error::other)?
        .unwrap_or_else(|| detect_output_encoding_from_source(&wav.source_encoding_name));

    let output_sample_rate = args
        .output_sample_rate
        .unwrap_or(wav.sample_rate_hz as usize);
    if output_sample_rate == 0 {
        return Err("--output-rate must be greater than zero".into());
    }
    if args.chunk_size == 0 {
        return Err("--chunk-size must be greater than zero".into());
    }
    if args.sub_chunks == 0 {
        return Err("--sub-chunks must be greater than zero".into());
    }
    let input_len = wav.samples.len();
    let conversion_context = format!(
        "input='{}' output='{}' input_rate={} output_rate={} channels={} input_samples={} source_encoding={} output_encoding={} chunk_size={} sub_chunks={}",
        args.input.display(),
        args.output.display(),
        wav.sample_rate_hz,
        output_sample_rate,
        wav.channels,
        input_len,
        wav.source_encoding_name,
        display_encoding(selected_encoding),
        args.chunk_size,
        args.sub_chunks
    );
    let converted_len = match wav.samples {
        InputSamples::F32(samples) => {
            let converted = resample_with_rubato_f32(
                &samples,
                wav.channels,
                wav.sample_rate_hz as usize,
                output_sample_rate,
                args.chunk_size,
                args.sub_chunks,
            )
            .map_err(|err| {
                std::io::Error::other(format!(
                    "resampling failed on f32 input ({conversion_context}): {err}"
                ))
            })?;
            let len = converted.len();
            write_wav_from_f32(
                &args.output,
                wav.channels,
                output_sample_rate as u32,
                &converted,
                selected_encoding,
            )
            .map_err(|err| {
                std::io::Error::other(format!(
                    "failed to write output WAV from f32 samples ({conversion_context}): {err}"
                ))
            })?;
            len
        }
        InputSamples::F64(samples) => {
            let converted = resample_with_rubato_f64(
                &samples,
                wav.channels,
                wav.sample_rate_hz as usize,
                output_sample_rate,
                args.chunk_size,
                args.sub_chunks,
            )
            .map_err(|err| {
                std::io::Error::other(format!(
                    "resampling failed on f64 input ({conversion_context}): {err}"
                ))
            })?;
            let len = converted.len();
            write_wav_from_f64(
                &args.output,
                wav.channels,
                output_sample_rate as u32,
                &converted,
                selected_encoding,
            )
            .map_err(|err| {
                std::io::Error::other(format!(
                    "failed to write output WAV from f64 samples ({conversion_context}): {err}"
                ))
            })?;
            len
        }
    };

    println!(
        "Converted {} -> {} ({} Hz to {} Hz, {} samples to {} samples, {})",
        args.input.display(),
        args.output.display(),
        wav.sample_rate_hz,
        output_sample_rate,
        input_len,
        converted_len,
        display_encoding(selected_encoding)
    );
    Ok(())
}

fn resample_with_rubato_f32(
    samples: &[f32],
    channels: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
    chunk_size: usize,
    sub_chunks: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if channels == 0 {
        return Err("channel count must be greater than zero".into());
    }
    if input_sample_rate == 0 || output_sample_rate == 0 {
        return Err("sample rates must be greater than zero".into());
    }
    if samples.len() % channels != 0 {
        return Err(format!(
            "interleaved sample length {} is not divisible by channel count {}",
            samples.len(),
            channels
        )
        .into());
    }

    let mut resampler = Fft::<f32>::new(
        input_sample_rate,
        output_sample_rate,
        chunk_size,
        sub_chunks,
        channels,
        DEFAULT_FIXED_SYNC,
    )?;

    let input_frames = samples.len() / channels;
    let input_adapter = InterleavedSlice::new(samples, channels, input_frames)?;

    let output_frames_needed = resampler.process_all_needed_output_len(input_frames);
    let mut output = vec![0.0_f32; output_frames_needed * channels];
    let out_capacity_frames = output.len() / channels;
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, channels, out_capacity_frames)?;

    let (_, output_frames_written) =
        resampler.process_all_into_buffer(&input_adapter, &mut output_adapter, input_frames, None)?;
    output.truncate(output_frames_written * channels);
    Ok(output)
}

fn resample_with_rubato_f64(
    samples: &[f64],
    channels: usize,
    input_sample_rate: usize,
    output_sample_rate: usize,
    chunk_size: usize,
    sub_chunks: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    if channels == 0 {
        return Err("channel count must be greater than zero".into());
    }
    if input_sample_rate == 0 || output_sample_rate == 0 {
        return Err("sample rates must be greater than zero".into());
    }
    if samples.len() % channels != 0 {
        return Err(format!(
            "interleaved sample length {} is not divisible by channel count {}",
            samples.len(),
            channels
        )
        .into());
    }

    let mut resampler = Fft::<f64>::new(
        input_sample_rate,
        output_sample_rate,
        chunk_size,
        sub_chunks,
        channels,
        DEFAULT_FIXED_SYNC,
    )?;

    let input_frames = samples.len() / channels;
    let input_adapter = InterleavedSlice::new(samples, channels, input_frames)?;

    let output_frames_needed = resampler.process_all_needed_output_len(input_frames);
    let mut output = vec![0.0_f64; output_frames_needed * channels];
    let out_capacity_frames = output.len() / channels;
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, channels, out_capacity_frames)?;

    let (_, output_frames_written) =
        resampler.process_all_into_buffer(&input_adapter, &mut output_adapter, input_frames, None)?;
    output.truncate(output_frames_written * channels);
    Ok(output)
}

fn detect_output_encoding_from_source(source_encoding: &str) -> OutputEncoding {
    let source = source_encoding.to_ascii_lowercase();
    if source.contains("float64") {
        return OutputEncoding::Float64;
    }
    if source.contains("float32") {
        return OutputEncoding::Float32;
    }
    if source.contains("24") {
        return OutputEncoding::Int24;
    }
    if source.contains("16") {
        return OutputEncoding::Int16;
    }
    if source.contains("32") {
        return OutputEncoding::Int32;
    }
    OutputEncoding::Float32
}

struct WavData {
    samples: InputSamples,
    channels: usize,
    sample_rate_hz: u32,
    source_encoding_name: String,
}

enum InputSamples {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl InputSamples {
    fn len(&self) -> usize {
        match self {
            Self::F32(samples) => samples.len(),
            Self::F64(samples) => samples.len(),
        }
    }
}

fn read_wav(path: &PathBuf) -> Result<WavData, Box<dyn Error>> {
    let probe = Wav::<f32>::from_path(path)?;
    let channels = probe.n_channels() as usize;
    let sample_rate_hz = probe.sample_rate() as u32;
    let source_encoding = probe.encoding();
    let source_encoding_name = format!("{source_encoding:?}");
    drop(probe);

    let samples = match source_encoding {
        WavType::Float32 | WavType::EFloat32 => {
            let (samples, _) = read::<f32, _>(path)?;
            InputSamples::F32(samples.as_ref().to_vec())
        }
        WavType::Float64 | WavType::EFloat64 => {
            let (samples, _) = read::<f64, _>(path)?;
            InputSamples::F64(samples.as_ref().to_vec())
        }
        // Per request, integer source content is resampled as f64.
        _ => {
            let (samples, _) = read::<f64, _>(path)?;
            InputSamples::F64(samples.as_ref().to_vec())
        }
    };

    Ok(WavData {
        samples,
        channels,
        sample_rate_hz,
        source_encoding_name,
    })
}

fn write_wav_from_f32(
    path: &PathBuf,
    channels: usize,
    sample_rate_hz: u32,
    samples: &[f32],
    encoding: OutputEncoding,
) -> Result<(), Box<dyn Error>> {
    let sample_rate = sample_rate_hz as i32;
    let n_channels = channels as u16;
    match encoding {
        OutputEncoding::Int16 => {
            let out: Vec<i16> = samples
                .iter()
                .map(|sample| float32_to_i16(*sample))
                .collect();
            write::<i16, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Int24 => {
            let out: Vec<i24> = samples
                .iter()
                .map(|sample| float32_to_i24(*sample))
                .collect();
            write::<i24, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Int32 => {
            let out: Vec<i32> = samples
                .iter()
                .map(|sample| float32_to_i32(*sample))
                .collect();
            write::<i32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Float32 => {
            write::<f32, _>(path, samples, sample_rate, n_channels)?;
        }
        OutputEncoding::Float64 => {
            let out: Vec<f64> = samples.iter().map(|sample| *sample as f64).collect();
            write::<f64, _>(path, &out, sample_rate, n_channels)?;
        }
    }
    Ok(())
}

fn write_wav_from_f64(
    path: &PathBuf,
    channels: usize,
    sample_rate_hz: u32,
    samples: &[f64],
    encoding: OutputEncoding,
) -> Result<(), Box<dyn Error>> {
    let sample_rate = sample_rate_hz as i32;
    let n_channels = channels as u16;
    match encoding {
        OutputEncoding::Int16 => {
            let out: Vec<i16> = samples
                .iter()
                .map(|sample| float64_to_i16(*sample))
                .collect();
            write::<i16, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Int24 => {
            let out: Vec<i24> = samples
                .iter()
                .map(|sample| float64_to_i24(*sample))
                .collect();
            write::<i24, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Int32 => {
            let out: Vec<i32> = samples
                .iter()
                .map(|sample| float64_to_i32(*sample))
                .collect();
            write::<i32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Float32 => {
            let out: Vec<f32> = samples.iter().map(|sample| *sample as f32).collect();
            write::<f32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutputEncoding::Float64 => {
            write::<f64, _>(path, samples, sample_rate, n_channels)?;
        }
    }
    Ok(())
}

fn float32_to_i16(value: f32) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i16::MAX as f32).round() as i16
}

fn float64_to_i16(value: f64) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i16::MAX as f64).round() as i16
}

fn float32_to_i24(value: f32) -> i24 {
    const I24_MAX: f32 = ((1 << 23) - 1) as f32;
    let clamped = value.clamp(-1.0, 1.0);
    let as_i32 = (clamped * I24_MAX).round() as i32;
    i24::from_i32(as_i32)
}

fn float64_to_i24(value: f64) -> i24 {
    const I24_MAX: f64 = ((1 << 23) - 1) as f64;
    let clamped = value.clamp(-1.0, 1.0);
    let as_i32 = (clamped * I24_MAX).round() as i32;
    i24::from_i32(as_i32)
}

fn float32_to_i32(value: f32) -> i32 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i32::MAX as f32).round() as i32
}

fn float64_to_i32(value: f64) -> i32 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i32::MAX as f64).round() as i32
}

fn display_encoding(encoding: OutputEncoding) -> &'static str {
    match encoding {
        OutputEncoding::Int16 => "16-bit PCM",
        OutputEncoding::Int24 => "24-bit PCM",
        OutputEncoding::Int32 => "32-bit PCM",
        OutputEncoding::Float32 => "32-bit float",
        OutputEncoding::Float64 => "64-bit float",
    }
}
