#!/usr/bin/env rust-script

//! ```cargo
//! [dependencies]
//! clap = { version = "4.5", features = ["derive"] }
//! i24 = "1.0.1"
//! wavers = "1.5.1"
//!
//! [target.'cfg(any(target_arch = "x86_64"))'.dependencies]
//! ardftsrc = { path = "../ardftsrc", features = ["avx"] }
//!
//! [target.'cfg(any(target_arch = "aarch64"))'.dependencies]
//! ardftsrc = { path = "../ardftsrc", features = ["neon"] }
//! ```
//!
use ardftsrc::{Ardftsrc, Config};
use clap::Parser;
use i24::i24;
use std::error::Error;
use std::path::PathBuf;
use wavers::{Wav, WavType, read, write};

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
    #[arg(long, default_value_t = 16_384)]
    quality: usize,
    #[arg(long, default_value_t = 0.99)]
    bandwidth: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !(0.0..=1.0).contains(&args.bandwidth) {
        return Err("--bandwidth must be within 0.0..=1.0".into());
    }

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

    let mut config = Config {
        input_sample_rate: wav.sample_rate_hz as usize,
        output_sample_rate,
        channels: wav.channels,
        ..Config::default()
    };
    config.quality = args.quality;
    config.bandwidth = args.bandwidth;

    let effective_quality = config.quality;
    let effective_bandwidth = config.bandwidth;
    let input_len = wav.samples.len();
    let conversion_context = format!(
        "input='{}' output='{}' input_rate={} output_rate={} channels={} input_samples={} source_encoding={} output_encoding={} quality={} bandwidth={}",
        args.input.display(),
        args.output.display(),
        wav.sample_rate_hz,
        output_sample_rate,
        wav.channels,
        input_len,
        wav.source_encoding_name,
        display_encoding(selected_encoding),
        effective_quality,
        effective_bandwidth
    );
    let converted_len = match wav.samples {
        InputSamples::F32(samples) => {
            let mut resampler = Ardftsrc::<f32>::new(config).map_err(|err| {
                std::io::Error::other(format!(
                    "failed to initialize f32 resampler ({conversion_context}): {err}"
                ))
            })?;
            let converted = resampler.process_all(&samples).map_err(|err| {
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
            let mut resampler = Ardftsrc::<f64>::new(config).map_err(|err| {
                std::io::Error::other(format!(
                    "failed to initialize f64 resampler ({conversion_context}): {err}"
                ))
            })?;
            let converted = resampler.process_all(&samples).map_err(|err| {
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
        "Converted {} -> {} ({} Hz to {} Hz, {} samples to {} samples, {}, quality={}, bandwidth={})",
        args.input.display(),
        args.output.display(),
        wav.sample_rate_hz,
        output_sample_rate,
        input_len,
        converted_len,
        display_encoding(selected_encoding),
        effective_quality,
        effective_bandwidth
    );
    Ok(())
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
