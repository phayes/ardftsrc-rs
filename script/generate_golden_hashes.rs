#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! clap = { version = "4.5", features = ["derive"] }
//! md5 = "0.8"
//! serde_json = "1.0"
//! wavers = "1.5.1"
//! ardftsrc = { path = ".." }
//! ```
//!
//! Batch-resample all top-level WAV files in test_wavs and write a
//! deterministic hash manifest to test_wavs/golden_hashes.json.

use ardftsrc::{Ardftsrc, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use clap::Parser;
use serde_json::json;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use wavers::{Wav, WavType, read};

#[derive(Debug, Parser)]
#[command(name = "generate_golden_hashes")]
struct Args {
    /// Directory with source WAV files.
    #[arg(long, default_value = "test_wavs")]
    input_dir: PathBuf,

    /// Output sample rates in Hz.
    #[arg(long, value_delimiter = ',', default_value = "22050,44100,48000,96000")]
    rates: Vec<usize>,

}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_rates(&args.rates)?;

    clear_previous_outputs(&args.input_dir)?;
    let wav_files = collect_top_level_wavs(&args.input_dir)?;

    if wav_files.is_empty() {
        return Err(format!(
            "no .wav files found in '{}'",
            args.input_dir.display()
        )
        .into());
    }

    println!("Input dir: {}", args.input_dir.display());
    println!("Rates: {:?}", args.rates);
    println!("Presets: fast, good, high, extreme");
    let mut hash_entries = Vec::new();

    for rate in &args.rates {
        for (preset_name, preset) in [
            ("fast", PRESET_FAST),
            ("good", PRESET_GOOD),
            ("high", PRESET_HIGH),
            ("extreme", PRESET_EXTREME),
        ] {
            for float_type in ["f32", "f64"] {
                for source_path in &wav_files {
                    let file_name = source_path
                        .file_name()
                        .ok_or("failed to read source filename")?;

                    let input = read_wav_f32(source_path)?;
                    let config = preset
                        .clone()
                        .with_input_rate(input.sample_rate_hz as usize)
                        .with_output_rate(*rate)
                        .with_channels(input.channels);

                    let converted_f32 = if float_type == "f32" {
                        let mut resampler = Ardftsrc::<f32>::new(config)?;
                        resampler.process_all(&input.samples)?
                    } else {
                        let mut resampler = Ardftsrc::<f64>::new(config)?;
                        let input_f64: Vec<f64> = input.samples.iter().map(|sample| *sample as f64).collect();
                        let converted = resampler.process_all(&input_f64)?;
                        converted.iter().map(|sample| *sample as f32).collect()
                    };
                    let output_md5 = hash_interleaved_pcm_f32(&converted_f32);
                    let pcm_md5_by_channel = hash_pcm_by_channel(&converted_f32, input.channels);
                    hash_entries.push(json!({
                        "float_type": float_type,
                        "sample_wav": file_name.to_string_lossy(),
                        "preset": preset_name,
                        "target_rate": *rate,
                        "md5": output_md5,
                        "pcm_md5_by_channel": pcm_md5_by_channel,
                    }));

                    println!(
                        "Processed {} ({} Hz -> {} Hz, preset={}, path={})",
                        source_path.display(),
                        input.sample_rate_hz,
                        rate,
                        preset_name,
                        float_type
                    );
                }
            }
        }
    }

    let hash_manifest_path = args.input_dir.join("golden_hashes.json");
    let hash_manifest_json = serde_json::to_string_pretty(&hash_entries)?;
    fs::write(&hash_manifest_path, format!("{hash_manifest_json}\n"))?;
    println!("Wrote {}", hash_manifest_path.display());

    Ok(())
}

fn clear_previous_outputs(input_dir: &Path) -> Result<(), Box<dyn Error>> {
    let hash_manifest_path = input_dir.join("golden_hashes.json");
    if hash_manifest_path.exists() {
        fs::remove_file(hash_manifest_path)?;
    }

    Ok(())
}

fn validate_rates(rates: &[usize]) -> Result<(), Box<dyn Error>> {
    if rates.is_empty() {
        return Err("at least one rate must be provided".into());
    }
    if let Some(zero_rate) = rates.iter().find(|rate| **rate == 0) {
        return Err(format!("invalid rate '{}': must be > 0", zero_rate).into());
    }
    Ok(())
}

fn collect_top_level_wavs(input_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut wavs = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("wav"))
            .unwrap_or(false)
        {
            wavs.push(path);
        }
    }
    wavs.sort();
    Ok(wavs)
}

struct WavData {
    samples: Vec<f32>,
    channels: usize,
    sample_rate_hz: u32,
}

fn read_wav_f32(path: &Path) -> Result<WavData, Box<dyn Error>> {
    let probe = Wav::<f32>::from_path(path)?;
    let channels = probe.n_channels() as usize;
    let sample_rate_hz = probe.sample_rate() as u32;
    let source_encoding = probe.encoding();
    drop(probe);

    let samples = match source_encoding {
        WavType::Float64 | WavType::EFloat64 => {
            let (samples, _) = read::<f64, _>(path)?;
            samples.as_ref().iter().map(|sample| *sample as f32).collect()
        }
        _ => {
            let (samples, _) = read::<f32, _>(path)?;
            samples.as_ref().to_vec()
        }
    };

    Ok(WavData {
        samples,
        channels,
        sample_rate_hz,
    })
}

fn hash_pcm_by_channel(samples: &[f32], channels: usize) -> Vec<String> {
    let frames = samples.len() / channels;
    let mut hashes = Vec::with_capacity(channels);

    for ch in 0..channels {
        let mut bytes = Vec::with_capacity(frames * std::mem::size_of::<f32>());
        for frame in 0..frames {
            let sample = samples[frame * channels + ch];
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        hashes.push(format!("{:x}", md5::compute(bytes)));
    }

    hashes
}

fn hash_interleaved_pcm_f32(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    format!("{:x}", md5::compute(bytes))
}
