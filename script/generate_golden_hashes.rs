#!/usr/bin/env rust-script

//! ```cargo
//! [dependencies]
//! md5 = "0.8"
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! wavers = "1.5.1"
//! ardftsrc = { path = "..", features = ["rayon"] }
//! ```

use ardftsrc::{Ardftsrc, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use wavers::{Wav, read};

const TARGET_RATES: [usize; 4] = [22_050, 44_100, 48_000, 96_000];
const OUTPUT_DIR: &str = "test_wavs";
const OUTPUT_BASENAME: &str = "golden_hashes";
const WAV_DIR: &str = "test_wavs";

#[derive(Clone)]
struct PresetSpec {
    name: &'static str,
    base: Config,
}

const PRESETS: [PresetSpec; 4] = [
    PresetSpec {
        name: "fast",
        base: PRESET_FAST,
    },
    PresetSpec {
        name: "good",
        base: PRESET_GOOD,
    },
    PresetSpec {
        name: "high",
        base: PRESET_HIGH,
    },
    PresetSpec {
        name: "extreme",
        base: PRESET_EXTREME,
    },
];

#[derive(Debug)]
struct WavInput {
    file_name: String,
    sample_rate: usize,
    channels: usize,
    samples: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct GoldenEntry {
    float_type: String,
    preset: String,
    target_rate: usize,
    hashes: BTreeMap<String, Vec<String>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let wav_paths = collect_wav_paths(Path::new(WAV_DIR))?;
    if wav_paths.is_empty() {
        return Err(format!("no WAV files found in '{WAV_DIR}'").into());
    }

    let wavs = wav_paths
        .iter()
        .map(|path| read_wav(path))
        .collect::<Result<Vec<_>, _>>()?;

    let mut entries = Vec::new();
    for target_rate in TARGET_RATES {
        for preset in PRESETS {
            let f32_hashes = generate_hashes(&wavs, preset.base.clone(), target_rate)?;
            entries.push(GoldenEntry {
                float_type: "f32".to_owned(),
                preset: preset.name.to_owned(),
                target_rate,
                hashes: f32_hashes,
            });

            let f64_hashes = generate_hashes_f64_from_f32(&wavs, preset.base.clone(), target_rate)?;
            entries.push(GoldenEntry {
                float_type: "f64".to_owned(),
                preset: preset.name.to_owned(),
                target_rate,
                hashes: f64_hashes,
            });
        }
    }

    let platform = std::env::consts::ARCH.to_owned();
    let output_path =format!("{OUTPUT_DIR}/{OUTPUT_BASENAME}.{platform}.json");
    let json = serde_json::to_string_pretty(&entries)?;
    fs::write(&output_path, json)?;
    println!("Wrote {}", output_path);

    Ok(())
}

fn collect_wav_paths(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
        })
        .collect::<Vec<_>>();

    paths.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
            .unwrap_or_default()
    });
    Ok(paths)
}

fn read_wav(path: &Path) -> Result<WavInput, Box<dyn Error>> {
    let wav = Wav::<f32>::from_path(path)?;
    let sample_rate = wav.sample_rate() as usize;
    let channels = wav.n_channels() as usize;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid UTF-8 file name: {}", path.display()))?
        .to_owned();
    drop(wav);

    let (samples, _) = read::<f32, _>(path)?;
    Ok(WavInput {
        file_name,
        sample_rate,
        channels,
        samples: samples.as_ref().to_vec(),
    })
}

fn generate_hashes(
    wavs: &[WavInput],
    preset_base: Config,
    target_rate: usize,
) -> Result<BTreeMap<String, Vec<String>>, Box<dyn Error>> {
    let mut grouped = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (index, wav) in wavs.iter().enumerate() {
        grouped
            .entry((wav.sample_rate, wav.channels))
            .or_default()
            .push(index);
    }

    let mut hashes = BTreeMap::new();
    for ((input_rate, channels), indices) in grouped {
        let config = preset_base
            .clone()
            .with_input_rate(input_rate)
            .with_output_rate(target_rate)
            .with_channels(channels);
        let driver = Ardftsrc::<f32>::new(config)?;
        let inputs = indices
            .iter()
            .map(|&i| wavs[i].samples.as_slice())
            .collect::<Vec<_>>();
        let outputs = driver.batch(&inputs)?;

        for (wav_index, output) in indices.into_iter().zip(outputs) {
            let channel_hashes = hash_interleaved_channels(&output, wavs[wav_index].channels)?;
            hashes.insert(wavs[wav_index].file_name.clone(), channel_hashes);
        }
    }

    Ok(hashes)
}

fn generate_hashes_f64_from_f32(
    wavs: &[WavInput],
    preset_base: Config,
    target_rate: usize,
) -> Result<BTreeMap<String, Vec<String>>, Box<dyn Error>> {
    let mut grouped = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (index, wav) in wavs.iter().enumerate() {
        grouped
            .entry((wav.sample_rate, wav.channels))
            .or_default()
            .push(index);
    }

    let mut hashes = BTreeMap::new();
    for ((input_rate, channels), indices) in grouped {
        let config = preset_base
            .clone()
            .with_input_rate(input_rate)
            .with_output_rate(target_rate)
            .with_channels(channels);
        let driver = Ardftsrc::<f64>::new(config)?;

        let inputs_f64 = indices
            .iter()
            .map(|&i| wavs[i].samples.iter().map(|&v| v as f64).collect::<Vec<f64>>())
            .collect::<Vec<_>>();
        let input_refs = inputs_f64.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let outputs_f64 = driver.batch(&input_refs)?;

        for (wav_index, output_f64) in indices.into_iter().zip(outputs_f64) {
            let output_f32 = output_f64.iter().map(|&v| v as f32).collect::<Vec<f32>>();
            let channel_hashes =
                hash_interleaved_channels(&output_f32, wavs[wav_index].channels)?;
            hashes.insert(wavs[wav_index].file_name.clone(), channel_hashes);
        }
    }

    Ok(hashes)
}

fn hash_interleaved_channels(
    interleaved: &[f32],
    channels: usize,
) -> Result<Vec<String>, Box<dyn Error>> {
    if channels == 0 {
        return Err("channels must be greater than zero".into());
    }
    if interleaved.len() % channels != 0 {
        return Err(format!(
            "interleaved sample length {} is not divisible by channels {}",
            interleaved.len(),
            channels
        )
        .into());
    }

    let frames = interleaved.len() / channels;
    let mut channel_hashes = Vec::with_capacity(channels);

    for channel in 0..channels {
        let mut bytes = Vec::with_capacity(frames * std::mem::size_of::<f32>());
        for frame in 0..frames {
            let sample = interleaved[frame * channels + channel];
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        channel_hashes.push(format!("{:x}", md5::compute(bytes)));
    }

    Ok(channel_hashes)
}

