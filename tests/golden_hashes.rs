//!
//! The golden_hashes test validates resampler determinism against checked-in
//! golden outputs in test_wavs/golden_hashes.<arch>json. It is intended to catch
//! unintended behavior changes.
//!
//! Run it with:
//!
//! cargo test --release --features=rayon golden_hashes -- --nocapture
//!
//! To regenerate test_wavs/golden_hashes.<arch>.json:
//!
//! rust-script script/genreate_golden_hashes.rs
//!
//! Updates to test_wavs/golden_hashes.<arch>.json are allowed, but only when accompanied
//! by verifiable quality improvements demonstrated with the HydrogenAudio SRC
//! test suite.
//! 
//! TODO: Generate golden hashes for x86_64 and remove the target_arch guard from test.
//! 
use ardftsrc::{Ardftsrc, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use wavers::{Wav, read};

const GOLDEN_HASHES_PATH: &str = "test_wavs/golden_hashes.aarch64.json";
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

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GoldenEntry {
    float_type: String,
    preset: String,
    target_rate: usize,
    hashes: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum GoldenHashesFileFormat {
    PlatformTagged {
        platform: String,
        entries: Vec<GoldenEntry>,
    },
    Legacy(Vec<GoldenEntry>),
}

#[test]
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        not(debug_assertions),
        not(feature = "avx"),
        not(feature = "neon"),
        not(feature = "sse"),
        not(feature = "wasm_simd")
    )),
    ignore = "requires release-mode aarch64 with no SIMD features enabled"
)]
fn golden_hashes_match() -> Result<(), Box<dyn Error>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let wav_dir = manifest_dir.join(WAV_DIR);
    let golden_path = manifest_dir.join(GOLDEN_HASHES_PATH);

    let wav_paths = collect_wav_paths(&wav_dir)?;
    if wav_paths.is_empty() {
        return Err(format!("no WAV files found in '{}'", wav_dir.display()).into());
    }

    let wavs = wav_paths
        .iter()
        .map(|path| read_wav(path))
        .collect::<Result<Vec<_>, _>>()?;

    let golden_json = fs::read_to_string(&golden_path)?;
    let golden_entries = parse_golden_entries(&golden_json)?;

    if golden_entries.is_empty() {
        return Err(format!("no entries found in '{}'", golden_path.display()).into());
    }

    for entry in golden_entries {
        let preset_base = preset_config_for_name(&entry.preset)?;
        let actual_hashes = match entry.float_type.as_str() {
            "f32" => generate_hashes_f32(&wavs, preset_base, entry.target_rate)?,
            "f64" => generate_hashes_f64_from_f32(&wavs, preset_base, entry.target_rate)?,
            other => {
                return Err(format!("unsupported float_type in golden hashes: '{other}'").into());
            }
        };

        assert_eq!(
            entry.hashes, actual_hashes,
            "hash mismatch for float_type='{}', preset='{}', target_rate={}",
            entry.float_type, entry.preset, entry.target_rate
        );
    }

    Ok(())
}

fn parse_golden_entries(golden_json: &str) -> Result<Vec<GoldenEntry>, Box<dyn Error>> {
    let parsed: GoldenHashesFileFormat = serde_json::from_str(golden_json)?;
    let entries = match parsed {
        GoldenHashesFileFormat::PlatformTagged { platform, entries } => {
            let expected_platform = std::env::consts::ARCH;
            if platform != expected_platform {
                return Err(format!(
                    "golden hash platform mismatch: expected '{expected_platform}', found '{platform}'"
                )
                .into());
            }
            entries
        }
        GoldenHashesFileFormat::Legacy(entries) => entries,
    };
    Ok(entries)
}

fn preset_config_for_name(name: &str) -> Result<Config, Box<dyn Error>> {
    PRESETS
        .iter()
        .find(|preset| preset.name == name)
        .map(|preset| preset.base.clone())
        .ok_or_else(|| format!("unknown preset in golden hashes: '{name}'").into())
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

fn generate_hashes_f32(
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
