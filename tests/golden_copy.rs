use ardftsrc::{Ardftsrc, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use wavers::{Wav, WavType, read};

#[derive(Debug, Deserialize)]
struct GoldenManifestEntry {
    float_type: String,
    sample_wav: String,
    preset: String,
    target_rate: usize,
    pcm_md5_by_channel: Vec<String>,
}

fn preset_from_name(name: &str) -> Config {
    match name {
        "fast" => PRESET_FAST,
        "good" => PRESET_GOOD,
        "high" => PRESET_HIGH,
        "extreme" => PRESET_EXTREME,
        other => panic!("unknown preset '{other}'"),
    }
}

fn read_wav_as_f32(path: &Path) -> (Vec<f32>, usize, usize) {
    let wav =
        Wav::<f32>::from_path(path).unwrap_or_else(|err| panic!("failed to open WAV '{}': {err}", path.display()));
    let channels = wav.n_channels() as usize;
    let sample_rate = wav.sample_rate() as usize;
    let source_encoding = wav.encoding();
    drop(wav);

    let samples = match source_encoding {
        WavType::Float64 | WavType::EFloat64 => {
            let (samples, _) = read::<f64, _>(path)
                .unwrap_or_else(|err| panic!("failed reading f64 WAV samples from '{}': {err}", path.display()));
            samples.as_ref().iter().map(|sample| *sample as f32).collect()
        }
        _ => {
            let (samples, _) = read::<f32, _>(path)
                .unwrap_or_else(|err| panic!("failed reading f32 WAV samples from '{}': {err}", path.display()));
            samples.as_ref().to_vec()
        }
    };

    (samples, sample_rate, channels)
}

fn pcm_md5_by_channel(samples: &[f32], channels: usize) -> Vec<String> {
    assert!(channels > 0, "channels must be > 0 for per-channel hashing");

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

#[test]
#[cfg_attr(
    any(feature = "avx", feature = "neon", feature = "sse", feature = "wasm_simd"),
    ignore = "wav_golden_copy is scalar-only; run without SIMD features"
)]
#[cfg_attr(
    debug_assertions,
    ignore = "wav_golden_copy is intended for release-mode determinism checks; only run in --release"
)]
fn wav_golden_copy() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest_path = root.join("test_wavs").join("golden_hashes.json");
    let manifest_text = fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
        panic!(
            "failed to read golden manifest '{}': {err}",
            manifest_path.display()
        )
    });
    let entries: Vec<GoldenManifestEntry> = serde_json::from_str(&manifest_text).unwrap_or_else(|err| {
        panic!(
            "failed to parse golden manifest '{}': {err}",
            manifest_path.display()
        )
    });
    assert!(
        !entries.is_empty(),
        "golden manifest '{}' has no entries",
        manifest_path.display()
    );

    let mut failed = false;

    for entry in entries {
        let source_path = root.join("test_wavs").join(&entry.sample_wav);
        let (input_samples, input_rate, channels) = read_wav_as_f32(&source_path);
        if entry.pcm_md5_by_channel.len() != channels {
            failed = true;
            eprintln!(
                "manifest channel hash count mismatch: sample='{}', expected={}, actual={}",
                entry.sample_wav,
                channels,
                entry.pcm_md5_by_channel.len()
            );
            continue;
        }

        let config = preset_from_name(&entry.preset)
            .with_input_rate(input_rate)
            .with_output_rate(entry.target_rate)
            .with_channels(channels);

        let output_samples_f32: Vec<f32> = match entry.float_type.as_str() {
            "f32" => {
                let mut resampler = Ardftsrc::<f32>::new(config).unwrap_or_else(|err| {
                    panic!(
                        "failed to initialize f32 resampler for '{}', preset='{}', rate={}: {err}",
                        entry.sample_wav, entry.preset, entry.target_rate
                    )
                });
                resampler.process_all(&input_samples).unwrap_or_else(|err| {
                    panic!(
                        "f32 resample failed for '{}', preset='{}', rate={}: {err}",
                        entry.sample_wav, entry.preset, entry.target_rate
                    )
                })
            }
            "f64" => {
                let mut resampler = Ardftsrc::<f64>::new(config).unwrap_or_else(|err| {
                    panic!(
                        "failed to initialize f64 resampler for '{}', preset='{}', rate={}: {err}",
                        entry.sample_wav, entry.preset, entry.target_rate
                    )
                });
                let input_f64: Vec<f64> = input_samples.iter().map(|sample| *sample as f64).collect();
                let output_f64 = resampler.process_all(&input_f64).unwrap_or_else(|err| {
                    panic!(
                        "f64 resample failed for '{}', preset='{}', rate={}: {err}",
                        entry.sample_wav, entry.preset, entry.target_rate
                    )
                });
                output_f64.iter().map(|sample| *sample as f32).collect()
            }
            other => panic!("unknown float_type '{other}' in '{}'", manifest_path.display()),
        };

        let actual_hashes = pcm_md5_by_channel(&output_samples_f32, channels);
        if actual_hashes != entry.pcm_md5_by_channel {
            failed = true;
            eprintln!(
                "golden per-channel PCM hash mismatch: sample='{}', preset='{}', rate={}, float_type='{}', actual={:?}, expected={:?}",
                entry.sample_wav,
                entry.preset,
                entry.target_rate,
                entry.float_type,
                actual_hashes,
                entry.pcm_md5_by_channel
            );
        }
    }

    assert!(
        !failed,
        "wav_golden_copy found one or more golden mismatches; see stderr output above"
    );
}
