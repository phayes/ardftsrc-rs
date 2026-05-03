use ardftsrc::{ArdftsrcCore, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use num_traits::Float;
use realfft::FftNum;
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

fn deinterleave_channel(samples: &[f32], channels: usize, channel: usize) -> Vec<f32> {
    samples
        .chunks_exact(channels)
        .map(|channel_samples| channel_samples[channel])
        .collect()
}

fn pcm_md5(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    format!("{:x}", md5::compute(bytes))
}

fn process_channel_streaming<T>(input: &[T], config: Config) -> Vec<T>
where
    T: Float + FftNum,
{
    let mut resampler = ArdftsrcCore::<T>::new(config).unwrap_or_else(|err| {
        panic!("failed to initialize core streaming resampler: {err}");
    });

    let input_chunk = resampler.input_buffer_size();
    let output_chunk = resampler.output_buffer_size();
    let mut output = Vec::new();
    let mut chunk_output_buffer = vec![T::zero(); output_chunk];
    let mut offset = 0;

    while offset + input_chunk <= input.len() {
        let written = resampler
            .process_chunk(&input[offset..offset + input_chunk], &mut chunk_output_buffer)
            .unwrap_or_else(|err| panic!("process_chunk failed: {err}"));
        output.extend_from_slice(&chunk_output_buffer[..written]);
        offset += input_chunk;
    }

    let written = resampler
        .process_chunk_final(&input[offset..], &mut chunk_output_buffer)
        .unwrap_or_else(|err| panic!("process_chunk_final failed: {err}"));
    output.extend_from_slice(&chunk_output_buffer[..written]);

    let written = resampler
        .finalize(&mut chunk_output_buffer)
        .unwrap_or_else(|err| panic!("finalize failed: {err}"));
    output.extend_from_slice(&chunk_output_buffer[..written]);

    output
}

#[test]
#[cfg_attr(
    any(feature = "avx", feature = "neon", feature = "sse", feature = "wasm_simd"),
    ignore = "wav_golden_copy_core is scalar-only; run without SIMD features"
)]
#[cfg_attr(
    debug_assertions,
    ignore = "wav_golden_copy_core is intended for release-mode determinism checks; only run in --release"
)]
fn wav_golden_copy_core() {
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

        let mut actual_hashes = Vec::with_capacity(channels);
        for channel in 0..channels {
            let channel_input_f32 = deinterleave_channel(&input_samples, channels, channel);
            let core_config = preset_from_name(&entry.preset)
                .with_input_rate(input_rate)
                .with_output_rate(entry.target_rate)
                .with_channels(1);

            let output_channel_f32: Vec<f32> = match entry.float_type.as_str() {
                "f32" => process_channel_streaming(&channel_input_f32, core_config),
                "f64" => {
                    let input_f64: Vec<f64> = channel_input_f32.iter().map(|sample| *sample as f64).collect();
                    process_channel_streaming(&input_f64, core_config)
                        .into_iter()
                        .map(|sample| sample as f32)
                        .collect()
                }
                other => panic!("unknown float_type '{other}' in '{}'", manifest_path.display()),
            };

            actual_hashes.push(pcm_md5(&output_channel_f32));
        }

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
        "wav_golden_copy_core found one or more golden mismatches; see stderr output above"
    );
}
