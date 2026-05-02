use ardftsrc::{Ardftsrc, Config};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use wavers::{Wav, WavType, read};

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

fn collect_test_wav_paths() -> Vec<PathBuf> {
    let wav_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_wavs");
    let mut paths = fs::read_dir(&wav_dir)
        .unwrap_or_else(|err| panic!("failed to read test_wavs directory '{}': {err}", wav_dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let is_wav = path
                .extension()
                .and_then(OsStr::to_str)
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false);
            if is_wav { Some(path) } else { None }
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn resample_all(input: &[f32], input_rate: usize, output_rate: usize, channels: usize) -> Vec<f32> {
    let config = Config {
        input_sample_rate: input_rate,
        output_sample_rate: output_rate,
        channels,
        quality: 2048,
        bandwidth: 0.95,
        ..Config::default()
    };
    let mut resampler = Ardftsrc::<f32>::new(config).unwrap_or_else(|err| {
        panic!(
            "failed to initialize f32 resampler for {} -> {} Hz: {err}",
            input_rate, output_rate
        )
    });
    resampler
        .process_all(input)
        .unwrap_or_else(|err| panic!("resampling failed for {} -> {} Hz: {err}", input_rate, output_rate))
}

fn choose_upsample_rate(input_rate: usize) -> usize {
    if input_rate < 96_000 { 96_000 } else { input_rate * 2 }
}

fn choose_downsample_rate(input_rate: usize) -> usize {
    let target = input_rate / 2;
    target.max(8_000)
}

fn compare_with_edge_trim(original: &[f32], roundtrip: &[f32], channels: usize) -> (f32, f32, usize, usize, usize) {
    let original_frames = original.len() / channels;
    let roundtrip_frames = roundtrip.len() / channels;
    let compared_frames = original_frames.min(roundtrip_frames);
    let edge_trim_frames = (compared_frames / 100).max(64).min(compared_frames / 4);
    let start_frame = edge_trim_frames;
    let end_frame = compared_frames.saturating_sub(edge_trim_frames);
    assert!(
        end_frame > start_frame,
        "insufficient compared frames after edge trim (orig_frames={original_frames}, roundtrip_frames={roundtrip_frames}, trim={edge_trim_frames})"
    );

    let start_sample = start_frame * channels;
    let end_sample = end_frame * channels;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut count = 0usize;
    for (src, rt) in original[start_sample..end_sample]
        .iter()
        .zip(roundtrip[start_sample..end_sample].iter())
    {
        let abs_diff = (src - rt).abs();
        sum_abs += abs_diff;
        max_abs = max_abs.max(abs_diff);
        count += 1;
    }
    assert!(count > 0, "no samples compared");
    let mean_abs = sum_abs / count as f32;
    (mean_abs, max_abs, original_frames, roundtrip_frames, edge_trim_frames)
}

#[derive(Clone, Copy, Debug)]
struct RoundtripBaseline {
    up_down_mean_abs: f32,
    up_down_peak_abs: f32,
    down_up_mean_abs: f32,
    down_up_peak_abs: f32,
}

fn roundtrip_baseline_for_fixture(file_name: &str) -> RoundtripBaseline {
    match file_name {
        "example-pcm16-44k1-stereo.wav" => RoundtripBaseline {
            up_down_mean_abs: 1.6740085e-5,
            up_down_peak_abs: 8.653873e-5,
            down_up_mean_abs: 5.05882e-5,
            down_up_peak_abs: 0.0019121403,
        },
        "gaplesstest_m-f32-96k.wav" => RoundtripBaseline {
            up_down_mean_abs: 5.985025e-5,
            up_down_peak_abs: 0.0006804466,
            down_up_mean_abs: 2.2124574e-7,
            down_up_peak_abs: 4.6640635e-6,
        },
        "gaplesstest_s-f32-96k.wav" => RoundtripBaseline {
            up_down_mean_abs: 0.011628367,
            up_down_peak_abs: 0.1402759,
            down_up_mean_abs: 1.2932492e-6,
            down_up_peak_abs: 9.179115e-6,
        },
        "intermodulation_sine-32bitfloat.wav" => RoundtripBaseline {
            up_down_mean_abs: 6.9167115e-8,
            up_down_peak_abs: 4.172325e-7,
            down_up_mean_abs: 7.005609e-8,
            down_up_peak_abs: 4.7683716e-7,
        },
        "sweep-f32-96k.wav" => RoundtripBaseline {
            up_down_mean_abs: 1.2541253e-6,
            up_down_peak_abs: 1.013279e-5,
            down_up_mean_abs: 0.1313251,
            down_up_peak_abs: 0.5000018,
        },
        "sweep-pcm16-22k05.wav" => RoundtripBaseline {
            up_down_mean_abs: 1.0688594e-6,
            up_down_peak_abs: 5.3048134e-6,
            down_up_mean_abs: 0.03661642,
            down_up_peak_abs: 0.7577832,
        },
        other => panic!("no roundtrip baseline configured for fixture '{other}'"),
    }
}

fn assert_not_worse_than_baseline(
    measured: f32,
    baseline: f32,
    metric_name: &str,
    direction: &str,
    fixture_name: &str,
) {
    assert!(
        measured <= baseline,
        "{direction} {metric_name} regression for '{fixture_name}': measured={measured}, baseline={baseline}"
    );
}

fn assert_all_finite(samples: &[f32], stage: &str, fixture_path: &Path) {
    assert!(
        samples.iter().all(|sample| sample.is_finite()),
        "non-finite sample found in {stage} output for '{}'",
        fixture_path.display()
    );
}

#[test]
fn test_wavs_f32_2048_bw095_outputs_are_finite() {
    let wav_paths = collect_test_wav_paths();
    assert!(!wav_paths.is_empty(), "expected at least one WAV in test_wavs/");

    for input_path in wav_paths {
        let (input_samples, input_sample_rate, channels) = read_wav_as_f32(&input_path);
        let config = Config {
            input_sample_rate,
            output_sample_rate: 44_100,
            channels,
            quality: 2048,
            bandwidth: 0.95,
            ..Config::default()
        };
        let mut resampler = Ardftsrc::<f32>::new(config).unwrap_or_else(|err| {
            panic!(
                "failed to initialize f32 resampler for '{}': {err}",
                input_path.display()
            )
        });
        let output_samples = resampler
            .process_all(&input_samples)
            .unwrap_or_else(|err| panic!("resampling failed for '{}': {err}", input_path.display()));

        assert_all_finite(&output_samples, "single-pass", &input_path);
    }
}

#[test]
fn test_wavs_roundtrip_up_then_down() {
    let wav_paths = collect_test_wav_paths();
    assert!(!wav_paths.is_empty(), "expected at least one WAV in test_wavs/");

    for input_path in wav_paths {
        let fixture_name = input_path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("invalid fixture file name for '{}'", input_path.display()));
        let baseline = roundtrip_baseline_for_fixture(fixture_name);
        let (input_samples, input_sample_rate, channels) = read_wav_as_f32(&input_path);
        let up_rate = choose_upsample_rate(input_sample_rate);

        let upsampled = resample_all(&input_samples, input_sample_rate, up_rate, channels);
        assert_all_finite(&upsampled, "upsample", &input_path);
        let roundtrip_up_down = resample_all(&upsampled, up_rate, input_sample_rate, channels);
        assert_all_finite(&roundtrip_up_down, "roundtrip up->down", &input_path);
        let (mean_abs_up_down, peak_abs_up_down, in_frames_ud, out_frames_ud, trim_ud) =
            compare_with_edge_trim(&input_samples, &roundtrip_up_down, channels);

        eprintln!(
            "roundtrip up->down '{}' sr={}=>{}=>{} ch={} frames_in={} frames_out={} trim={} mean_abs={} peak_abs={}",
            input_path.display(),
            input_sample_rate,
            up_rate,
            input_sample_rate,
            channels,
            in_frames_ud,
            out_frames_ud,
            trim_ud,
            mean_abs_up_down,
            peak_abs_up_down
        );
        assert_not_worse_than_baseline(
            mean_abs_up_down,
            baseline.up_down_mean_abs,
            "mean_abs",
            "up->down",
            fixture_name,
        );
        assert_not_worse_than_baseline(
            peak_abs_up_down,
            baseline.up_down_peak_abs,
            "peak_abs",
            "up->down",
            fixture_name,
        );
    }
}

#[test]
fn test_wavs_roundtrip_down_then_up() {
    let wav_paths = collect_test_wav_paths();
    assert!(!wav_paths.is_empty(), "expected at least one WAV in test_wavs/");

    for input_path in wav_paths {
        let fixture_name = input_path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("invalid fixture file name for '{}'", input_path.display()));
        let baseline = roundtrip_baseline_for_fixture(fixture_name);
        let (input_samples, input_sample_rate, channels) = read_wav_as_f32(&input_path);
        let down_rate = choose_downsample_rate(input_sample_rate);

        let downsampled = resample_all(&input_samples, input_sample_rate, down_rate, channels);
        assert_all_finite(&downsampled, "downsample", &input_path);
        let roundtrip_down_up = resample_all(&downsampled, down_rate, input_sample_rate, channels);
        assert_all_finite(&roundtrip_down_up, "roundtrip down->up", &input_path);
        let (mean_abs_down_up, peak_abs_down_up, in_frames_du, out_frames_du, trim_du) =
            compare_with_edge_trim(&input_samples, &roundtrip_down_up, channels);

        eprintln!(
            "roundtrip down->up '{}' sr={}=>{}=>{} ch={} frames_in={} frames_out={} trim={} mean_abs={} peak_abs={}",
            input_path.display(),
            input_sample_rate,
            down_rate,
            input_sample_rate,
            channels,
            in_frames_du,
            out_frames_du,
            trim_du,
            mean_abs_down_up,
            peak_abs_down_up
        );
        assert_not_worse_than_baseline(
            mean_abs_down_up,
            baseline.down_up_mean_abs,
            "mean_abs",
            "down->up",
            fixture_name,
        );
        assert_not_worse_than_baseline(
            peak_abs_down_up,
            baseline.down_up_peak_abs,
            "peak_abs",
            "down->up",
            fixture_name,
        );
    }
}
