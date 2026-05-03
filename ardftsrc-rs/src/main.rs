use ardftsrc::{Ardftsrc, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
use clap::{Parser, ValueEnum};
use flac_codec::decode::FlacSampleReader;
use flac_codec::encode::{FlacSampleWriter, Options as FlacOptions};
use flac_codec::metadata::Metadata;
use i24::i24;
use mimalloc::MiMalloc;
use std::error::Error;
use std::path::{Path, PathBuf};
use wavers::{Wav, WavType, read, write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    /// Fastest preset; lowest quality (quality = 512, bandwidth = 0.8323).
    Fast,
    /// Balanced quality/speed preset (quality = 2048, bandwidth = 0.95).
    Good,
    /// High quality preset for offline or quality-sensitive use (quality = 65536, bandwidth = 0.97).
    High,
    /// Maximum quality preset; slowest (quality = 524288, bandwidth = 0.9932).
    Extreme,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TaperTypeArg {
    /// Planck taper transition.
    Planck,
    /// Sigmoid-warped cosine taper transition.
    Cosine,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutFormatArg {
    /// Match the source format when possible.
    Same,
    /// 16-bit signed integer PCM.
    I16,
    /// 24-bit signed integer PCM.
    I24,
    /// 32-bit signed integer PCM.
    I32,
    /// 32-bit floating point PCM (WAV only).
    F32,
    /// 64-bit floating point PCM (WAV only).
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioContainer {
    Wav,
    Flac,
}

#[derive(Debug, Parser)]
#[command(name = "ardftsrc-rs")]
#[command(about = "General-purpose wav and flac sample-rate converter powered by ardftsrc")]
struct Args {
    /// One or more input audio paths (.wav or .flac).
    #[arg(long = "input", required = true)]
    input: Vec<PathBuf>,

    /// One or more output audio paths (.wav or .flac), matching --input count.
    #[arg(long = "output", required = true)]
    output: Vec<PathBuf>,

    /// Target sample rate in Hz.
    #[arg(long = "output-rate")]
    output_rate: usize,

    /// Resampler quality preset.
    #[arg(long, value_enum, default_value_t = PresetArg::High)]
    preset: PresetArg,

    /// Override preset quality (higher is slower, typically higher quality). Useful values are between 512 and 524288.
    #[arg(long)]
    quality: Option<usize>,

    /// Override normalized low-pass bandwidth in [0.0, 1.0]. Useful values are between 0.8 and 0.99.
    #[arg(long)]
    bandwidth: Option<f32>,

    /// Cosine taper alpha (cannot be used with --taper-type planck). Higher values are sharper cutoff; lower values are smoother. Useful values are between 1 and 4.
    #[arg(long)]
    alpha: Option<f32>,

    /// Transition taper profile.
    #[arg(long = "taper-type", value_enum)]
    taper_type: Option<TaperTypeArg>,

    /// Output sample format. For .flac output, float formats are rejected.
    #[arg(long = "out-format", value_enum, default_value_t = OutFormatArg::Same)]
    out_format: OutFormatArg,

    /// Use gapless batch mode to preserve adjacent-track context.
    #[arg(long)]
    gapless: bool,
}

#[derive(Debug)]
struct InputTrack {
    path: PathBuf,
    samples_f64: Vec<f64>,
    channels: usize,
    input_rate_hz: usize,
    source_out_format: OutFormatArg,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    let tracks = args
        .input
        .iter()
        .map(|path| read_input_as_f64(path))
        .collect::<Result<Vec<_>, _>>()?;

    validate_batch_compatibility(&tracks)?;

    let first = tracks.first().ok_or("at least one --input is required")?;
    let config = build_config(&args, first.input_rate_hz, first.channels)?;
    let processor = Ardftsrc::<f64>::new(config)?;

    let input_refs = tracks
        .iter()
        .map(|track| track.samples_f64.as_slice())
        .collect::<Vec<_>>();
    let converted = if args.gapless {
        processor.batch_gapless(&input_refs)?
    } else {
        processor.batch(&input_refs)?
    };

    for ((track, output_path), converted_samples) in
        tracks.iter().zip(args.output.iter()).zip(converted.into_iter())
    {
        write_output_audio(
            output_path,
            first.channels,
            args.output_rate as u32,
            &converted_samples,
            args.out_format,
            track.source_out_format,
        )?;
    }

    Ok(())
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.input.len() != args.output.len() {
        return Err(format!(
            "--input count ({}) must match --output count ({})",
            args.input.len(),
            args.output.len()
        )
        .into());
    }
    if args.output_rate == 0 {
        return Err("--output-rate must be greater than zero".into());
    }
    for path in &args.input {
        if audio_container(path).is_none() {
            return Err(format!(
                "unsupported input extension for {} (supported: .wav, .flac)",
                path.display()
            )
            .into());
        }
    }
    for path in &args.output {
        if audio_container(path).is_none() {
            return Err(format!(
                "unsupported output extension for {} (supported: .wav, .flac)",
                path.display()
            )
            .into());
        }
    }
    if matches!(args.taper_type, Some(TaperTypeArg::Planck)) && args.alpha.is_some() {
        return Err("--alpha cannot be used with --taper-type=planck".into());
    }
    Ok(())
}

fn validate_batch_compatibility(tracks: &[InputTrack]) -> Result<(), Box<dyn Error>> {
    if tracks.is_empty() {
        return Err("at least one input track is required".into());
    }
    let reference = &tracks[0];
    for track in tracks {
        if track.channels != reference.channels {
            return Err(format!(
                "all inputs must have the same channel count for batch processing ({} has {}, expected {})",
                track.path.display(),
                track.channels,
                reference.channels
            )
            .into());
        }
        if track.input_rate_hz != reference.input_rate_hz {
            return Err(format!(
                "all inputs must have the same input sample rate for batch processing ({} has {}, expected {})",
                track.path.display(),
                track.input_rate_hz,
                reference.input_rate_hz
            )
            .into());
        }
    }
    Ok(())
}

fn build_config(
    args: &Args,
    input_sample_rate: usize,
    channels: usize,
) -> Result<Config, Box<dyn Error>> {
    let mut config = match args.preset {
        PresetArg::Fast => PRESET_FAST,
        PresetArg::Good => PRESET_GOOD,
        PresetArg::High => PRESET_HIGH,
        PresetArg::Extreme => PRESET_EXTREME,
    }
    .with_input_rate(input_sample_rate)
    .with_output_rate(args.output_rate)
    .with_channels(channels);

    if let Some(quality) = args.quality {
        config.quality = quality;
    }
    if let Some(bandwidth) = args.bandwidth {
        config.bandwidth = bandwidth;
    }

    if let Some(taper_type) = args.taper_type {
        config.taper_type = match taper_type {
            TaperTypeArg::Planck => TaperType::Planck,
            TaperTypeArg::Cosine => {
                let alpha = args.alpha.unwrap_or_else(|| preset_alpha(config.taper_type));
                TaperType::Cosine(alpha)
            }
        };
    } else if let Some(alpha) = args.alpha {
        config.taper_type = TaperType::Cosine(alpha);
    }

    config.validate()?;
    Ok(config)
}

fn preset_alpha(taper_type: TaperType) -> f32 {
    match taper_type {
        TaperType::Cosine(alpha) => alpha,
        TaperType::Planck => 3.4375,
    }
}

fn read_input_as_f64(path: &Path) -> Result<InputTrack, Box<dyn Error>> {
    match audio_container(path) {
        Some(AudioContainer::Wav) => read_wav_as_f64(path),
        Some(AudioContainer::Flac) => read_flac_as_f64(path),
        None => Err(format!(
            "unsupported input extension for {} (supported: .wav, .flac)",
            path.display()
        )
        .into()),
    }
}

fn read_wav_as_f64(path: &Path) -> Result<InputTrack, Box<dyn Error>> {
    let probe = Wav::<f32>::from_path(path)?;
    let channels = probe.n_channels() as usize;
    let input_rate_hz = probe.sample_rate() as usize;
    let source_format = probe.encoding();
    drop(probe);

    let (samples, _) = read::<f64, _>(path)?;
    Ok(InputTrack {
        path: path.to_path_buf(),
        samples_f64: samples.as_ref().to_vec(),
        channels,
        input_rate_hz,
        source_out_format: wav_source_to_out_format(source_format),
    })
}

fn read_flac_as_f64(path: &Path) -> Result<InputTrack, Box<dyn Error>> {
    let mut reader = FlacSampleReader::open(path)?;
    let channels = reader.channel_count() as usize;
    let input_rate_hz = reader.sample_rate() as usize;
    let bits_per_sample = reader.bits_per_sample();
    let source_out_format = flac_bits_to_out_format(bits_per_sample);
    let scale = pcm_scale_from_bits(bits_per_sample)?;

    let mut samples_i32 = Vec::<i32>::new();
    reader.read_to_end(&mut samples_i32)?;
    let samples_f64 = samples_i32
        .into_iter()
        .map(|sample| (sample as f64 / scale).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();

    Ok(InputTrack {
        path: path.to_path_buf(),
        samples_f64,
        channels,
        input_rate_hz,
        source_out_format,
    })
}

fn write_output_audio(
    path: &Path,
    channels: usize,
    output_rate_hz: u32,
    samples_f64: &[f64],
    out_format: OutFormatArg,
    source_out_format: OutFormatArg,
) -> Result<(), Box<dyn Error>> {
    let target_format = match out_format {
        OutFormatArg::Same => source_out_format,
        OutFormatArg::I16 => OutFormatArg::I16,
        OutFormatArg::I24 => OutFormatArg::I24,
        OutFormatArg::I32 => OutFormatArg::I32,
        OutFormatArg::F32 => OutFormatArg::F32,
        OutFormatArg::F64 => OutFormatArg::F64,
    };

    let aligned_len = samples_f64.len() - (samples_f64.len() % channels.max(1));
    let aligned_samples = &samples_f64[..aligned_len];

    match audio_container(path) {
        Some(AudioContainer::Wav) => write_output_wav(path, channels, output_rate_hz, aligned_samples, target_format),
        Some(AudioContainer::Flac) => {
            write_output_flac(path, channels, output_rate_hz, aligned_samples, target_format)
        }
        None => Err(format!(
            "unsupported output extension for {} (supported: .wav, .flac)",
            path.display()
        )
        .into()),
    }
}

fn write_output_wav(
    path: &Path,
    channels: usize,
    output_rate_hz: u32,
    samples_f64: &[f64],
    target_format: OutFormatArg,
) -> Result<(), Box<dyn Error>> {
    let sample_rate = output_rate_hz as i32;
    let n_channels = channels as u16;
    match target_format {
        OutFormatArg::Same => unreachable!("same is resolved to a concrete format"),
        OutFormatArg::I16 => {
            let out = samples_f64
                .iter()
                .copied()
                .map(float64_to_i16)
                .collect::<Vec<_>>();
            write::<i16, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::I24 => {
            let out = samples_f64
                .iter()
                .copied()
                .map(float64_to_i24)
                .collect::<Vec<_>>();
            write::<i24, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::I32 => {
            let out = samples_f64
                .iter()
                .copied()
                .map(float64_to_i32)
                .collect::<Vec<_>>();
            write::<i32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::F32 => {
            let out = samples_f64.iter().map(|sample| *sample as f32).collect::<Vec<_>>();
            write::<f32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::F64 => {
            write::<f64, _>(path, samples_f64, sample_rate, n_channels)?;
        }
    }
    Ok(())
}

fn write_output_flac(
    path: &Path,
    channels: usize,
    output_rate_hz: u32,
    samples_f64: &[f64],
    target_format: OutFormatArg,
) -> Result<(), Box<dyn Error>> {
    if channels == 0 {
        return Err("cannot write FLAC with zero channels".into());
    }
    if channels > 8 {
        return Err(format!("FLAC supports up to 8 channels, got {}", channels).into());
    }
    if samples_f64.len() % channels != 0 {
        return Err(format!(
            "sample buffer length ({}) is not divisible by channel count ({})",
            samples_f64.len(),
            channels
        )
        .into());
    }

    let bits_per_sample = match target_format {
        OutFormatArg::I16 => 16_u32,
        OutFormatArg::I24 => 24_u32,
        OutFormatArg::I32 => 32_u32,
        OutFormatArg::F32 | OutFormatArg::F64 => {
            return Err(
                "FLAC output does not support float sample formats; use --out-format i16/i24/i32 or same"
                    .into(),
            )
        }
        OutFormatArg::Same => unreachable!("same is resolved to a concrete format"),
    };

    let channels_u8 = u8::try_from(channels).map_err(|_| "invalid FLAC channel count")?;
    let total_samples = samples_f64.len() as u64;
    let encoded = match target_format {
        OutFormatArg::I16 => samples_f64
            .iter()
            .copied()
            .map(float64_to_i16)
            .map(i32::from)
            .collect::<Vec<_>>(),
        OutFormatArg::I24 => samples_f64
            .iter()
            .copied()
            .map(float64_to_i24_i32)
            .collect::<Vec<_>>(),
        OutFormatArg::I32 => samples_f64
            .iter()
            .copied()
            .map(float64_to_i32)
            .collect::<Vec<_>>(),
        OutFormatArg::F32 | OutFormatArg::F64 | OutFormatArg::Same => unreachable!(),
    };

    let mut writer = FlacSampleWriter::create(
        path,
        FlacOptions::default(),
        output_rate_hz,
        bits_per_sample,
        channels_u8,
        Some(total_samples),
    )?;
    writer.write(&encoded)?;
    writer.finalize()?;
    Ok(())
}

fn wav_source_to_out_format(source: WavType) -> OutFormatArg {
    match source {
        WavType::Float64 | WavType::EFloat64 => OutFormatArg::F64,
        WavType::Float32 | WavType::EFloat32 => OutFormatArg::F32,
        _ => {
            let encoding = format!("{source:?}").to_ascii_lowercase();
            if encoding.contains("24") {
                OutFormatArg::I24
            } else if encoding.contains("16") {
                OutFormatArg::I16
            } else if encoding.contains("32") {
                OutFormatArg::I32
            } else {
                OutFormatArg::F32
            }
        }
    }
}

fn flac_bits_to_out_format(bits_per_sample: u32) -> OutFormatArg {
    if bits_per_sample <= 16 {
        OutFormatArg::I16
    } else if bits_per_sample <= 24 {
        OutFormatArg::I24
    } else {
        OutFormatArg::I32
    }
}

fn pcm_scale_from_bits(bits_per_sample: u32) -> Result<f64, Box<dyn Error>> {
    if !(1..=32).contains(&bits_per_sample) {
        return Err(format!(
            "unsupported FLAC bits-per-sample value: {} (expected 1..=32)",
            bits_per_sample
        )
        .into());
    }
    let max_int = ((1_i64 << (bits_per_sample - 1)) - 1) as f64;
    Ok(max_int.max(1.0))
}

fn float64_to_i16(value: f64) -> i16 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i16::MAX as f64).round() as i16
}

fn float64_to_i24(value: f64) -> i24 {
    const I24_MAX: f64 = ((1 << 23) - 1) as f64;
    let clamped = value.clamp(-1.0, 1.0);
    let as_i32 = (clamped * I24_MAX).round() as i32;
    i24::from_i32(as_i32)
}

fn float64_to_i24_i32(value: f64) -> i32 {
    const I24_MAX: f64 = ((1 << 23) - 1) as f64;
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * I24_MAX).round() as i32
}

fn float64_to_i32(value: f64) -> i32 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i32::MAX as f64).round() as i32
}

fn audio_container(path: &Path) -> Option<AudioContainer> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| {
            if ext.eq_ignore_ascii_case("wav") {
                Some(AudioContainer::Wav)
            } else if ext.eq_ignore_ascii_case("flac") {
                Some(AudioContainer::Flac)
            } else {
                None
            }
        })
}
