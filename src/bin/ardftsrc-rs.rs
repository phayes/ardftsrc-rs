use ardftsrc::{Ardftsrc, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, TaperType};
use clap::{Parser, ValueEnum};
use i24::i24;
use mimalloc::MiMalloc;
use std::error::Error;
use std::path::{Path, PathBuf};
use wavers::{Wav, WavType, read, write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    Fast,
    Good,
    High,
    Extreme,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TaperTypeArg {
    Planck,
    Cosine,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutFormatArg {
    Same,
    I16,
    I24,
    I32,
    F32,
    F64,
}

#[derive(Debug, Parser)]
#[command(name = "ardftsrc-rs")]
#[command(about = "General-purpose WAV converter powered by ardftsrc")]
struct Args {
    #[arg(long = "input", required = true)]
    input: Vec<PathBuf>,

    #[arg(long = "output", required = true)]
    output: Vec<PathBuf>,

    #[arg(long = "output-rate")]
    output_rate: usize,

    #[arg(long, value_enum, default_value_t = PresetArg::Good)]
    preset: PresetArg,

    #[arg(long)]
    quality: Option<usize>,

    #[arg(long)]
    bandwidth: Option<f32>,

    #[arg(long)]
    alpha: Option<f32>,

    #[arg(long = "taper-type", value_enum)]
    taper_type: Option<TaperTypeArg>,

    #[arg(long = "out-format", value_enum, default_value_t = OutFormatArg::Same)]
    out_format: OutFormatArg,

    #[arg(long)]
    gapless: bool,
}

#[derive(Debug)]
struct InputTrack {
    path: PathBuf,
    samples_f64: Vec<f64>,
    channels: usize,
    input_rate_hz: usize,
    source_format: WavType,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    let tracks = args
        .input
        .iter()
        .map(|path| read_wav_as_f64(path))
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
        write_output_wav(
            output_path,
            first.channels,
            args.output_rate as u32,
            &converted_samples,
            args.out_format,
            track.source_format,
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
    for path in args.input.iter().chain(args.output.iter()) {
        if !is_wav_path(path) {
            return Err(format!("only .wav files are supported: {}", path.display()).into());
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
        source_format,
    })
}

fn write_output_wav(
    path: &Path,
    channels: usize,
    output_rate_hz: u32,
    samples_f64: &[f64],
    out_format: OutFormatArg,
    source_format: WavType,
) -> Result<(), Box<dyn Error>> {
    let target_format = match out_format {
        OutFormatArg::Same => source_to_out_format(source_format),
        OutFormatArg::I16 => OutFormatArg::I16,
        OutFormatArg::I24 => OutFormatArg::I24,
        OutFormatArg::I32 => OutFormatArg::I32,
        OutFormatArg::F32 => OutFormatArg::F32,
        OutFormatArg::F64 => OutFormatArg::F64,
    };

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

fn source_to_out_format(source: WavType) -> OutFormatArg {
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

fn float64_to_i32(value: f64) -> i32 {
    let clamped = value.clamp(-1.0, 1.0);
    (clamped * i32::MAX as f64).round() as i32
}

fn is_wav_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
}
