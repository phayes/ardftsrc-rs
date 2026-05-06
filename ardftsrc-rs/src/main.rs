use ardftsrc::{
    BatchResampler, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, PlanarVecs, TaperType,
};
use clap::{Parser, ValueEnum};
use flac_codec::decode::FlacChannelReader;
use flac_codec::encode::{FlacChannelWriter, Options as FlacOptions};
use flac_codec::metadata::Metadata;
use i24::i24;
use mimalloc::MiMalloc;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use wavers::{Wav, WavType, read, write};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DEFAULT_ALPHA: f32 = 3.4375;
const FLAC_WRITE_CHUNK_FRAMES: usize = 32768;

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

    /// Treat multiple inputs as adjacent tracks and use neighboring tracks as edge context.
    #[arg(long)]
    gapless: bool,
}

#[derive(Debug)]
struct InputTrack {
    samples_f64: PlanarVecs<f64>,
    channels: usize,
    input_rate_hz: usize,
    source_out_format: OutFormatArg,
}

#[derive(Debug)]
struct InputJob {
    output_path: PathBuf,
    track: InputTrack,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct BatchGroupKey {
    channels: usize,
    input_rate_hz: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    let tracks = args
        .input
        .iter()
        .map(|path| read_audio_file(path))
        .collect::<Result<Vec<_>, _>>()?;

    let jobs = tracks
        .into_iter()
        .zip(args.output.iter().cloned())
        .map(|(track, output_path)| InputJob { output_path, track })
        .collect::<Vec<_>>();

    let grouped_jobs = group_compatible_jobs(jobs)?;
    if args.gapless && grouped_jobs.len() > 1 {
        return Err("--gapless requires all inputs to have matching channel count and sample rate".into());
    }

    process_and_write_all_groups(&args, grouped_jobs)?;

    Ok(())
}

#[cfg(feature = "rayon")]
fn process_and_write_all_groups(args: &Args, grouped_jobs: Vec<Vec<InputJob>>) -> Result<(), Box<dyn Error>> {
    grouped_jobs
        .into_par_iter()
        .map(|group| process_batch_group(args, group).map_err(|err| err.to_string()))
        .collect::<Result<Vec<_>, _>>()
        .map(|_| ())
        .map_err(|err: String| -> Box<dyn Error> { std::io::Error::other(err).into() })
}

#[cfg(not(feature = "rayon"))]
fn process_and_write_all_groups(args: &Args, grouped_jobs: Vec<Vec<InputJob>>) -> Result<(), Box<dyn Error>> {
    for group in grouped_jobs {
        process_batch_group(args, group)?;
    }
    Ok(())
}

fn process_batch_group(args: &Args, group: Vec<InputJob>) -> Result<(), Box<dyn Error>> {
    let output_rate_hz = args.output_rate as u32;
    let out_format = args.out_format;
    let first = group.first().ok_or("batch group cannot be empty")?;
    let config = build_config(args, first.track.input_rate_hz, first.track.channels)?;
    let processor = BatchResampler::<f64>::new(config)?;

    let mut metadata = Vec::with_capacity(group.len());
    let mut inputs = Vec::with_capacity(group.len());
    for job in group {
        metadata.push((job.output_path, job.track.source_out_format));
        inputs.push(job.track.samples_f64);
    }

    let converted = if args.gapless {
        processor.batch_planar_gapless(inputs)?
    } else {
        processor.batch_planar(inputs)?
    };

    if metadata.len() != converted.len() {
        return Err(std::io::Error::other(format!(
            "batch conversion output count mismatch: expected {}, got {}",
            metadata.len(),
            converted.len()
        ))
        .into());
    }

    let write_output_results = write_output(metadata, converted, output_rate_hz, out_format);
    let failed_writes = write_output_results
        .into_iter()
        .filter_map(|(output_path, maybe_err)| maybe_err.map(|err| (output_path, err)))
        .collect::<Vec<_>>();

    if !failed_writes.is_empty() {
        let mut message = format!("failed to write {} track(s):", failed_writes.len());
        for (output_path, err) in failed_writes {
            message.push_str(&format!("\n- {}: {}", output_path.display(), err));
        }
        return Err(std::io::Error::other(message).into());
    }

    Ok(())
}

#[cfg(feature = "rayon")]
fn write_output(
    metadata: Vec<(PathBuf, OutFormatArg)>,
    converted: Vec<PlanarVecs<f64>>,
    output_rate_hz: u32,
    out_format: OutFormatArg,
) -> Vec<(PathBuf, Option<String>)> {
    metadata
        .into_par_iter()
        .zip(converted.into_par_iter())
        .map(|((output_path, source_out_format), converted_samples)| {
            let maybe_err = write_output_audio(
                &output_path,
                output_rate_hz,
                converted_samples,
                out_format,
                source_out_format,
            )
            .err()
            .map(|err| err.to_string());
            (output_path, maybe_err)
        })
        .collect::<Vec<_>>()
}

#[cfg(not(feature = "rayon"))]
fn write_output(
    metadata: Vec<(PathBuf, OutFormatArg)>,
    converted: Vec<PlanarVecs<f64>>,
    output_rate_hz: u32,
    out_format: OutFormatArg,
) -> Vec<(PathBuf, Option<String>)> {
    metadata
        .into_iter()
        .zip(converted.into_iter())
        .map(|((output_path, source_out_format), converted_samples)| {
            let maybe_err = write_output_audio(
                &output_path,
                output_rate_hz,
                converted_samples,
                out_format,
                source_out_format,
            )
            .err()
            .map(|err| err.to_string());
            (output_path, maybe_err)
        })
        .collect::<Vec<_>>()
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
    let unique_outputs = args.output.iter().collect::<HashSet<_>>();
    if unique_outputs.len() != args.output.len() {
        return Err("--output must not contain duplicate paths".into());
    }
    if matches!(args.taper_type, Some(TaperTypeArg::Planck)) && args.alpha.is_some() {
        return Err("--alpha cannot be used with --taper-type=planck".into());
    }
    Ok(())
}

fn group_compatible_jobs(jobs: Vec<InputJob>) -> Result<Vec<Vec<InputJob>>, Box<dyn Error>> {
    if jobs.is_empty() {
        return Err("at least one input track is required".into());
    }

    let mut groups = Vec::<Vec<InputJob>>::new();
    let mut group_index_by_key = HashMap::<BatchGroupKey, usize>::new();

    for job in jobs {
        let key = BatchGroupKey {
            channels: job.track.channels,
            input_rate_hz: job.track.input_rate_hz,
        };
        if let Some(group_idx) = group_index_by_key.get(&key).copied() {
            groups[group_idx].push(job);
        } else {
            let group_idx = groups.len();
            groups.push(vec![job]);
            group_index_by_key.insert(key, group_idx);
        }
    }

    Ok(groups)
}

fn build_config(args: &Args, input_sample_rate: usize, channels: usize) -> Result<Config, Box<dyn Error>> {
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
                let alpha = args.alpha.unwrap_or(DEFAULT_ALPHA);
                TaperType::Cosine(alpha)
            }
        };
    } else if let Some(alpha) = args.alpha {
        config.taper_type = TaperType::Cosine(alpha);
    }

    config.validate()?;
    Ok(config)
}

fn read_audio_file(path: &Path) -> Result<InputTrack, Box<dyn Error>> {
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
        samples_f64: interleaved_to_planar(samples.as_ref(), channels)?,
        channels,
        input_rate_hz,
        source_out_format: wav_source_to_out_format(source_format),
    })
}

fn read_flac_as_f64(path: &Path) -> Result<InputTrack, Box<dyn Error>> {
    let mut reader = FlacChannelReader::open(path)?;
    let channels = reader.channel_count() as usize;
    let input_rate_hz = reader.sample_rate() as usize;
    let bits_per_sample = reader.bits_per_sample();
    let source_out_format = flac_bits_to_out_format(bits_per_sample);
    let scale = pcm_scale_from_bits(bits_per_sample)?;

    let mut per_channel = (0..channels)
        .map(|_| Vec::with_capacity(reader.total_samples().unwrap_or(0) as usize))
        .collect::<Vec<_>>();

    loop {
        let frames = {
            let decoded = reader.fill_buf()?;
            let frames = decoded.first().map_or(0, |channel| channel.len());
            if frames == 0 {
                break;
            }
            if decoded.len() != channels || decoded.iter().any(|channel| channel.len() != frames) {
                return Err("FLAC reader returned inconsistent channel buffers".into());
            }

            for (dst, src) in per_channel.iter_mut().zip(decoded) {
                dst.extend(src.iter().map(|sample| (*sample as f64 / scale).clamp(-1.0, 1.0)));
            }
            frames
        };
        reader.consume(frames);
    }

    let samples_f64 = PlanarVecs::new(per_channel)?;

    Ok(InputTrack {
        samples_f64,
        channels,
        input_rate_hz,
        source_out_format,
    })
}

fn interleaved_to_planar(samples: &[f64], channels: usize) -> Result<PlanarVecs<f64>, Box<dyn Error>> {
    if channels == 0 {
        return Err("audio input cannot have zero channels".into());
    }
    if samples.len() % channels != 0 {
        return Err(format!(
            "interleaved input length ({}) is not divisible by channel count ({})",
            samples.len(),
            channels
        )
        .into());
    }

    let frames = samples.len() / channels;
    let mut planar = vec![vec![0.0; frames]; channels];
    for (frame_idx, frame) in samples.chunks_exact(channels).enumerate() {
        for (channel_idx, sample) in frame.iter().enumerate() {
            planar[channel_idx][frame_idx] = *sample;
        }
    }

    Ok(PlanarVecs::new(planar)?)
}

fn write_output_audio(
    path: &Path,
    output_rate_hz: u32,
    samples: PlanarVecs<f64>,
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

    match audio_container(path) {
        Some(AudioContainer::Wav) => write_output_wav(path, output_rate_hz, &samples, target_format),
        Some(AudioContainer::Flac) => write_output_flac(path, output_rate_hz, samples, target_format),
        None => Err(format!(
            "unsupported output extension for {} (supported: .wav, .flac)",
            path.display()
        )
        .into()),
    }
}

fn write_output_wav(
    path: &Path,
    output_rate_hz: u32,
    samples_f64: &PlanarVecs<f64>,
    target_format: OutFormatArg,
) -> Result<(), Box<dyn Error>> {
    let sample_rate = output_rate_hz as i32;
    let n_channels = samples_f64.channels() as u16;
    match target_format {
        OutFormatArg::Same => unreachable!("same is resolved to a concrete format"),
        OutFormatArg::I16 => {
            let out = interleave_planar_mapped(samples_f64, float64_to_i16);
            write::<i16, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::I24 => {
            let out = interleave_planar_mapped(samples_f64, float64_to_i24);
            write::<i24, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::I32 => {
            let out = interleave_planar_mapped(samples_f64, float64_to_i32);
            write::<i32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::F32 => {
            let out = interleave_planar_mapped(samples_f64, |sample| sample as f32);
            write::<f32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::F64 => {
            let out = interleave_planar_mapped(samples_f64, std::convert::identity);
            write::<f64, _>(path, &out, sample_rate, n_channels)?;
        }
    }
    Ok(())
}

fn interleave_planar_mapped<T>(samples: &PlanarVecs<f64>, mut map_sample: impl FnMut(f64) -> T) -> Vec<T> {
    let channels = samples.channels();
    let frames = samples.frames();
    let per_channel = samples.as_slice();
    let mut output = Vec::with_capacity(channels * frames);

    for frame_idx in 0..frames {
        for channel in per_channel {
            output.push(map_sample(channel[frame_idx]));
        }
    }

    output
}

fn write_output_flac(
    path: &Path,
    output_rate_hz: u32,
    samples: PlanarVecs<f64>,
    target_format: OutFormatArg,
) -> Result<(), Box<dyn Error>> {
    let channels = samples.channels();
    if channels == 0 {
        return Err("cannot write FLAC with zero channels".into());
    }
    if channels > 8 {
        return Err(format!("FLAC supports up to 8 channels, got {}", channels).into());
    }

    let bits_per_sample = match target_format {
        OutFormatArg::I16 => 16_u32,
        OutFormatArg::I24 => 24_u32,
        OutFormatArg::I32 => 32_u32,
        OutFormatArg::F32 | OutFormatArg::F64 => {
            return Err(
                "FLAC output does not support float sample formats; use --out-format i16/i24/i32 or same".into(),
            );
        }
        OutFormatArg::Same => unreachable!("same is resolved to a concrete format"),
    };

    let frames = samples.frames();
    let planar = samples.as_slice();
    let channels_u8 = u8::try_from(channels).map_err(|_| "invalid FLAC channel count")?;

    let mut writer = FlacChannelWriter::create(
        path,
        FlacOptions::default(),
        output_rate_hz,
        bits_per_sample,
        channels_u8,
        Some(frames as u64),
    )?;

    let mut frame_offset = 0;
    while frame_offset < frames {
        let chunk_frames = (frames - frame_offset).min(FLAC_WRITE_CHUNK_FRAMES);
        let mut encoded_chunk = Vec::with_capacity(channels);
        for channel in planar {
            let end = frame_offset + chunk_frames;
            let encoded_channel = channel[frame_offset..end]
                .iter()
                .copied()
                .map(|sample| encode_flac_sample(sample, target_format))
                .collect::<Vec<i32>>();
            encoded_chunk.push(encoded_channel);
        }
        writer.write(&encoded_chunk)?;
        frame_offset += chunk_frames;
    }

    writer.finalize()?;
    Ok(())
}

fn encode_flac_sample(sample: f64, target_format: OutFormatArg) -> i32 {
    match target_format {
        OutFormatArg::I16 => i32::from(float64_to_i16(sample)),
        OutFormatArg::I24 => float64_to_i24_i32(sample),
        OutFormatArg::I32 => float64_to_i32(sample),
        OutFormatArg::F32 | OutFormatArg::F64 | OutFormatArg::Same => unreachable!(),
    }
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
    path.extension().and_then(|ext| ext.to_str()).and_then(|ext| {
        if ext.eq_ignore_ascii_case("wav") {
            Some(AudioContainer::Wav)
        } else if ext.eq_ignore_ascii_case("flac") {
            Some(AudioContainer::Flac)
        } else {
            None
        }
    })
}
