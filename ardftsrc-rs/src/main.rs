use ardftsrc::{
    AliasFloor, Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH, PlanarResampler, PlanarVecs, TaperType,
};
#[cfg(feature = "gpu")]
use ardftsrc::{GpuContext, GpuDevice, GpuError, PlanarGpuResampler};
use clap::{Parser, ValueEnum};
use flac_codec::encode::{FlacChannelWriter, Options as FlacOptions};
use i24::i24;
use mimalloc::MiMalloc;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(feature = "gpu")]
use std::sync::Arc;
use symphonia::core::audio::conv::ConvertibleSample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use wavers::{Wav, WavType, read, write};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DEFAULT_ALPHA: f32 = 3.4375;
#[cfg(feature = "bessel")]
const DEFAULT_BESSEL_ALPHA: f32 = 6.0;
const DEFAULT_BETA_CDF_ALPHA: f32 = 10.0;
const DEFAULT_BETA_CDF_BETA: f32 = 10.0;
const DEFAULT_ALLOW_ALIASING_DB: f32 = -3.0;
const FLAC_WRITE_CHUNK_FRAMES: usize = 32768;
const MAX_F32_QUALITY: usize = 8192;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    /// Fastest preset; lowest quality (quality = 512, bandwidth = 0.8323).
    Fast,
    /// Balanced quality/speed preset (quality = 1878, bandwidth = 0.911).
    Good,
    /// High quality preset for offline or quality-sensitive use (quality = 73622, bandwidth = 0.987).
    High,
    /// Maximum quality preset; slowest (quality = 524514, bandwidth = 0.995).
    Extreme,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TaperTypeArg {
    /// Planck taper transition.
    Planck,
    /// Cumulative Bessel-I0 taper transition.
    #[cfg(feature = "bessel")]
    Bessel,
    /// Descending Kaiser-Bessel-derived transition.
    #[cfg(feature = "bessel")]
    Kbd,
    /// Endpoint-normalized descending half-Kaiser transition.
    #[cfg(feature = "bessel")]
    #[value(name = "half_kaiser", alias = "half-kaiser")]
    HalfKaiser,
    /// Endpoint-normalized hyperbolic tangent transition.
    Tanh,
    /// Sigmoid-warped cosine taper transition.
    Cosine,
    /// Beta-CDF taper transition.
    #[value(name = "beta_cdf", alias = "beta-cdf")]
    BetaCdf,
}

impl TaperTypeArg {
    fn cli_name(self) -> &'static str {
        match self {
            Self::Planck => "planck",
            #[cfg(feature = "bessel")]
            Self::Bessel => "bessel",
            #[cfg(feature = "bessel")]
            Self::Kbd => "kbd",
            #[cfg(feature = "bessel")]
            Self::HalfKaiser => "half_kaiser",
            Self::Tanh => "tanh",
            Self::Cosine => "cosine",
            Self::BetaCdf => "beta_cdf",
        }
    }

    fn accepts_alpha(self) -> bool {
        match self {
            Self::Cosine | Self::BetaCdf | Self::Tanh => true,
            Self::Planck => false,
            #[cfg(feature = "bessel")]
            Self::Bessel | Self::Kbd | Self::HalfKaiser => true,
        }
    }

    fn compatible_alpha_types() -> &'static str {
        #[cfg(feature = "bessel")]
        {
            "cosine, beta_cdf, tanh, bessel, kbd, half_kaiser"
        }
        #[cfg(not(feature = "bessel"))]
        {
            "cosine, beta_cdf, tanh"
        }
    }

    fn compatible_beta_types() -> &'static str {
        "beta_cdf"
    }
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
#[command(
    about = "General-purpose wav and flac sample-rate converter powered by ardftsrc. Uses a compatible GPU automatically when available; pass --cpu to force CPU."
)]
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

    /// Taper alpha. Requires a compatible --taper-type (cosine, beta_cdf, tanh, or a Bessel-family taper).
    ///
    /// For cosine, higher values are sharper cutoff; lower values are smoother.
    /// Defaults: cosine 3.4375, beta_cdf 10.0, tanh 3.0, bessel/kbd/half_kaiser 6.0.
    #[arg(long)]
    alpha: Option<f32>,

    /// Beta-CDF taper beta. Used by --taper-type beta_cdf. Default is 10.0.
    #[arg(long)]
    beta: Option<f32>,

    /// Transition taper profile.
    #[arg(long = "taper-type", value_enum)]
    taper_type: Option<TaperTypeArg>,

    /// Frequency-dependent phase rotation in [-1.0, 1.0]. Negative values can reduce pre-ringing.
    #[arg(long)]
    phase: Option<f32>,

    /// Phase rotation intensity in [0.0, 100.0]. Default is 50.0. Ignored when --phase is 0.
    #[arg(long = "phase-intensity")]
    phase_intensity: Option<f32>,

    /// Permit aliasing/imaging inside the low-pass transition band (similar to SoX `rate -a`).
    /// Widens the transition to reduce ringing at the cost of alias rejection; the passband is
    /// unchanged. Equivalent to --alias-floor-db -3.
    #[arg(short = 'a', long = "allow-aliasing", conflicts_with_all = ["alias_floor", "alias_floor_db"])]
    allow_aliasing: bool,

    /// Lowest frequency that may receive folded/imaged energy, as a fraction of the lower Nyquist
    /// in [bandwidth, 1.0]. 1.0 (default) disables aliasing.
    #[arg(long = "alias-floor", conflicts_with = "alias_floor_db")]
    alias_floor: Option<f32>,

    /// Fold/image only down to the frequency where the filter response is this many dB (< 0).
    #[arg(long = "alias-floor-db", allow_negative_numbers = true)]
    alias_floor_db: Option<f32>,

    /// Enable 2:1 pre-decimation for large downsampling ratios (4:1 or higher).
    #[arg(long)]
    decimate: bool,

    /// Use a high-precision FFT backend. Much slower; intended for extreme quality.
    /// double-double (~106-bit) is the fastest; f256 (~237-bit) is the most precise.
    #[cfg(feature = "high_precision")]
    #[arg(long = "high-precision", value_enum, conflicts_with = "use_f32")]
    high_precision: Option<HighPrecisionArg>,

    /// Process as 32-bit floats instead of the default 64-bit. Quality above 8192 is not
    /// supported (`--preset high` and `--preset extreme` exceed this limit).
    #[arg(long = "f32")]
    use_f32: bool,

    /// Force CPU resampling even when a compatible GPU is available.
    #[arg(long = "cpu")]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    force_cpu: bool,

    /// FFT chunks combined into each GPU submission. Default is 4. Use 1 for low latency;
    /// values above 4 may help large offline jobs. Ignored on CPU.
    #[cfg(feature = "gpu")]
    #[arg(long = "gpu-group-chunks")]
    gpu_group_chunks: Option<usize>,

    /// Reusable GPU work groups in the streaming ring. Default is 4 (also the effective
    /// minimum). 8 can help when host-to-GPU submission is starved. Ignored on CPU.
    #[cfg(feature = "gpu")]
    #[arg(long = "gpu-ring-slots")]
    gpu_ring_slots: Option<usize>,

    /// Output sample format. For .flac output, float formats are rejected.
    #[arg(long = "out-format", value_enum, default_value_t = OutFormatArg::Same)]
    out_format: OutFormatArg,

    /// Treat multiple inputs as adjacent tracks and use neighboring tracks as edge context.
    #[arg(long)]
    gapless: bool,
}

#[derive(Debug)]
struct InputTrack<T> {
    samples: PlanarVecs<T>,
    channels: usize,
    input_rate_hz: usize,
    source_out_format: OutFormatArg,
}

#[derive(Debug)]
struct InputJob<T> {
    output_path: PathBuf,
    track: InputTrack<T>,
}

/// Decode, resample, and encode sample type (`f32` or `f64`).
trait ProcessingSample: Copy + Default + Send + Sync + ConvertibleSample + 'static {
    fn read_wav_planar(path: &Path, channels: usize) -> Result<PlanarVecs<Self>, Box<dyn Error>>;
    fn to_i16(self) -> i16;
    fn to_i24(self) -> i24;
    fn to_i24_i32(self) -> i32;
    fn to_i32(self) -> i32;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;

    fn resample(
        config: Config,
        inputs: Vec<PlanarVecs<Self>>,
        gapless: bool,
        #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
    ) -> Result<Vec<PlanarVecs<Self>>, Box<dyn Error>>;
}

impl ProcessingSample for f32 {
    fn read_wav_planar(path: &Path, channels: usize) -> Result<PlanarVecs<Self>, Box<dyn Error>> {
        let (samples, _) = read::<f32, _>(path)?;
        interleaved_to_planar(samples.as_ref(), channels)
    }

    fn to_i16(self) -> i16 {
        (self.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
    }

    fn to_i24(self) -> i24 {
        i24::from_i32(self.to_i24_i32())
    }

    fn to_i24_i32(self) -> i32 {
        const I24_MAX: f32 = ((1 << 23) - 1) as f32;
        (self.clamp(-1.0, 1.0) * I24_MAX).round() as i32
    }

    fn to_i32(self) -> i32 {
        (self.clamp(-1.0, 1.0) * i32::MAX as f32).round() as i32
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn resample(
        config: Config,
        inputs: Vec<PlanarVecs<Self>>,
        gapless: bool,
        #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
    ) -> Result<Vec<PlanarVecs<Self>>, Box<dyn Error>> {
        resample_f32(
            config,
            inputs,
            gapless,
            #[cfg(feature = "gpu")]
            gpu,
        )
    }
}

impl ProcessingSample for f64 {
    fn read_wav_planar(path: &Path, channels: usize) -> Result<PlanarVecs<Self>, Box<dyn Error>> {
        let (samples, _) = read::<f64, _>(path)?;
        interleaved_to_planar(samples.as_ref(), channels)
    }

    fn to_i16(self) -> i16 {
        (self.clamp(-1.0, 1.0) * f64::from(i16::MAX)).round() as i16
    }

    fn to_i24(self) -> i24 {
        i24::from_i32(self.to_i24_i32())
    }

    fn to_i24_i32(self) -> i32 {
        const I24_MAX: f64 = ((1 << 23) - 1) as f64;
        (self.clamp(-1.0, 1.0) * I24_MAX).round() as i32
    }

    fn to_i32(self) -> i32 {
        (self.clamp(-1.0, 1.0) * f64::from(i32::MAX)).round() as i32
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn resample(
        config: Config,
        inputs: Vec<PlanarVecs<Self>>,
        gapless: bool,
        #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
    ) -> Result<Vec<PlanarVecs<Self>>, Box<dyn Error>> {
        resample_f64(
            config,
            inputs,
            gapless,
            #[cfg(feature = "gpu")]
            gpu,
        )
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct BatchGroupKey {
    channels: usize,
    input_rate_hz: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    #[cfg(feature = "gpu")]
    let gpu = select_gpu(&args);

    if args.use_f32 {
        convert_all::<f32>(
            &args,
            #[cfg(feature = "gpu")]
            gpu.as_ref(),
        )
    } else {
        convert_all::<f64>(
            &args,
            #[cfg(feature = "gpu")]
            gpu.as_ref(),
        )
    }
}

fn convert_all<T: ProcessingSample>(
    args: &Args,
    #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
) -> Result<(), Box<dyn Error>> {
    let tracks = args
        .input
        .iter()
        .map(|path| read_audio_file::<T>(path))
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

    process_and_write_all_groups(
        args,
        grouped_jobs,
        #[cfg(feature = "gpu")]
        gpu,
    )
}

#[cfg(feature = "rayon")]
fn process_and_write_all_groups<T: ProcessingSample>(
    args: &Args,
    grouped_jobs: Vec<Vec<InputJob<T>>>,
    #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
) -> Result<(), Box<dyn Error>> {
    grouped_jobs
        .into_par_iter()
        .map(|group| {
            process_batch_group(
                args,
                group,
                #[cfg(feature = "gpu")]
                gpu,
            )
            .map_err(|err| err.to_string())
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|_| ())
        .map_err(|err: String| -> Box<dyn Error> { std::io::Error::other(err).into() })
}

#[cfg(not(feature = "rayon"))]
fn process_and_write_all_groups<T: ProcessingSample>(
    args: &Args,
    grouped_jobs: Vec<Vec<InputJob<T>>>,
    #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
) -> Result<(), Box<dyn Error>> {
    for group in grouped_jobs {
        process_batch_group(
            args,
            group,
            #[cfg(feature = "gpu")]
            gpu,
        )?;
    }
    Ok(())
}

fn process_batch_group<T: ProcessingSample>(
    args: &Args,
    group: Vec<InputJob<T>>,
    #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
) -> Result<(), Box<dyn Error>> {
    let output_rate_hz = args.output_rate as u32;
    let out_format = args.out_format;
    let first = group.first().ok_or("batch group cannot be empty")?;
    let config = build_config(args, first.track.input_rate_hz, first.track.channels)?;

    let mut metadata = Vec::with_capacity(group.len());
    let mut inputs = Vec::with_capacity(group.len());
    for job in group {
        metadata.push((job.output_path, job.track.source_out_format));
        inputs.push(job.track.samples);
    }

    let converted = T::resample(
        config,
        inputs,
        args.gapless,
        #[cfg(feature = "gpu")]
        gpu,
    )?;

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
fn write_output<T: ProcessingSample>(
    metadata: Vec<(PathBuf, OutFormatArg)>,
    converted: Vec<PlanarVecs<T>>,
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
fn write_output<T: ProcessingSample>(
    metadata: Vec<(PathBuf, OutFormatArg)>,
    converted: Vec<PlanarVecs<T>>,
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
    if args.alpha.is_some() {
        match args.taper_type {
            Some(taper) if taper.accepts_alpha() => {}
            Some(taper) => {
                return Err(format!(
                    "--alpha is not compatible with --taper-type={} (compatible: {})",
                    taper.cli_name(),
                    TaperTypeArg::compatible_alpha_types()
                )
                .into());
            }
            None => {
                return Err(format!(
                    "--alpha requires a compatible --taper-type (compatible: {})",
                    TaperTypeArg::compatible_alpha_types()
                )
                .into());
            }
        }
    }
    if args.beta.is_some() {
        match args.taper_type {
            Some(TaperTypeArg::BetaCdf) => {}
            Some(taper) => {
                return Err(format!(
                    "--beta is not compatible with --taper-type={} (compatible: {})",
                    taper.cli_name(),
                    TaperTypeArg::compatible_beta_types()
                )
                .into());
            }
            None => {
                return Err(format!(
                    "--beta requires a compatible --taper-type (compatible: {})",
                    TaperTypeArg::compatible_beta_types()
                )
                .into());
            }
        }
    }
    if args.use_f32 {
        let quality = args.quality.unwrap_or(preset_config(args.preset).quality);
        if quality > MAX_F32_QUALITY {
            return Err(format!(
                "--f32 does not support quality {quality} (maximum is {MAX_F32_QUALITY}). \
                 Use --preset good or --preset fast, or omit --f32 to keep f64 processing."
            )
            .into());
        }
    }
    Ok(())
}

fn group_compatible_jobs<T>(jobs: Vec<InputJob<T>>) -> Result<Vec<Vec<InputJob<T>>>, Box<dyn Error>> {
    if jobs.is_empty() {
        return Err("at least one input track is required".into());
    }

    let mut groups = Vec::<Vec<InputJob<T>>>::new();
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

fn preset_config(preset: PresetArg) -> Config {
    match preset {
        PresetArg::Fast => PRESET_FAST,
        PresetArg::Good => PRESET_GOOD,
        PresetArg::High => PRESET_HIGH,
        PresetArg::Extreme => PRESET_EXTREME,
    }
}

fn build_config(args: &Args, input_sample_rate: usize, channels: usize) -> Result<Config, Box<dyn Error>> {
    let mut config = preset_config(args.preset)
        .with_input_rate(input_sample_rate)
        .with_output_rate(args.output_rate)
        .with_channels(channels);

    if let Some(quality) = args.quality {
        config.quality = quality;
    }
    if let Some(bandwidth) = args.bandwidth {
        config.bandwidth = bandwidth;
    }
    if let Some(phase) = args.phase {
        config.phase = phase;
    }
    if let Some(phase_intensity) = args.phase_intensity {
        config.phase_intensity = phase_intensity;
    }
    if args.allow_aliasing {
        config.alias_floor = AliasFloor::Decibels(DEFAULT_ALLOW_ALIASING_DB);
    }
    if let Some(fraction) = args.alias_floor {
        config.alias_floor = AliasFloor::Fraction(fraction);
    }
    if let Some(db) = args.alias_floor_db {
        config.alias_floor = AliasFloor::Decibels(db);
    }
    if args.decimate {
        config.decimate = true;
    }
    #[cfg(feature = "high_precision")]
    if let Some(high_precision) = args.high_precision {
        config.high_precision = Some(high_precision.into());
    }
    #[cfg(feature = "gpu")]
    if let Some(gpu_group_chunks) = args.gpu_group_chunks {
        config.gpu_group_chunks = gpu_group_chunks;
    }
    #[cfg(feature = "gpu")]
    if let Some(gpu_ring_slots) = args.gpu_ring_slots {
        config.gpu_ring_slots = gpu_ring_slots;
    }

    if let Some(taper_type) = args.taper_type {
        config.taper_type = match taper_type {
            TaperTypeArg::Planck => TaperType::Planck,
            #[cfg(feature = "bessel")]
            TaperTypeArg::Bessel => {
                let alpha = args.alpha.unwrap_or(DEFAULT_BESSEL_ALPHA);
                TaperType::Bessel(alpha)
            }
            #[cfg(feature = "bessel")]
            TaperTypeArg::Kbd => TaperType::Kbd(args.alpha.unwrap_or(DEFAULT_BESSEL_ALPHA)),
            #[cfg(feature = "bessel")]
            TaperTypeArg::HalfKaiser => TaperType::HalfKaiser(args.alpha.unwrap_or(DEFAULT_BESSEL_ALPHA)),
            TaperTypeArg::Tanh => TaperType::Tanh(args.alpha.unwrap_or(3.0)),
            TaperTypeArg::Cosine => {
                let alpha = args.alpha.unwrap_or(DEFAULT_ALPHA);
                TaperType::Cosine(alpha)
            }
            TaperTypeArg::BetaCdf => {
                let alpha = args.alpha.unwrap_or(DEFAULT_BETA_CDF_ALPHA);
                let beta = args.beta.unwrap_or(DEFAULT_BETA_CDF_BETA);
                TaperType::BetaCdf { alpha, beta }
            }
        };
    }

    config.validate()?;
    Ok(config)
}

#[cfg(feature = "high_precision")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum HighPrecisionArg {
    /// Double-double (~106-bit mantissa).
    DoubleDouble,
    /// IEEE binary128 (113-bit mantissa).
    F128,
    /// IEEE binary256 (237-bit mantissa).
    F256,
}

#[cfg(feature = "high_precision")]
impl From<HighPrecisionArg> for ardftsrc::HighPrecision {
    fn from(arg: HighPrecisionArg) -> Self {
        match arg {
            HighPrecisionArg::DoubleDouble => Self::DoubleDouble,
            HighPrecisionArg::F128 => Self::F128,
            HighPrecisionArg::F256 => Self::F256,
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu_config_supported(args: &Args) -> bool {
    if args.decimate {
        return false;
    }
    #[cfg(feature = "high_precision")]
    if args.high_precision.is_some() {
        return false;
    }
    true
}

#[cfg(feature = "gpu")]
fn select_gpu(args: &Args) -> Option<Arc<GpuDevice>> {
    if args.force_cpu || !gpu_config_supported(args) {
        return None;
    }
    match GpuDevice::auto_select_compatible(!args.use_f32) {
        Ok(device) => {
            eprintln!("ardftsrc-rs: using GPU ({})", device.device_name());
            Some(Arc::new(device))
        }
        Err(_) => None,
    }
}

fn resample_f64(
    config: Config,
    inputs: Vec<PlanarVecs<f64>>,
    gapless: bool,
    #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
) -> Result<Vec<PlanarVecs<f64>>, Box<dyn Error>> {
    #[cfg(feature = "gpu")]
    if let Some(device) = gpu {
        match start_gpu_resampler_f64(device, config.clone()) {
            Ok(processor) => return gpu_batch_f64(processor, inputs, gapless),
            Err(err) => {
                eprintln!("ardftsrc-rs: GPU resampler failed to start ({err}); using CPU");
            }
        }
    }
    resample_cpu_f64(config, inputs, gapless)
}

fn resample_f32(
    config: Config,
    inputs: Vec<PlanarVecs<f32>>,
    gapless: bool,
    #[cfg(feature = "gpu")] gpu: Option<&Arc<GpuDevice>>,
) -> Result<Vec<PlanarVecs<f32>>, Box<dyn Error>> {
    #[cfg(feature = "gpu")]
    if let Some(device) = gpu {
        match start_gpu_resampler_f32(device, config.clone()) {
            Ok(processor) => return gpu_batch_f32(processor, inputs, gapless),
            Err(err) => {
                eprintln!("ardftsrc-rs: GPU resampler failed to start ({err}); using CPU");
            }
        }
    }
    resample_cpu_f32(config, inputs, gapless)
}

fn resample_cpu_f64(
    config: Config,
    inputs: Vec<PlanarVecs<f64>>,
    gapless: bool,
) -> Result<Vec<PlanarVecs<f64>>, Box<dyn Error>> {
    let processor = PlanarResampler::<f64>::new(config)?;
    if gapless {
        Ok(processor.batch_gapless(inputs)?)
    } else {
        Ok(processor.batch(inputs)?)
    }
}

fn resample_cpu_f32(
    config: Config,
    inputs: Vec<PlanarVecs<f32>>,
    gapless: bool,
) -> Result<Vec<PlanarVecs<f32>>, Box<dyn Error>> {
    let processor = PlanarResampler::<f32>::new(config)?;
    if gapless {
        Ok(processor.batch_gapless(inputs)?)
    } else {
        Ok(processor.batch(inputs)?)
    }
}

#[cfg(feature = "gpu")]
fn start_gpu_resampler_f64(device: &Arc<GpuDevice>, config: Config) -> Result<PlanarGpuResampler<f64>, GpuError> {
    PlanarGpuResampler::new(GpuContext::<f64>::with_device(Arc::clone(device), config)?)
}

#[cfg(feature = "gpu")]
fn start_gpu_resampler_f32(device: &Arc<GpuDevice>, config: Config) -> Result<PlanarGpuResampler<f32>, GpuError> {
    PlanarGpuResampler::new(GpuContext::<f32>::with_device(Arc::clone(device), config)?)
}

#[cfg(feature = "gpu")]
fn gpu_batch_f64(
    processor: PlanarGpuResampler<f64>,
    inputs: Vec<PlanarVecs<f64>>,
    gapless: bool,
) -> Result<Vec<PlanarVecs<f64>>, Box<dyn Error>> {
    if gapless {
        Ok(processor.batch_gapless(inputs)?)
    } else {
        Ok(processor.batch(inputs)?)
    }
}

#[cfg(feature = "gpu")]
fn gpu_batch_f32(
    processor: PlanarGpuResampler<f32>,
    inputs: Vec<PlanarVecs<f32>>,
    gapless: bool,
) -> Result<Vec<PlanarVecs<f32>>, Box<dyn Error>> {
    if gapless {
        Ok(processor.batch_gapless(inputs)?)
    } else {
        Ok(processor.batch(inputs)?)
    }
}

fn read_audio_file<T: ProcessingSample>(path: &Path) -> Result<InputTrack<T>, Box<dyn Error>> {
    let result = match audio_container(path) {
        Some(AudioContainer::Wav) => read_wav(path),
        Some(AudioContainer::Flac) => read_flac(path),
        None => {
            return Err(format!(
                "unsupported input extension for {} (supported: .wav, .flac)",
                path.display()
            )
            .into());
        }
    };
    result.map_err(|err| format!("failed to read {}: {err}", path.display()).into())
}

fn read_wav<T: ProcessingSample>(path: &Path) -> Result<InputTrack<T>, Box<dyn Error>> {
    let probe = Wav::<f32>::from_path(path)?;
    let channels = probe.n_channels() as usize;
    let input_rate_hz = probe.sample_rate() as usize;
    let source_format = probe.encoding();
    drop(probe);

    Ok(InputTrack {
        samples: T::read_wav_planar(path, channels)?,
        channels,
        input_rate_hz,
        source_out_format: wav_source_to_out_format(source_format),
    })
}

fn read_flac<T: ProcessingSample>(path: &Path) -> Result<InputTrack<T>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }

    let mut format =
        symphonia::default::get_probe().probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or("FLAC file does not contain a default audio track")?
        .clone();
    let track_id = track.id;
    let audio = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or("FLAC track is missing audio codec parameters")?;
    let channels = audio
        .channels
        .as_ref()
        .ok_or("FLAC track is missing channel information")?
        .count();
    if channels == 0 {
        return Err("FLAC track has zero channels".into());
    }
    let input_rate_hz = audio.sample_rate.ok_or("FLAC track is missing sample rate")? as usize;
    let bits_per_sample = audio.bits_per_sample.unwrap_or(16);
    let source_out_format = flac_bits_to_out_format(bits_per_sample);

    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(audio, &AudioDecoderOptions::default())?;
    let mut per_channel = (0..channels)
        .map(|_| Vec::with_capacity(track.num_frames.unwrap_or(0) as usize))
        .collect::<Vec<_>>();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => {
                return Err("FLAC decoder reset required mid-stream".into());
            }
            Err(err) => return Err(err.into()),
        };
        if packet.track_id != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        if decoded.spec().channels().count() != channels {
            return Err("FLAC packet channel count does not match the track".into());
        }

        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }

        for channel in &mut per_channel {
            channel.resize(channel.len() + frames, T::default());
        }
        let mut planes = per_channel
            .iter_mut()
            .map(|channel| {
                let start = channel.len() - frames;
                &mut channel[start..]
            })
            .collect::<Vec<_>>();
        decoded.copy_to_slice_planar(&mut planes);
    }

    Ok(InputTrack {
        samples: PlanarVecs::new(per_channel)?,
        channels,
        input_rate_hz,
        source_out_format,
    })
}

fn interleaved_to_planar<T: Copy + Default>(samples: &[T], channels: usize) -> Result<PlanarVecs<T>, Box<dyn Error>> {
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
    let mut planar = vec![vec![T::default(); frames]; channels];
    for (frame_idx, frame) in samples.chunks_exact(channels).enumerate() {
        for (channel_idx, sample) in frame.iter().enumerate() {
            planar[channel_idx][frame_idx] = *sample;
        }
    }

    Ok(PlanarVecs::new(planar)?)
}

fn write_output_audio<T: ProcessingSample>(
    path: &Path,
    output_rate_hz: u32,
    samples: PlanarVecs<T>,
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

fn write_output_wav<T: ProcessingSample>(
    path: &Path,
    output_rate_hz: u32,
    samples: &PlanarVecs<T>,
    target_format: OutFormatArg,
) -> Result<(), Box<dyn Error>> {
    let sample_rate = output_rate_hz as i32;
    let n_channels = samples.channels() as u16;
    match target_format {
        OutFormatArg::Same => unreachable!("same is resolved to a concrete format"),
        OutFormatArg::I16 => {
            let out = interleave_planar_mapped(samples, T::to_i16);
            write::<i16, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::I24 => {
            let out = interleave_planar_mapped(samples, T::to_i24);
            write::<i24, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::I32 => {
            let out = interleave_planar_mapped(samples, T::to_i32);
            write::<i32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::F32 => {
            let out = interleave_planar_mapped(samples, T::to_f32);
            write::<f32, _>(path, &out, sample_rate, n_channels)?;
        }
        OutFormatArg::F64 => {
            let out = interleave_planar_mapped(samples, T::to_f64);
            write::<f64, _>(path, &out, sample_rate, n_channels)?;
        }
    }
    Ok(())
}

fn interleave_planar_mapped<S: Copy, T>(samples: &PlanarVecs<S>, mut map_sample: impl FnMut(S) -> T) -> Vec<T> {
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

fn write_output_flac<T: ProcessingSample>(
    path: &Path,
    output_rate_hz: u32,
    samples: PlanarVecs<T>,
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
                .map(|sample| encode_flac_sample::<T>(sample, target_format))
                .collect::<Vec<i32>>();
            encoded_chunk.push(encoded_channel);
        }
        writer.write(&encoded_chunk)?;
        frame_offset += chunk_frames;
    }

    writer.finalize()?;
    Ok(())
}

fn encode_flac_sample<T: ProcessingSample>(sample: T, target_format: OutFormatArg) -> i32 {
    match target_format {
        OutFormatArg::I16 => i32::from(sample.to_i16()),
        OutFormatArg::I24 => sample.to_i24_i32(),
        OutFormatArg::I32 => sample.to_i32(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_taper_cli_options_apply_defaults_and_alpha() {
        let cases = [
            ("tanh", TaperType::Tanh(3.0), TaperType::Tanh(2.0)),
            #[cfg(feature = "bessel")]
            ("kbd", TaperType::Kbd(6.0), TaperType::Kbd(2.0)),
            #[cfg(feature = "bessel")]
            ("half_kaiser", TaperType::HalfKaiser(6.0), TaperType::HalfKaiser(2.0)),
            #[cfg(feature = "bessel")]
            ("half-kaiser", TaperType::HalfKaiser(6.0), TaperType::HalfKaiser(2.0)),
        ];
        for (name, default, overridden) in cases {
            let mut argv = vec![
                "ardftsrc-rs",
                "--input",
                "in.wav",
                "--output",
                "out.wav",
                "--output-rate",
                "48000",
                "--taper-type",
                name,
            ];
            let args = Args::try_parse_from(&argv).unwrap();
            assert!(args.taper_type.unwrap().accepts_alpha());
            assert_eq!(build_config(&args, 44_100, 2).unwrap().taper_type, default);
            argv.extend(["--alpha", "2"]);
            let args = Args::try_parse_from(&argv).unwrap();
            assert_eq!(build_config(&args, 44_100, 2).unwrap().taper_type, overridden);
        }
    }

    #[test]
    fn cpu_flag_is_accepted() {
        let args = Args::try_parse_from([
            "ardftsrc-rs",
            "--input",
            "in.wav",
            "--output",
            "out.wav",
            "--output-rate",
            "48000",
            "--cpu",
        ])
        .unwrap();
        assert!(args.force_cpu);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_config_flags_apply_to_config() {
        let args = Args::try_parse_from([
            "ardftsrc-rs",
            "--input",
            "in.wav",
            "--output",
            "out.wav",
            "--output-rate",
            "48000",
            "--gpu-group-chunks",
            "8",
            "--gpu-ring-slots",
            "8",
        ])
        .unwrap();
        let config = build_config(&args, 44_100, 2).unwrap();
        assert_eq!(config.gpu_group_chunks, 8);
        assert_eq!(config.gpu_ring_slots, 8);
    }

    #[test]
    fn f32_quality_limit_matches_supported_presets() {
        assert!(preset_config(PresetArg::Fast).quality <= MAX_F32_QUALITY);
        assert!(preset_config(PresetArg::Good).quality <= MAX_F32_QUALITY);
        assert!(preset_config(PresetArg::High).quality > MAX_F32_QUALITY);
        assert!(preset_config(PresetArg::Extreme).quality > MAX_F32_QUALITY);
    }

    #[test]
    fn interleaved_to_planar_is_generic_over_precision() {
        let interleaved_f32 = [0.5_f32, 1.0, -0.25, 0.0];
        let planar = interleaved_to_planar(&interleaved_f32, 2).unwrap();
        assert_eq!(planar.as_slice(), &[vec![0.5_f32, -0.25], vec![1.0, 0.0]]);
    }

    #[test]
    fn f32_pcm_conversions_stay_in_f32() {
        assert_eq!(1.0_f32.to_i16(), i16::MAX);
        assert_eq!((-1.0_f32).to_i16(), -i16::MAX);
        assert_eq!(1.0_f32.to_f32(), 1.0);
        assert_eq!(0.5_f32.to_f64(), 0.5);
    }

    #[test]
    fn f64_pcm_conversions_stay_in_f64() {
        assert_eq!(1.0_f64.to_i16(), i16::MAX);
        assert_eq!(1.0_f64.to_f64(), 1.0);
        assert_eq!(0.5_f64.to_f32(), 0.5);
    }
}
