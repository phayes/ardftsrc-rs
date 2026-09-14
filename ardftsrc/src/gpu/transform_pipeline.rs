use std::sync::Arc;

use vkfft_rs::{Precision, RealFftKind};

use super::buffer::{GpuBuffer, GpuScalar};
use super::context::{GpuDevice, RecordingCommandBuffer};
use super::error::GpuError;
use super::fft_program::{FromF64, GpuRealFft};
use super::remap_shader::{RemapGeometry, RemapShader};
use super::shaders::TransformShaders;

/// One complete "forward R2C -> spectral remap -> inverse C2R" GPU pipeline for a fixed
/// `(input_fft_size, output_fft_size, batch_count)` -- the whole of `gpu_plan.md` milestone 5,
/// and the building block [`GpuCore`](super::GpuCore) records its overlap-add/normalization work
/// around.
///
/// [`GpuTransformPipeline::record`] appends every pass of the forward FFT, the remap
/// dispatch, and every pass of the inverse FFT (each followed by a full compute barrier) into
/// one device-branded recording command buffer. [`GpuDevice::submit_async`] submits that
/// whole recording as one unit, so the spectrum never leaves the GPU
/// between the forward and inverse FFT -- satisfying `gpu_plan.md`'s design rule against a
/// host round-trip there, without touching `vkfft-rs` internals.
pub(crate) struct GpuTransformPipeline<T> {
    // Must be dropped before `forward` and `inverse`: its descriptors bind buffers owned by
    // both FFTs. Rust drops struct fields in declaration order.
    remap: RemapShader<T>,
    forward: GpuRealFft<T>,
    inverse: GpuRealFft<T>,
}

impl<T: GpuScalar + FromF64> GpuTransformPipeline<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        context: &Arc<GpuDevice>,
        precision: Precision,
        input_fft_size: usize,
        output_fft_size: usize,
        batch_count: usize,
        geometry: &RemapGeometry<'_, T>,
        shaders: &TransformShaders,
    ) -> Result<Self, GpuError> {
        if shaders.batch_count != batch_count {
            return Err(GpuError::IncompatibleShaders(
                "transform batch count does not match".to_string(),
            ));
        }
        let forward = GpuRealFft::<T>::build(
            context,
            RealFftKind::RealToComplex,
            input_fft_size,
            batch_count,
            precision,
            false,
            &shaders.forward_fft,
        )?;
        // `realfft`/`CpuCore` convention: the inverse transform is normalized by its own
        // length (see `CpuCore::transform_chunk`'s manual `normalize` scale), so this asks
        // `vkfft-rs` to do that division as part of its own generated passes rather than
        // needing a separate GPU normalization step.
        let inverse = GpuRealFft::<T>::build(
            context,
            RealFftKind::ComplexToReal,
            output_fft_size,
            batch_count,
            precision,
            true,
            &shaders.inverse_fft,
        )?;

        let remap = RemapShader::<T>::build(
            context,
            forward.output_buffer(),
            inverse.input_buffer(),
            geometry,
            shaders.remap.as_spirv(),
        )?;

        Ok(Self {
            remap,
            forward,
            inverse,
        })
    }

    /// The buffer real input samples must be written into (packed as `Complex<T>`, `im = 0`),
    /// batched contiguously at [`GpuTransformPipeline::input_stride`] complex slots per batch.
    pub(crate) fn input_buffer(&self) -> &GpuBuffer<T> {
        self.forward.input_buffer()
    }

    /// The buffer the final (already-normalized) real output samples are available in after
    /// [`GpuTransformPipeline::record`] runs, packed as `Complex<T>` (`im` always `0`).
    pub(crate) fn output_buffer(&self) -> &GpuBuffer<T> {
        self.inverse.output_buffer()
    }

    pub(crate) fn input_stride(&self) -> usize {
        self.forward.input_stride()
    }

    pub(crate) fn output_stride(&self) -> usize {
        self.inverse.output_stride()
    }

    /// Records the forward FFT, remap, and inverse FFT in order.
    pub(crate) fn record(&self, command: &RecordingCommandBuffer<'_>) {
        self.forward.record(command);
        self.remap.record(command);
        self.inverse.record(command);
    }
}

#[cfg(test)]
mod tests {
    use realfft::RealFftPlanner;
    use realfft::num_complex::Complex;

    use super::*;
    use crate::{Config, GpuContext};

    /// Runs one prepared FFT window (no streaming overlap -- `gpu_plan.md` milestone 5) through
    /// [`GpuTransformPipeline`] on real hardware, and checks it against a from-scratch CPU
    /// reference built directly from `realfft` + `SpectralPlan` (deliberately not reusing
    /// `CpuCore`, so this is an independent check of the same math, not a copy of its call
    /// path).
    fn assert_single_transform_matches_cpu(input_rate: usize, output_rate: usize) {
        let context = match GpuDevice::auto_select() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping GPU single-transform test: {err}");
                return;
            }
        };

        let config = Config::new(input_rate, output_rate, 1);
        let derived = config.derive_config::<f32>().expect("valid config");

        let input_fft_size = derived.input_fft_size;
        let output_fft_size = derived.output_fft_size;
        let input_offset = derived.input_offset;

        let input_hz = 1_000.0f64;
        let mut window = vec![0.0f32; input_fft_size];
        for (i, sample) in window[input_offset..input_offset + derived.input_chunk_frames]
            .iter_mut()
            .enumerate()
        {
            *sample = (2.0 * std::f64::consts::PI * input_hz * i as f64 / input_rate as f64).sin() as f32;
        }

        // --- CPU reference: independent realfft + SpectralPlan round trip ---
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(input_fft_size);
        let inverse = planner.plan_fft_inverse(output_fft_size);
        let mut cpu_window = window.clone();
        let mut cpu_spectrum = forward.make_output_vec();
        forward
            .process(&mut cpu_window, &mut cpu_spectrum)
            .expect("cpu forward fft");
        let mut cpu_remapped = inverse.make_input_vec();
        derived.spectral.apply(&cpu_spectrum, &mut cpu_remapped);
        let mut cpu_output = inverse.make_output_vec();
        inverse
            .process(&mut cpu_remapped, &mut cpu_output)
            .expect("cpu inverse fft");
        let cpu_normalize = 1.0f32 / output_fft_size as f32;
        for sample in &mut cpu_output {
            *sample *= cpu_normalize;
        }

        // --- GPU pipeline: build (Milestone 5), upload, record, submit, download ---
        let geometry = RemapGeometry {
            direction_up: derived.input_chunk_frames < derived.output_chunk_frames,
            n: derived.spectral.geometry.lower_nyquist_bin,
            r0: derived.spectral.geometry.reflect_start_bin(),
            nyquist_fold: if derived.input_chunk_frames > derived.output_chunk_frames {
                2.0
            } else {
                1.0
            },
            gain: &derived.spectral.gain,
            phase: derived
                .spectral
                .phase_enabled
                .then_some(derived.spectral.phase.as_slice()),
        };

        let device = Arc::new(context);
        let context = GpuContext::<f32>::with_device(Arc::clone(&device), config, 1).expect("build GPU context");
        let shaders = context.compiled_shaders().expect("compile shaders");
        let pipeline = GpuTransformPipeline::<f32>::build(
            &device,
            Precision::F32,
            input_fft_size,
            output_fft_size,
            1,
            &geometry,
            &shaders.single,
        )
        .expect("build GPU transform pipeline");

        let mut gpu_input = vec![Complex::new(0.0f32, 0.0f32); pipeline.input_stride()];
        for (dst, &src) in gpu_input.iter_mut().zip(window.iter()) {
            *dst = Complex::new(src, 0.0);
        }
        let gpu_input_flat: &[f32] = bytemuck_cast(&gpu_input);
        pipeline
            .input_buffer()
            .upload(gpu_input_flat)
            .expect("upload gpu input");

        let submission = device
            .submit_async(pipeline, |command, pipeline| pipeline.record(command))
            .map_err(|(err, _)| err)
            .expect("submit GPU transform pipeline");
        let pipeline = submission.resolve().expect("run GPU transform pipeline");

        let mut gpu_output = vec![Complex::new(0.0f32, 0.0f32); pipeline.output_stride()];
        let gpu_output_flat: &mut [f32] = bytemuck_cast_mut(&mut gpu_output);
        pipeline
            .output_buffer()
            .download(gpu_output_flat)
            .expect("download gpu output");

        let mut max_abs_error = 0.0f32;
        for (cpu, gpu) in cpu_output.iter().zip(gpu_output.iter().take(output_fft_size)) {
            assert!(
                gpu.im.abs() < 1e-3,
                "GPU real output has non-negligible imaginary part: {gpu:?}"
            );
            max_abs_error = max_abs_error.max((cpu - gpu.re).abs());
        }
        assert!(
            max_abs_error < 1e-3,
            "{input_rate} -> {output_rate}: GPU/CPU single-transform mismatch, max abs error {max_abs_error}"
        );
    }

    fn bytemuck_cast<T>(values: &[Complex<T>]) -> &[T] {
        // SAFETY: `Complex<T>` is `#[repr(C)]` with two contiguous `T` fields and no padding
        // (see `realfft`'s `num_complex::Complex`), so `n` complex values are exactly `2n`
        // contiguous `T`s.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len() * 2) }
    }

    fn bytemuck_cast_mut<T>(values: &mut [Complex<T>]) -> &mut [T] {
        let len = values.len() * 2;
        // SAFETY: same layout guarantee as `bytemuck_cast`, for a uniquely-borrowed slice.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<T>(), len) }
    }

    #[test]
    fn single_transform_matches_cpu_44100_to_48000() {
        assert_single_transform_matches_cpu(44_100, 48_000);
    }

    #[test]
    fn single_transform_matches_cpu_48000_to_44100() {
        assert_single_transform_matches_cpu(48_000, 44_100);
    }

    #[test]
    fn single_transform_matches_cpu_44100_to_96000() {
        assert_single_transform_matches_cpu(44_100, 96_000);
    }
}
