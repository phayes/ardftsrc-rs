use std::sync::Arc;

use ash::vk;
use vkfft_rs::backend::vulkan::runtime::{VulkanBufferSlice, VulkanComputePipeline};
use vkfft_rs::ScalarType;

use super::buffer::{GpuBuffer, GpuScalar};
use super::context::GpuContext;
use super::error::GpuError;
use super::fft_program::record_compute_barrier;
use super::overlap_shader::{OverlapMode, build_pipeline_with_slices};

/// Sequential-but-single-submission overlap-add assembly for [`super::batch_core::GpuCore`]
/// (`gpu_plan.md` sections 14-16).
///
/// The forward FFT / spectral remap / inverse FFT for *every* window in a batch (one synthetic
/// start window, one real window per input chunk, and one optional synthetic finalize-tail
/// window) run as a single, fully-parallel batched dispatch (`GpuTransformPipeline` with
/// `batch_count = channels * total_windows`) before this type does anything -- that big batched
/// transform is what satisfies "does not need a rolling overlap state during the transform
/// phase" (section 14).
///
/// Turning those independent per-window transforms into a correct overlap-add sequence still
/// requires reading each window's second half back into the next window's first half, in order
/// (`gpu_plan.md` section 16's `Y[k] = A[k] + B[k-1]`). Rather than a single fully-parallel 3D
/// `(sample, channel, chunk)` dispatch, this reuses [`super::overlap_shader`]'s existing
/// Normal/Start/End GLSL bodies and simply builds one small pipeline per window, each bound (via
/// [`VulkanBufferSlice`] byte-range offsets, not separate buffers) to that window's own slice of
/// the big transform-pipeline output and to a shared, tiny persistent overlap buffer -- then
/// [`BatchOverlapShader::record`] dispatches all of them in window order into one command buffer.
/// This is still one GPU submission with no CPU-side serial overlap-add (satisfying "do not
/// perform overlap-add serially on the CPU"), just not maximally parallel across windows; a fully
/// parallel 3D dispatch is a possible follow-up if benchmarking (`gpu_plan.md` milestone 9) shows
/// this dominates.
pub(crate) struct BatchOverlapShader<T> {
    /// One pipeline per window, in window order (window 0 is always `OverlapMode::Start`;
    /// windows `1..=num_real_chunks` are always `OverlapMode::Normal`; an optional final window
    /// is `OverlapMode::End`), each already bound to its own slice of the transform pipeline's
    /// output buffer and to `output`/`overlap` below.
    pipelines: Vec<VulkanComputePipeline>,
    /// Real per-chunk output, `channels * num_real_chunks * output_chunk_frames` real `T`s,
    /// laid out `[chunk][channel][sample]`; only `OverlapMode::Normal` windows write into it (at
    /// their own `(chunk_index * channels * output_chunk_frames)` offset).
    output: GpuBuffer<T>,
    /// Persistent overlap state shared and updated in place by every window's dispatch, in
    /// order -- `channels * output_chunk_frames` real `T`s, laid out `[channel][sample]`.
    overlap: GpuBuffer<T>,
    /// Throwaway target for `OverlapMode::Start`/`End` dispatches, which write zeros into their
    /// bound output buffer but never have their output read; reused across every such window
    /// instead of allocating one per window.
    scratch_output: GpuBuffer<T>,
    num_real_chunks: usize,
    output_chunk_frames: usize,
    channels: usize,
}

impl<T: GpuScalar> BatchOverlapShader<T> {
    /// Builds one pipeline per entry in `window_modes` (window 0 must be `Start`; the only
    /// `End` entry, if present, must be last; every other entry must be `Normal`), each bound to
    /// its own `channels`-wide slice of `ifft_output` (a big buffer holding
    /// `channels * window_modes.len()` forward/remap/inverse FFT results, one per window, laid
    /// out `[window][channel]`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        context: &Arc<GpuContext>,
        scalar: ScalarType,
        ifft_output: &GpuBuffer<T>,
        ifft_stride: usize,
        output_chunk_frames: usize,
        input_chunk_frames: usize,
        channels: usize,
        window_modes: &[OverlapMode],
    ) -> Result<Self, GpuError> {
        let num_real_chunks = window_modes.iter().filter(|mode| **mode == OverlapMode::Normal).count();

        let overlap = GpuBuffer::<T>::new(context, channels * output_chunk_frames, vk::BufferUsageFlags::empty())?;
        overlap.upload(&vec![T::default(); overlap.len()])?;
        let scratch_output = GpuBuffer::<T>::new(context, channels * output_chunk_frames, vk::BufferUsageFlags::empty())?;
        let output = GpuBuffer::<T>::new(context, channels * num_real_chunks * output_chunk_frames, vk::BufferUsageFlags::empty())?;

        let elem_size = std::mem::size_of::<T>() as vk::DeviceSize;
        let window_stride_bytes = (channels as vk::DeviceSize) * (ifft_stride as vk::DeviceSize) * 2 * elem_size;
        let chunk_output_bytes = (channels as vk::DeviceSize) * (output_chunk_frames as vk::DeviceSize) * elem_size;
        let overlap_slice = VulkanBufferSlice::whole(overlap.handle());
        let scratch_output_slice = VulkanBufferSlice::whole(scratch_output.handle());

        let mut pipelines = Vec::with_capacity(window_modes.len());
        let mut chunk_index = 0;
        for (window_index, &mode) in window_modes.iter().enumerate() {
            let ifft_slice = VulkanBufferSlice {
                buffer: ifft_output.handle(),
                offset: window_index as vk::DeviceSize * window_stride_bytes,
                range: window_stride_bytes,
            };
            let output_slice = if mode == OverlapMode::Normal {
                let slice = VulkanBufferSlice {
                    buffer: output.handle(),
                    offset: chunk_index as vk::DeviceSize * chunk_output_bytes,
                    range: chunk_output_bytes,
                };
                chunk_index += 1;
                slice
            } else {
                scratch_output_slice
            };

            let pipeline = build_pipeline_with_slices::<T>(
                context,
                scalar,
                mode,
                ifft_slice,
                ifft_stride,
                output_chunk_frames,
                input_chunk_frames,
                channels,
                output_slice,
                overlap_slice,
            )?;
            pipelines.push(pipeline);
        }

        Ok(Self {
            pipelines,
            output,
            overlap,
            scratch_output,
            num_real_chunks,
            output_chunk_frames,
            channels,
        })
    }

    /// The assembled per-chunk output buffer, `channels * num_real_chunks * output_chunk_frames`
    /// real `T`s laid out `[chunk][channel][sample]`.
    pub(crate) fn output_buffer(&self) -> &GpuBuffer<T> {
        &self.output
    }

    /// The final overlap state after every window has been dispatched -- the finalize-tail
    /// output, `channels * output_chunk_frames` real `T`s laid out `[channel][sample]`, read the
    /// same way `CpuCore::finalize` reads its own `overlap` field directly.
    pub(crate) fn overlap_buffer(&self) -> &GpuBuffer<T> {
        &self.overlap
    }

    pub(crate) fn num_real_chunks(&self) -> usize {
        self.num_real_chunks
    }

    pub(crate) fn output_chunk_frames(&self) -> usize {
        self.output_chunk_frames
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    /// Total GPU-buffer bytes this shader owns (`overlap` + `scratch_output` + `output`), for
    /// sizing a `GpuCore` ring slot against a memory budget.
    pub(crate) fn total_bytes(&self) -> u64 {
        self.overlap.byte_len() + self.scratch_output.byte_len() + self.output.byte_len()
    }

    /// Records a device-to-device copy of `src` (another `BatchOverlapShader`'s
    /// [`BatchOverlapShader::overlap_buffer`], the previous group's ending overlap state) into
    /// this shader's own `overlap` buffer, followed by a barrier making that write visible to
    /// the compute dispatches [`BatchOverlapShader::record`] appends after it in the same
    /// command buffer. This is how the pipelined `GpuCore` carries overlap state from one
    /// group's GPU submission into the next without ever downloading it to the host: the whole
    /// carry stays device-resident, recorded as the first commands of the *next* group's own
    /// submission rather than a separate round-trip.
    ///
    /// Must be recorded before [`BatchOverlapShader::record`] in the same command buffer.
    pub(crate) fn record_seed_copy(&self, device: &ash::Device, command_buffer: vk::CommandBuffer, src: &GpuBuffer<T>) {
        let region = vk::BufferCopy::default().size(self.overlap.byte_len());
        // SAFETY: `command_buffer` is in the recording state (caller contract); `src` and
        // `self.overlap` are both live buffers of identical length (every `BatchOverlapShader`
        // for a given `GpuCore` is built with the same `channels`/`output_chunk_frames`).
        unsafe { device.cmd_copy_buffer(command_buffer, src.handle(), self.overlap.handle(), &[region]) };

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: `command_buffer` is in the recording state; this makes the copy above visible
        // to the compute dispatches recorded after it.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
    }

    /// Records every window's dispatch, in order, each followed by a full compute barrier so
    /// the next window's read of `overlap` observes the previous window's write.
    pub(crate) fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        for pipeline in &self.pipelines {
            // SAFETY: `command_buffer` is in the recording state (caller contract); every
            // pipeline was fully built (pipeline + bound descriptor set) in `build` above.
            unsafe { pipeline.record_dispatch(command_buffer) };
            record_compute_barrier(device, command_buffer);
        }
    }
}
