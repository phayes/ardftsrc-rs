use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;

use super::context::GpuDevice;
use super::error::GpuError;

/// A device-local Vulkan storage buffer holding `len` values of `T` (`f32` or `f64`).
///
/// `T` is a plain scalar: a complex-valued buffer of `n` complex numbers is a `GpuBuffer<T>`
/// of `len == n * 2` (matching `vkfft-rs`'s `Complex<T> { re, im }` interleaved layout, which
/// is `#[repr(C)]` and therefore just two contiguous `T`s per element).
///
/// On a device where the memory backing this buffer is *also* host-visible (true unified-memory
/// hardware, e.g. Apple Silicon via MoltenVK -- as opposed to a discrete GPU with separate VRAM),
/// [`GpuBuffer::new`] maps it once, persistently, and [`GpuBuffer::upload`]/[`GpuBuffer::download`]
/// become a plain host `memcpy` with no GPU submission, no fence wait, and no transient staging
/// allocation at all. Otherwise they fall back to the "conventional staging buffer" path
/// `gpu_plan.md` explicitly allows for the initial implementation: a transient host-visible
/// staging buffer and an immediate command-buffer copy, submitted and waited on synchronously.
/// Callers never see which path is in use -- the API shape is identical either way.
///
/// Holds `Arc<GpuDevice>` (not a bare `Arc<ash::Device>`) so that, as long as every struct
/// embedding a `GpuBuffer<T>` also declares it before any `Arc<GpuDevice>` field of its own,
/// Rust's declaration-order field drop takes care of destroying this buffer before the
/// context it was built from.
pub(crate) struct GpuBuffer<T> {
    allocation: Arc<BufferAllocation>,
    len: usize,
    accessible: Cell<bool>,
    _marker: PhantomData<T>,
}

/// Shared ownership of the Vulkan allocation itself. Synchronous copies clone this into their
/// submission payload, so an indeterminate fence result retains the memory independently of the
/// `GpuBuffer` handle that initiated the operation.
struct BufferAllocation {
    context: Arc<GpuDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Present only when `memory` is host-visible: a persistent mapping used by
    /// [`GpuBuffer::upload`]/[`GpuBuffer::download`] instead of the staging-buffer path.
    mapped: Option<MappedMemory>,
}

// SAFETY: Vulkan object ownership may move between threads, and access to the mapped pointer is
// exposed only through `GpuBuffer`, which is deliberately `!Sync`. Other `Arc` clones exist only
// as opaque lifetime guards owned by submissions and never dereference the mapping.
unsafe impl Send for BufferAllocation {}
// SAFETY: shared references held by submission lifetime guards do not access or mutate the
// allocation; all actual mapped-memory access remains serialized by `GpuBuffer`'s `!Sync` API.
unsafe impl Sync for BufferAllocation {}

/// A persistent mapping of a [`GpuBuffer`]'s own device memory, used only when that memory is
/// host-visible (see [`GpuBuffer`]'s own doc).
struct MappedMemory {
    ptr: std::ptr::NonNull<u8>,
    /// Whether the memory type is also `HOST_COHERENT`; if not, every host read/write must be
    /// bracketed with `vkInvalidateMappedMemoryRanges`/`vkFlushMappedMemoryRanges`.
    coherent: bool,
}

// SAFETY: `MappedMemory::ptr` uniquely maps this `GpuBuffer`'s own device memory, which nothing
// else holds a reference to; moving a `GpuBuffer` (and therefore this pointer) to another thread
// is sound as long as it is not used concurrently from two threads at once. `GpuBuffer`
// deliberately does *not* implement `Sync`: unlike the staging-buffer path (serialized through
// `GpuDevice`'s own `execution_lock`), a direct mapped read/write has no such serialization, so
// concurrent `upload`/`download` calls from multiple threads sharing one `&GpuBuffer` would race
// -- `Send` alone (ownership transfer, never concurrent access) is safe.
unsafe impl<T> Send for GpuBuffer<T> {}

/// Marker for scalar element types a [`GpuBuffer`] can hold.
///
/// `scalar_type`/`precision` let generic GPU-core code recover which concrete `vkfft-rs` type
/// tags correspond to a compile-time-generic `T`, without needing every call site to pass them
/// separately (and risk passing a pair that doesn't actually match `T`).
pub(crate) trait GpuScalar: Copy + Default + bytemuck_like::Pod + 'static {
    fn scalar_type() -> vkfft_rs::ScalarType;
    fn precision() -> vkfft_rs::Precision;
}
impl GpuScalar for f32 {
    fn scalar_type() -> vkfft_rs::ScalarType {
        vkfft_rs::ScalarType::F32
    }
    fn precision() -> vkfft_rs::Precision {
        vkfft_rs::Precision::F32
    }
}
impl GpuScalar for f64 {
    fn scalar_type() -> vkfft_rs::ScalarType {
        vkfft_rs::ScalarType::F64
    }
    fn precision() -> vkfft_rs::Precision {
        vkfft_rs::Precision::F64
    }
}

/// A tiny local stand-in for the parts of `bytemuck::Pod` this module needs: `T` must be
/// safely readable/writable as raw bytes (true for `f32`/`f64`, the only types this module
/// instantiates `GpuBuffer` with).
mod bytemuck_like {
    /// # Safety
    ///
    /// Implementors must have no padding and be valid for any bit pattern, so a `&[Self]`
    /// may be reinterpreted as `&[u8]` (and vice versa) freely.
    pub(crate) unsafe trait Pod: Sized {}
    // SAFETY: `f32`/`f64` are IEEE-754 values with no padding; every bit pattern (including
    // NaNs/infinities) is a valid value of the type.
    unsafe impl Pod for f32 {}
    unsafe impl Pod for f64 {}
}

fn as_bytes<T: GpuScalar>(values: &[T]) -> &[u8] {
    // SAFETY: `T: GpuScalar` guarantees `T` has no padding and every bit pattern is valid, so
    // reinterpreting `&[T]` as `&[u8]` is sound; the resulting slice's lifetime/length are
    // derived correctly from the source slice.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn as_bytes_mut<T: GpuScalar>(values: &mut [T]) -> &mut [u8] {
    let len_bytes = std::mem::size_of_val(values);
    // SAFETY: same as `as_bytes`, for a mutable, uniquely-borrowed slice.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len_bytes) }
}

/// Finds a memory type index matching `type_bits` (from `VkMemoryRequirements`) that has all
/// of `required` set.
fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory_properties.memory_type_count).find(|&index| {
        let type_supported = (type_bits & (1 << index)) != 0;
        let flags_supported = memory_properties.memory_types[index as usize]
            .property_flags
            .contains(required);
        type_supported && flags_supported
    })
}

impl<T: GpuScalar> GpuBuffer<T> {
    /// Allocates a new device-local storage buffer holding `len` values of `T`, uninitialized.
    ///
    /// `len` is a plain element count (see the type-level doc for the complex-buffer
    /// convention); `usage` is ORed with `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST`, which
    /// every `GpuBuffer` needs for shader binding and staged upload/download.
    pub(crate) fn new(context: &Arc<GpuDevice>, len: usize, usage: vk::BufferUsageFlags) -> Result<Self, GpuError> {
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| GpuError::AllocationFailed("buffer element count overflows byte size".to_string()))?;
        // Vulkan buffers of size zero are invalid; a zero-length `GpuBuffer` should not be
        // constructed by callers, but round up defensively rather than let driver-specific
        // behavior decide what happens.
        let byte_len = byte_len.max(1) as vk::DeviceSize;

        let device = context.device();
        let usage = usage
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(byte_len)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `device` is a live logical device owned by `context`, which this `GpuBuffer`
        // keeps alive via its own `Arc<GpuDevice>` field.
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|err| GpuError::AllocationFailed(format!("failed to create Vulkan buffer: {err}")))?;

        // SAFETY: `buffer` was just created successfully on this same `device`.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Prefer a memory type that is both `DEVICE_LOCAL` and `HOST_VISIBLE` -- true on unified
        // memory hardware (e.g. Apple Silicon via MoltenVK), never on a discrete GPU with
        // separate VRAM. Finding one is what lets `upload`/`download` skip the staging-buffer
        // path entirely (see the type's own doc).
        let host_visible_device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        let direct = find_memory_type(
            context.memory_properties(),
            requirements.memory_type_bits,
            host_visible_device_local,
        )
        .map(|index| {
            (
                index,
                context.memory_properties().memory_types[index as usize].property_flags,
            )
        });

        let memory_type = direct.map(|(index, _)| index).or_else(|| {
            find_memory_type(
                context.memory_properties(),
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                // Every Vulkan implementation guarantees at least one memory type with no
                // required properties; this is the last-resort fallback for a device with no
                // dedicated device-local heap (uncommon, but defensive).
                find_memory_type(
                    context.memory_properties(),
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
        });
        let Some(memory_type) = memory_type else {
            // SAFETY: `buffer` has no bound memory yet, so destroying it here is sound.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(GpuError::AllocationFailed(
                "no suitable Vulkan memory type for buffer".to_string(),
            ));
        };

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: `device` is live; `allocate_info` describes a size/type consistent with
        // `requirements`.
        let memory = match unsafe { device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: `buffer` has no bound memory yet.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(GpuError::AllocationFailed(format!(
                    "failed to allocate Vulkan device memory: {err}"
                )));
            }
        };

        // SAFETY: `buffer` and `memory` were just created on this same `device`, `memory` is
        // large enough per `requirements`, and neither has been bound/freed yet.
        if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: neither object has any other outstanding use yet.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(GpuError::AllocationFailed(format!(
                "failed to bind Vulkan buffer memory: {err}"
            )));
        }

        let mapped = match direct {
            Some((index, flags)) if index == memory_type => {
                // SAFETY: `memory` was just allocated with a `HOST_VISIBLE` type and bound to
                // `buffer` above; it is not mapped anywhere else yet.
                match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
                    Ok(ptr) => std::ptr::NonNull::new(ptr.cast::<u8>()).map(|ptr| MappedMemory {
                        ptr,
                        coherent: flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                    }),
                    // Mapping is an optimization, not a correctness requirement (`gpu_plan.md`
                    // explicitly allows the staging-buffer path) -- fail soft to it rather than
                    // erroring the whole allocation over a failed map of otherwise-good memory.
                    Err(_) => None,
                }
            }
            _ => None,
        };

        Ok(Self {
            allocation: Arc::new(BufferAllocation {
                context: Arc::clone(context),
                buffer,
                memory,
                mapped,
            }),
            len,
            accessible: Cell::new(true),
            _marker: PhantomData,
        })
    }

    /// Number of `T` elements this buffer holds.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Bytes this buffer occupies (`len() * size_of::<T>()`), for GPU-memory budgeting.
    pub(crate) fn byte_len(&self) -> u64 {
        (self.len * std::mem::size_of::<T>()) as u64
    }

    /// The raw Vulkan buffer handle, for binding into a compute pipeline's descriptor set.
    pub(crate) fn handle(&self) -> vk::Buffer {
        self.allocation.buffer
    }

    /// Uploads `values` (which must have length [`GpuBuffer::len`]) into this buffer. On
    /// host-visible memory (see the type's own doc), this is a plain `memcpy` into the
    /// persistent mapping with no GPU submission at all; otherwise it goes through a transient
    /// host-visible staging buffer, a one-shot command buffer, and a synchronous fence wait.
    pub(crate) fn upload(&self, values: &[T]) -> Result<(), GpuError> {
        self.ensure_accessible()?;
        if values.len() != self.len {
            return Err(GpuError::AllocationFailed(format!(
                "upload length {} does not match buffer length {}",
                values.len(),
                self.len
            )));
        }
        if let Some(mapped) = &self.allocation.mapped {
            let bytes = as_bytes(values);
            // SAFETY: `mapped.ptr` is valid for this buffer's whole byte length (mapped at
            // construction and never unmapped until `Drop`); `bytes` is exactly that many bytes
            // (checked above); the source (host-owned `values`) and destination (a distinct
            // device-memory mapping) cannot alias. No GPU work reads this memory until a command
            // buffer referencing it is submitted, which every caller sequences after this call.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.ptr.as_ptr(), bytes.len()) };
            if !mapped.coherent {
                self.flush_or_invalidate(true)?;
            }
            return Ok(());
        }
        let staging = StagingBuffer::new(&self.allocation.context, as_bytes(values))?;
        self.copy_with_staging(staging, true).map(|_| ())
    }

    /// Downloads this buffer's contents into `out` (which must have length [`GpuBuffer::len`]).
    /// On host-visible memory (see the type's own doc), this is a plain `memcpy` from the
    /// persistent mapping with no GPU submission at all; otherwise it goes through a transient
    /// host-visible staging buffer, a one-shot command buffer, and a synchronous fence wait.
    pub(crate) fn download(&self, out: &mut [T]) -> Result<(), GpuError> {
        self.ensure_accessible()?;
        if out.len() != self.len {
            return Err(GpuError::AllocationFailed(format!(
                "download length {} does not match buffer length {}",
                out.len(),
                self.len
            )));
        }
        if let Some(mapped) = &self.allocation.mapped {
            if !mapped.coherent {
                self.flush_or_invalidate(false)?;
            }
            let bytes = as_bytes_mut(out);
            // SAFETY: same reasoning as `upload`'s mapped path, in reverse; the caller is
            // responsible (as with the staging path) for only calling this once the GPU work
            // that wrote this buffer has actually completed (e.g. after a fence wait).
            unsafe { std::ptr::copy_nonoverlapping(mapped.ptr.as_ptr(), bytes.as_mut_ptr(), bytes.len()) };
            return Ok(());
        }
        let staging =
            StagingBuffer::new_uninit(&self.allocation.context, std::mem::size_of_val(out) as vk::DeviceSize)?;
        let staging = self.copy_with_staging(staging, false)?;
        staging.read_into(as_bytes_mut(out))
    }

    fn ensure_accessible(&self) -> Result<(), GpuError> {
        if self.accessible.get() {
            Ok(())
        } else {
            Err(GpuError::InvalidSubmissionState(
                "GPU buffer completion is unknown after a failed fence wait".to_string(),
            ))
        }
    }

    fn copy_with_staging(&self, staging: StagingBuffer, upload: bool) -> Result<StagingBuffer, GpuError> {
        let resources = CopyResources {
            allocation: Arc::clone(&self.allocation),
            staging,
            upload,
        };
        let submission = match self.allocation.context.submit_async(resources, |command, resources| {
            let (src, dst) = if resources.upload {
                (resources.staging.buffer, resources.allocation.buffer)
            } else {
                (resources.allocation.buffer, resources.staging.buffer)
            };
            // SAFETY: both allocations are owned by `resources`, belong to `command`'s
            // device, include transfer usage, and have the same checked logical byte length.
            unsafe { command.copy_raw_buffer(src, dst, resources.staging.byte_len) };
        }) {
            Ok(submission) => submission,
            Err((err, _resources)) => return Err(err),
        };
        match submission.resolve() {
            Ok(resources) => Ok(resources.staging),
            Err(err) => {
                self.accessible.set(false);
                Err(err)
            }
        }
    }

    /// Flushes or invalidates the whole persistently mapped allocation. `VK_WHOLE_SIZE` avoids
    /// a logical buffer length that is not aligned to `nonCoherentAtomSize`; the mapping starts
    /// at offset zero, which satisfies the corresponding offset requirement.
    fn flush_or_invalidate(&self, is_write: bool) -> Result<(), GpuError> {
        let range = vk::MappedMemoryRange::default()
            .memory(self.allocation.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        let ranges = [range];
        // SAFETY: the allocation is currently mapped (guaranteed by `mapped` being `Some` at
        // both call sites), and the range covers that whole mapping.
        let result = if is_write {
            unsafe { self.allocation.context.device().flush_mapped_memory_ranges(&ranges) }
        } else {
            unsafe {
                self.allocation
                    .context
                    .device()
                    .invalidate_mapped_memory_ranges(&ranges)
            }
        };
        result.map_err(|err| GpuError::ExecutionFailed(format!("failed to synchronize mapped Vulkan memory: {err}")))
    }
}

impl Drop for BufferAllocation {
    fn drop(&mut self) {
        let device = self.context.device();
        if self.mapped.is_some() {
            // SAFETY: `self.memory` is currently mapped and about to be freed; unmapping first
            // is the documented-clean teardown order (freeing alone implicitly unmaps, but this
            // avoids relying on that and keeps validation layers quiet).
            unsafe { device.unmap_memory(self.memory) };
        }
        // SAFETY: the final `Arc<BufferAllocation>` cannot be released while a submission owns
        // an allocation lifetime guard, so no GPU work can still reference these objects.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A transient host-visible buffer used only to stage data across the host/device boundary
/// for one [`GpuBuffer::upload`]/[`GpuBuffer::download`] call.
struct StagingBuffer {
    context: Arc<GpuDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    byte_len: vk::DeviceSize,
}

struct CopyResources {
    allocation: Arc<BufferAllocation>,
    staging: StagingBuffer,
    upload: bool,
}

impl StagingBuffer {
    fn new_uninit(context: &Arc<GpuDevice>, byte_len: vk::DeviceSize) -> Result<Self, GpuError> {
        let device = context.device();
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(byte_len.max(1))
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `device` is live for the duration of this call (kept alive by `context`).
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|err| GpuError::AllocationFailed(format!("failed to create Vulkan staging buffer: {err}")))?;

        // SAFETY: `buffer` was just created successfully on this same `device`.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let required = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(memory_type) = find_memory_type(context.memory_properties(), requirements.memory_type_bits, required)
        else {
            // SAFETY: `buffer` has no bound memory yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(GpuError::AllocationFailed(
                "no host-visible/coherent Vulkan memory type for staging".to_string(),
            ));
        };

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: `device` is live; sizes/types are consistent with `requirements`.
        let memory = match unsafe { device.allocate_memory(&allocate_info, None) } {
            Ok(memory) => memory,
            Err(err) => {
                // SAFETY: `buffer` has no bound memory yet.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(GpuError::AllocationFailed(format!(
                    "failed to allocate Vulkan staging memory: {err}"
                )));
            }
        };

        // SAFETY: `buffer`/`memory` were just created on this device and neither is bound yet.
        if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: neither object has any other outstanding use yet.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(GpuError::AllocationFailed(format!(
                "failed to bind Vulkan staging memory: {err}"
            )));
        }

        Ok(Self {
            context: Arc::clone(context),
            buffer,
            memory,
            byte_len,
        })
    }

    fn new(context: &Arc<GpuDevice>, initial_data: &[u8]) -> Result<Self, GpuError> {
        let staging = Self::new_uninit(context, initial_data.len() as vk::DeviceSize)?;
        staging.write(initial_data)?;
        Ok(staging)
    }

    fn write(&self, data: &[u8]) -> Result<(), GpuError> {
        let device = self.context.device();
        // SAFETY: `memory` is host-visible/coherent (guaranteed at construction), is not
        // currently mapped elsewhere, and `data.len()` was used as this buffer's own byte
        // length, so the mapped range covers the whole buffer.
        let ptr = unsafe { device.map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
            .map_err(|err| GpuError::AllocationFailed(format!("failed to map Vulkan staging memory: {err}")))?;
        // SAFETY: `ptr` is valid for `self.byte_len` bytes (the size just mapped), and `data`
        // is exactly that many bytes; the regions do not overlap (host memory vs. a fresh
        // device-memory mapping).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len()) };
        // SAFETY: `self.memory` is the same mapped allocation.
        unsafe { device.unmap_memory(self.memory) };
        Ok(())
    }

    fn read_into(&self, out: &mut [u8]) -> Result<(), GpuError> {
        let device = self.context.device();
        // SAFETY: same reasoning as `write`: `memory` is host-visible/coherent and unmapped.
        let ptr = unsafe { device.map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
            .map_err(|err| GpuError::AllocationFailed(format!("failed to map Vulkan staging memory: {err}")))?;
        // SAFETY: `ptr` is valid for `self.byte_len` bytes, `out` is exactly that many bytes
        // (checked by the caller), and the regions do not overlap.
        unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), out.len()) };
        // SAFETY: `self.memory` is the same mapped allocation.
        unsafe { device.unmap_memory(self.memory) };
        Ok(())
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        let device = self.context.device();
        // SAFETY: this `StagingBuffer` is the sole owner of `buffer`/`memory`, and any copy
        // command referencing them has already been waited on by the time this drops (see
        // `GpuDevice::copy_buffer`).
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms the direct host-visible-mapped path (not just the staging-buffer fallback)
    /// actually round-trips data correctly, and reports whether this hardware took it at all --
    /// on unified-memory hardware (this M1 Max dev machine included) it should.
    #[test]
    fn upload_download_round_trips_on_whichever_path_this_device_takes() {
        let context = match GpuDevice::auto_select() {
            Ok(context) => Arc::new(context),
            Err(err) => {
                eprintln!("skipping GPU buffer test: {err}");
                return;
            }
        };

        let buffer = GpuBuffer::<f32>::new(&context, 1024, vk::BufferUsageFlags::empty()).expect("allocate buffer");
        eprintln!(
            "GpuBuffer<f32> on {}: direct-mapped = {}",
            context.device_name(),
            buffer.allocation.mapped.is_some()
        );

        let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5 - 17.0).collect();
        buffer.upload(&values).expect("upload");
        let mut readback = vec![0.0f32; 1024];
        buffer.download(&mut readback).expect("download");
        assert_eq!(values, readback);

        // A second round trip, to make sure a mapped buffer doesn't require re-mapping and
        // stays correct across repeated use (the realistic usage pattern: one `GpuBuffer` is
        // upload/downloaded many times over its life).
        let more_values: Vec<f32> = (0..1024).map(|i| -(i as f32) * 2.25 + 3.0).collect();
        buffer.upload(&more_values).expect("second upload");
        buffer.download(&mut readback).expect("second download");
        assert_eq!(more_values, readback);
    }
}
