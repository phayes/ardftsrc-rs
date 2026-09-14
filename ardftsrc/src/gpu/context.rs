use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ash::vk;
use vkfft_rs::{Backend, DeviceProfile, GpuVendor};

use super::error::GpuError;

/// Capabilities of the Vulkan device backing a [`GpuContext`].
///
/// GPU cores must consult this before executing `f64` work rather than assuming support:
/// notably, Apple GPUs (Metal via MoltenVK) never report `shader_float64` support, since
/// Metal has no double-precision shader type at all.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuCapabilities {
    /// Whether the device can execute compute shaders using 64-bit floating point.
    pub shader_float64: bool,
    /// Largest range (in bytes) a single storage buffer descriptor may cover.
    pub max_storage_buffer_range: u64,
    /// Maximum compute workgroup *count* (dispatch grid dimensions), per axis.
    pub max_compute_workgroup_count: [u32; 3],
    /// Maximum compute workgroup *size* (local invocations), per axis.
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum total invocations in one compute workgroup.
    pub max_compute_work_group_invocations: u32,
    /// Human-readable device name, for diagnostics/logging.
    pub device_name: String,
}

/// Owns the long-lived Vulkan objects shared by all GPU cores.
///
/// This is infrastructure only: it does not know about ARDFTSRC's DSP geometry, FFT plans,
/// or shaders. GPU cores are constructed with an `Arc<GpuContext>` and build their own
/// FFT plans/buffers/pipelines on top of it, using `vkfft-rs`'s public planner/shader-lowering
/// API (`ProgramIr`, `VulkanGlslBackend::lower_real_fft`, `VulkanComputePipeline`) against
/// buffers and a command stream that this crate owns -- not `vkfft-rs`'s own hidden
/// `VulkanExecutionContext`. This is what lets a GPU core record its own custom
/// spectral-remap/overlap-add shaders in between the forward and inverse FFT passes within a
/// single command buffer submission, with no host round-trip in between.
pub struct GpuContext {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: Arc<ash::Device>,
    compute_queue: vk::Queue,
    queue_family: u32,
    command_pool: vk::CommandPool,
    pipeline_cache: vk::PipelineCache,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device_profile: DeviceProfile,
    capabilities: GpuCapabilities,
    /// Serializes command-pool/queue use across `run_one_shot` calls: `vk::CommandPool` is not
    /// safe to record from concurrently, and `vkQueueSubmit` on the same queue needs external
    /// synchronization too. `vkfft-rs`'s own `VulkanExecutionContext` guards its queue/pool the
    /// same way, for the same reason.
    execution_lock: Mutex<()>,
}

impl GpuContext {
    /// Creates a `GpuContext` on the best available Vulkan 1.2+ compute-capable device.
    ///
    /// Returns [`GpuError::VulkanInitFailed`] if the Vulkan loader/instance cannot be
    /// created, or [`GpuError::NoCompatibleDevice`] if no physical device exposes a compute
    /// queue family. This does *not* fail just because the device lacks `shaderFloat64`;
    /// callers that need `f64` should check [`GpuContext::capabilities`] (or call
    /// [`GpuContext::require_f64`]) themselves, so non-f64 GPU work can still proceed.
    pub fn new() -> Result<Self, GpuError> {
        // SAFETY: loading the Vulkan loader is inherently unsafe (it dynamically loads a
        // system library); we immediately check the result rather than assuming success.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|err| GpuError::VulkanInitFailed(format!("failed to load Vulkan loader: {err}")))?;

        let app_name = CString::new("ardftsrc").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .api_version(vk::API_VERSION_1_2);

        // MoltenVK (macOS/iOS) implements Vulkan as a "portability" implementation: since
        // Vulkan 1.3.216 loaders require VK_KHR_portability_enumeration to even enumerate
        // such devices, and the corresponding VK_KHR_portability_subset device extension is
        // mandatory to enable once such a device is selected.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let instance_extensions: Vec<*const std::os::raw::c_char> =
            vec![ash::khr::portability_enumeration::NAME.as_ptr()];
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let instance_extensions: Vec<*const std::os::raw::c_char> = Vec::new();

        #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(unused_mut))]
        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            instance_create_info = instance_create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        // SAFETY: `entry` was just loaded successfully above, and `instance_create_info`
        // borrows only locals that outlive this call.
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .map_err(|err| GpuError::VulkanInitFailed(format!("failed to create Vulkan instance: {err}")))?;

        let selected = match Self::select_physical_device(&instance) {
            Ok(selected) => selected,
            Err(err) => {
                // SAFETY: no device/other child objects were created from this instance yet.
                unsafe { instance.destroy_instance(None) };
                return Err(err);
            }
        };

        match Self::finish_new(entry, instance, selected) {
            Ok(context) => Ok(context),
            Err(boxed) => {
                let (instance, err) = *boxed;
                // SAFETY: only the instance itself needs cleanup; device (if any) creation
                // failed inside `finish_new`, which cleans up after itself on error.
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    fn finish_new(
        entry: ash::Entry,
        instance: ash::Instance,
        selected: SelectedDevice,
    ) -> Result<Self, Box<(ash::Instance, GpuError)>> {
        let SelectedDevice {
            physical_device,
            queue_family,
            capabilities,
            supports_portability_subset,
            vendor,
        } = selected;

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        let queue_create_infos = [queue_create_info];

        let mut device_extensions: Vec<*const std::os::raw::c_char> = Vec::new();
        if supports_portability_subset {
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let enabled_features =
            vk::PhysicalDeviceFeatures::default().shader_float64(capabilities.shader_float64);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&enabled_features);

        // SAFETY: `physical_device` came from this same `instance`'s enumeration, and all
        // borrowed slices/refs outlive this call.
        let device = match unsafe { instance.create_device(physical_device, &device_create_info, None) } {
            Ok(device) => device,
            Err(err) => {
                return Err(Box::new((
                    instance,
                    GpuError::DeviceCreationFailed(format!("failed to create Vulkan logical device: {err}")),
                )));
            }
        };

        // SAFETY: queue index 0 was requested (and is available) in `queue_create_info` above.
        let compute_queue = unsafe { device.get_device_queue(queue_family, 0) };

        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        // SAFETY: `device` was just created successfully above.
        let command_pool = match unsafe { device.create_command_pool(&command_pool_create_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                // SAFETY: `device` has no other child objects yet.
                unsafe { device.destroy_device(None) };
                return Err(Box::new((
                    instance,
                    GpuError::AllocationFailed(format!("failed to create Vulkan command pool: {err}")),
                )));
            }
        };

        let pipeline_cache_create_info = vk::PipelineCacheCreateInfo::default();
        // SAFETY: `device` was just created successfully above.
        let pipeline_cache = match unsafe { device.create_pipeline_cache(&pipeline_cache_create_info, None) } {
            Ok(cache) => cache,
            Err(err) => {
                // SAFETY: `command_pool` is the only other child object created on this
                // device so far.
                unsafe {
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_device(None);
                }
                return Err(Box::new((
                    instance,
                    GpuError::AllocationFailed(format!("failed to create Vulkan pipeline cache: {err}")),
                )));
            }
        };

        // SAFETY: `physical_device` came from this same `instance`'s enumeration above.
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // `DeviceProfile::generic` is `vkfft-rs`'s documented fail-soft portable profile: any
        // backend/vendor pair without a recognized fixed-upstream scheduler profile uses this
        // instead of borrowing another vendor's heuristics. Its `supports_f64` is *not* real
        // hardware truth (see `GpuCapabilities::shader_float64`/`GpuContext::require_f64` for
        // that); it only gates which precision `RealFftIr::build` is willing to plan for, so it
        // must mirror what we actually queried, not the backend-wide default.
        let device_profile = DeviceProfile {
            supports_f64: capabilities.shader_float64,
            ..DeviceProfile::generic(Backend::Vulkan, vendor)
        };

        let device = Arc::new(device);

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family,
            command_pool,
            pipeline_cache,
            memory_properties,
            device_profile,
            capabilities,
            execution_lock: Mutex::new(()),
        })
    }

    /// Picks a physical device with a compute-capable queue family, preferring discrete GPUs.
    fn select_physical_device(instance: &ash::Instance) -> Result<SelectedDevice, GpuError> {
        // SAFETY: `instance` is valid and was just created successfully by the caller.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|err| GpuError::VulkanInitFailed(format!("failed to enumerate Vulkan devices: {err}")))?;

        let mut best: Option<(u32, SelectedDevice)> = None;
        for physical_device in physical_devices {
            // SAFETY: `physical_device` came from `instance`'s own enumeration above.
            let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let Some(queue_family) = queue_families
                .iter()
                .position(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|index| index as u32)
            else {
                continue;
            };

            // SAFETY: `physical_device` is valid, from the same instance.
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            // SAFETY: `physical_device` is valid, from the same instance.
            let features = unsafe { instance.get_physical_device_features(physical_device) };
            let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .unwrap_or_default();
            let supports_portability_subset = extensions.iter().any(|ext| {
                ext.extension_name_as_c_str().ok() == Some(ash::khr::portability_subset::NAME)
            });

            let device_name = properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unknown Vulkan device>".to_string());

            let capabilities = GpuCapabilities {
                shader_float64: features.shader_float64 == vk::TRUE,
                max_storage_buffer_range: properties.limits.max_storage_buffer_range as u64,
                max_compute_workgroup_count: properties.limits.max_compute_work_group_count,
                max_compute_workgroup_size: properties.limits.max_compute_work_group_size,
                max_compute_work_group_invocations: properties.limits.max_compute_work_group_invocations,
                device_name,
            };

            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };

            let candidate = SelectedDevice {
                physical_device,
                queue_family,
                capabilities,
                supports_portability_subset,
                vendor: gpu_vendor_from_pci_id(properties.vendor_id),
            };

            if best.as_ref().is_none_or(|(best_score, _)| score > *best_score) {
                best = Some((score, candidate));
            }
        }

        best.map(|(_, selected)| selected).ok_or(GpuError::NoCompatibleDevice)
    }

    /// This device's reported capabilities (precision support, workgroup/buffer limits).
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Human-readable Vulkan device name, for diagnostics/logging.
    pub fn device_name(&self) -> &str {
        &self.capabilities.device_name
    }

    /// Returns `Ok(())` if this device can execute `f64` compute shaders, or
    /// [`GpuError::Fp64Unsupported`] otherwise.
    ///
    /// GPU cores constructed with an `f64` element type must call this (rather than
    /// silently falling back to `f32`) before doing any GPU work.
    pub fn require_f64(&self) -> Result<(), GpuError> {
        if self.capabilities.shader_float64 {
            Ok(())
        } else {
            Err(GpuError::Fp64Unsupported)
        }
    }

    /// The `vkfft-rs` device profile derived from this context's real device, used to plan
    /// FFT IRs (`RealFftIr::build`) and lower them to shaders.
    pub(crate) fn device_profile(&self) -> DeviceProfile {
        self.device_profile
    }

    /// The raw `ash::Device`, shared (via `Arc`) with every `VulkanComputePipeline` built on
    /// this context so pipelines can outlive a single call while the context stays alive.
    pub(crate) fn device(&self) -> &Arc<ash::Device> {
        &self.device
    }

    /// The physical device selected for this context, for memory/format queries.
    pub(crate) fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// This device's memory heaps/types, for choosing a memory type index when allocating
    /// buffers.
    pub(crate) fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    /// The queue used for all compute submissions on this context.
    pub(crate) fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    /// The persistent command pool used to allocate command buffers for this context.
    pub(crate) fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// The persistent pipeline cache shared by all compute pipelines built on this context.
    pub(crate) fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    /// Records `record` into a fresh one-time-submit command buffer, submits it to this
    /// context's compute queue, and blocks until it completes.
    ///
    /// This is the building block every synchronous GPU operation in this module is built
    /// from: a single chunk's forward-FFT/remap/inverse-FFT/overlap-add pipeline is one call
    /// to this (recording every pass's dispatch plus the barriers between them via `record`),
    /// not one call per pass -- that single-submission property is what satisfies
    /// `gpu_plan.md`'s "never transfer the spectrum back to the CPU between forward and
    /// inverse FFT" rule.
    pub(crate) fn run_one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<(), GpuError> {
        let _guard = self
            .execution_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = self.device();

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `self.command_pool` belongs to this `device` and is not being reset/recorded
        // concurrently (serialized by `execution_lock`).
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }
            .map_err(|err| GpuError::ExecutionFailed(format!("failed to allocate Vulkan command buffer: {err}")))?;
        let command_buffer = command_buffers[0];

        let free_command_buffer = |device: &ash::Device| {
            // SAFETY: `command_buffer` was allocated from `self.command_pool` above and is not
            // in use by any pending submission by the time each call site below runs it.
            unsafe { device.free_command_buffers(self.command_pool, &command_buffers) };
        };

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `command_buffer` was just allocated and is in the initial state.
        if let Err(err) = unsafe { device.begin_command_buffer(command_buffer, &begin_info) } {
            free_command_buffer(device);
            return Err(GpuError::ExecutionFailed(format!("failed to begin Vulkan command buffer: {err}")));
        }

        record(command_buffer);

        // SAFETY: `command_buffer` is in the recording state from `begin_command_buffer` above.
        if let Err(err) = unsafe { device.end_command_buffer(command_buffer) } {
            free_command_buffer(device);
            return Err(GpuError::ExecutionFailed(format!("failed to end Vulkan command buffer: {err}")));
        }

        let fence_create_info = vk::FenceCreateInfo::default();
        // SAFETY: `device` is live.
        let fence = match unsafe { device.create_fence(&fence_create_info, None) } {
            Ok(fence) => fence,
            Err(err) => {
                free_command_buffer(device);
                return Err(GpuError::ExecutionFailed(format!("failed to create Vulkan fence: {err}")));
            }
        };

        let command_buffers_ref = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers_ref);
        // SAFETY: `command_buffer` finished recording above, `fence` was just created and is
        // unsignaled, and `self.compute_queue` accepts compute/transfer work; submissions to
        // it are serialized by `execution_lock`.
        let submit_result = unsafe { device.queue_submit(self.compute_queue, &[submit_info], fence) };
        if let Err(err) = submit_result {
            // SAFETY: `fence` was never submitted, so it is safe to destroy immediately.
            unsafe { device.destroy_fence(fence, None) };
            free_command_buffer(device);
            return Err(GpuError::ExecutionFailed(format!("failed to submit Vulkan command buffer: {err}")));
        }

        // SAFETY: `fence` was just submitted with the command buffer above.
        let wait_result = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) };
        // SAFETY: the fence wait above (successful or not) means the submission is no longer
        // in flight in any way that would race with destroying these objects.
        unsafe { device.destroy_fence(fence, None) };
        free_command_buffer(device);
        wait_result.map_err(|err| GpuError::ExecutionFailed(format!("failed waiting for Vulkan fence: {err}")))?;
        Ok(())
    }

    /// Records `record` into a fresh one-time-submit command buffer and submits it, *without*
    /// waiting for it to complete -- the asynchronous counterpart to [`GpuContext::run_one_shot`],
    /// for callers (currently [`super::batch_core::GpuBatchCore`]) that want to keep preparing
    /// the *next* piece of work (reading from disk, building the next FFT window, uploading it)
    /// while this GPU submission is still executing, rather than blocking on it immediately.
    ///
    /// Returns a [`GpuSubmission`] the caller must eventually resolve via
    /// [`GpuContext::is_submission_ready`]/[`GpuContext::wait_submission`] and then
    /// [`GpuContext::destroy_submission`] -- dropping a [`GpuSubmission`] without destroying it
    /// leaks its command buffer and fence.
    pub(crate) fn submit_async(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<GpuSubmission, GpuError> {
        let _guard = self
            .execution_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = self.device();

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `self.command_pool` belongs to this `device` and is not being reset/recorded
        // concurrently (serialized by `execution_lock`).
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }
            .map_err(|err| GpuError::ExecutionFailed(format!("failed to allocate Vulkan command buffer: {err}")))?;
        let command_buffer = command_buffers[0];

        let free_command_buffer = || {
            // SAFETY: `command_buffer` was allocated from `self.command_pool` above and is not
            // in use by any pending submission by the time each call site below runs it.
            unsafe { device.free_command_buffers(self.command_pool, &command_buffers) };
        };

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `command_buffer` was just allocated and is in the initial state.
        if let Err(err) = unsafe { device.begin_command_buffer(command_buffer, &begin_info) } {
            free_command_buffer();
            return Err(GpuError::ExecutionFailed(format!("failed to begin Vulkan command buffer: {err}")));
        }

        record(command_buffer);

        // SAFETY: `command_buffer` is in the recording state from `begin_command_buffer` above.
        if let Err(err) = unsafe { device.end_command_buffer(command_buffer) } {
            free_command_buffer();
            return Err(GpuError::ExecutionFailed(format!("failed to end Vulkan command buffer: {err}")));
        }

        let fence_create_info = vk::FenceCreateInfo::default();
        // SAFETY: `device` is live.
        let fence = match unsafe { device.create_fence(&fence_create_info, None) } {
            Ok(fence) => fence,
            Err(err) => {
                free_command_buffer();
                return Err(GpuError::ExecutionFailed(format!("failed to create Vulkan fence: {err}")));
            }
        };

        let command_buffers_ref = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers_ref);
        // SAFETY: `command_buffer` finished recording above, `fence` was just created and is
        // unsignaled, and `self.compute_queue` accepts compute/transfer work; submissions to it
        // are serialized by `execution_lock`. Unlike `run_one_shot`, this does not wait for the
        // fence before returning -- that is the entire point of this method.
        if let Err(err) = unsafe { device.queue_submit(self.compute_queue, &[submit_info], fence) } {
            // SAFETY: `fence` was never submitted, so it is safe to destroy immediately.
            unsafe { device.destroy_fence(fence, None) };
            free_command_buffer();
            return Err(GpuError::ExecutionFailed(format!("failed to submit Vulkan command buffer: {err}")));
        }

        Ok(GpuSubmission { command_buffer, fence })
    }

    /// Non-blocking check for whether `submission` has finished executing on the GPU.
    pub(crate) fn is_submission_ready(&self, submission: &GpuSubmission) -> Result<bool, GpuError> {
        // SAFETY: `submission.fence` was created and submitted by `submit_async` and has not
        // been destroyed yet (caller contract).
        unsafe { self.device.get_fence_status(submission.fence) }
            .map_err(|err| GpuError::ExecutionFailed(format!("failed to query Vulkan fence status: {err}")))
    }

    /// Blocks until `submission` finishes executing on the GPU.
    pub(crate) fn wait_submission(&self, submission: &GpuSubmission) -> Result<(), GpuError> {
        // SAFETY: `submission.fence` was created and submitted by `submit_async` and has not
        // been destroyed yet (caller contract). Waiting on a fence does not touch the command
        // pool or queue, so this needs no lock.
        unsafe { self.device.wait_for_fences(&[submission.fence], true, u64::MAX) }
            .map_err(|err| GpuError::ExecutionFailed(format!("failed waiting for Vulkan fence: {err}")))
    }

    /// Frees `submission`'s command buffer and destroys its fence. The caller must have already
    /// confirmed completion (via [`GpuContext::wait_submission`] or a `true` result from
    /// [`GpuContext::is_submission_ready`]) -- destroying a fence/command buffer with work still
    /// in flight against them is undefined behavior.
    pub(crate) fn destroy_submission(&self, submission: GpuSubmission) {
        let _guard = self
            .execution_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: caller contract guarantees `submission`'s GPU work has completed, so neither
        // object is in use by any pending or executing submission.
        unsafe {
            self.device.destroy_fence(submission.fence, None);
            self.device.free_command_buffers(self.command_pool, &[submission.command_buffer]);
        }
    }

    /// Copies `byte_len` bytes from `src` to `dst` via one immediate command buffer.
    pub(crate) fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, byte_len: vk::DeviceSize) -> Result<(), GpuError> {
        self.run_one_shot(|command_buffer| {
            let region = vk::BufferCopy::default().size(byte_len);
            // SAFETY: `command_buffer` is in the recording state (this closure only ever runs
            // inside `run_one_shot`); `src`/`dst` are valid buffers at least `byte_len` bytes,
            // per every caller of `copy_buffer`.
            unsafe { self.device().cmd_copy_buffer(command_buffer, src, dst, &[region]) };
        })
    }
}

/// A GPU submission recorded and submitted by [`GpuContext::submit_async`], not yet waited on.
///
/// Must eventually be resolved via [`GpuContext::wait_submission`] or a `true` result from
/// [`GpuContext::is_submission_ready`], then released via [`GpuContext::destroy_submission`].
pub(crate) struct GpuSubmission {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

struct SelectedDevice {
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    capabilities: GpuCapabilities,
    supports_portability_subset: bool,
    vendor: GpuVendor,
}

/// Maps a Vulkan `vendorID` (the PCI-SIG registered vendor ID VkPhysicalDeviceProperties
/// reports) to `vkfft-rs`'s `GpuVendor`, so its planner can pick vendor-tuned scheduling where
/// it has one, and fail soft to its portable path otherwise.
fn gpu_vendor_from_pci_id(vendor_id: u32) -> GpuVendor {
    match vendor_id {
        0x10DE => GpuVendor::Nvidia,
        0x1002 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        0x106B => GpuVendor::Apple,
        other => GpuVendor::Other(other),
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // SAFETY: these are destroyed in strict reverse-creation order. `self.device` is an
        // `Arc`, shared with every `VulkanComputePipeline`/`GpuBuffer` built on this context,
        // so destroying it here is only sound because every such consumer holds an
        // `Arc<GpuContext>` alongside its device clone and, within its own struct, declares
        // that device-touching field *before* the `Arc<GpuContext>` field -- Rust drops
        // struct fields in declaration order, so those consumers' Vulkan objects (and their
        // device-Arc clones) are always dropped before the last `Arc<GpuContext>` (and hence
        // this `Drop`) can run. No GPU work is in flight for the same reason: nothing reaches
        // this point while a core still holds work submitted against this device.
        unsafe {
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

// SAFETY: `GpuContext` does not expose interior mutability; all Vulkan handles are safe to
// share across threads per the Vulkan spec as long as external synchronization is applied
// to any single handle's use, which is the caller's responsibility (matching `ash`'s own
// `Send`/`Sync` handle types).
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

#[cfg(test)]
mod tests {
    use super::*;

    /// On macOS, `ash::Entry::load()` only searches the default `dlopen` fallback paths,
    /// which do not include Homebrew's `/opt/homebrew/lib` (where `brew install
    /// vulkan-loader` puts `libvulkan.dylib`). Run this test with
    /// `DYLD_LIBRARY_PATH=/opt/homebrew/lib` to exercise real device creation instead of
    /// just the graceful `VulkanInitFailed` skip path.
    #[test]
    fn reports_capabilities_or_skips_without_vulkan() {
        let context = match GpuContext::new() {
            Ok(context) => context,
            Err(err) => {
                eprintln!("skipping GPU context test: {err}");
                return;
            }
        };

        assert!(!context.device_name().is_empty());
        assert!(context.capabilities().max_compute_workgroup_count.iter().all(|&n| n > 0));

        match context.require_f64() {
            Ok(()) => {
                eprintln!("device {} reports shaderFloat64 support", context.device_name());
            }
            Err(GpuError::Fp64Unsupported) => {
                eprintln!(
                    "device {} does not support shaderFloat64 (expected on Apple/MoltenVK)",
                    context.device_name()
                );
            }
            Err(other) => panic!("unexpected error from require_f64: {other}"),
        }
    }
}
