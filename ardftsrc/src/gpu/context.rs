use std::ffi::CString;
use std::sync::{Arc, Mutex, RwLock};

use ash::vk;
use num_traits::Float;
use vkfft_rs::backend::vulkan::VulkanSpirvShader;
use vkfft_rs::backend::vulkan::runtime::VulkanComputePipeline;
use vkfft_rs::{Backend, DeviceProfile, GpuVendor};

use super::buffer::{GpuScalar, GpuStorageBufferSlice};
use super::error::GpuError;
use super::fft_program::FromF64;
use super::shaders::GpuShaders;
use crate::Config;
use crate::config::DerivedConfig;

/// Stable Vulkan identity used to select a physical GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuDeviceId {
    /// Vulkan device UUID reported by `VkPhysicalDeviceIDProperties`.
    pub device_uuid: [u8; vk::UUID_SIZE],
}

/// Vulkan identifiers used to validate opaque pipeline-cache bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPipelineCacheId {
    /// PCI vendor identifier reported by Vulkan.
    pub vendor_id: u32,
    /// PCI device identifier reported by Vulkan.
    pub device_id: u32,
    /// UUID defining compatibility for opaque Vulkan pipeline-cache bytes.
    pub pipeline_cache_uuid: [u8; vk::UUID_SIZE],
}

/// Broad Vulkan physical-device classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuDeviceType {
    Other,
    Integrated,
    Discrete,
    Virtual,
    Cpu,
}

/// Identity, capabilities, and limits of the Vulkan device backing a [`GpuDevice`].
///
/// GPU cores must consult this before executing `f64` work rather than assuming support:
/// notably, Apple GPUs (Metal via MoltenVK) never report `shader_float64` support, since
/// Metal has no double-precision shader type at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// Physical-device identity used for explicit selection.
    pub id: GpuDeviceId,
    /// Identity tuple governing compatibility of opaque pipeline-cache data.
    pub pipeline_cache_id: GpuPipelineCacheId,
    /// Physical-device classification.
    pub device_type: GpuDeviceType,
    /// Vulkan API version reported by the device.
    pub api_version: u32,
    /// Vendor-defined Vulkan driver version.
    pub driver_version: u32,
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
    /// Human-readable device name, for UI.
    pub device_name: String,
}

/// Owns the long-lived Vulkan objects shared by all GPU cores.
///
/// This is infrastructure only: it does not know about ARDFTSRC's DSP geometry, FFT plans,
/// or shaders. GPU cores are constructed with an `Arc<GpuDevice>` and build their own
/// FFT plans/buffers/pipelines on top of it, using `vkfft-rs`'s public planner/shader-lowering
/// API (`ProgramIr`, `VulkanGlslBackend::lower_real_fft`, `VulkanComputePipeline`) against
/// buffers and a command stream that this crate owns -- not `vkfft-rs`'s own hidden
/// `VulkanExecutionContext`. This is what lets a GPU core record its own custom
/// spectral-remap/overlap-add shaders in between the forward and inverse FFT passes within a
/// single command buffer submission, with no host round-trip in between.
pub struct GpuDevice {
    entry: ash::Entry,
    instance: ash::Instance,
    device: Arc<ash::Device>,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_lock: Mutex<()>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    storage_buffer_offset_alignment: vk::DeviceSize,
    device_profile: DeviceProfile,
    info: GpuInfo,
    /// Serializes command-pool recording/release and queue submission: `vk::CommandPool` is not
    /// safe to record from concurrently, and `vkQueueSubmit` on the same queue needs external
    /// synchronization too. `vkfft-rs`'s own `VulkanExecutionContext` guards its queue/pool the
    /// same way, for the same reason.
    execution_lock: Mutex<()>,
}

/// A compute pipeline branded with the logical device that created it.
///
/// Keeping the device identity alongside the Vulkan object lets recording reject accidental
/// cross-device use before it reaches `ash`.
pub(crate) struct GpuComputePipeline {
    pipeline: VulkanComputePipeline,
    device: Arc<GpuDevice>,
    expected_scalar: vkfft_rs::ScalarType,
    expected_bindings: Vec<u32>,
    _bindings: Vec<GpuStorageBufferSlice>,
}

impl GpuComputePipeline {
    /// Binds validated storage-buffer slices and retains their allocations for this pipeline's
    /// full lifetime.
    pub(crate) fn bind_storage_buffers(&mut self, buffers: Vec<(u32, GpuStorageBufferSlice)>) -> Result<(), GpuError> {
        let actual_bindings = buffers.iter().map(|(binding, _)| *binding).collect::<Vec<_>>();
        if actual_bindings != self.expected_bindings {
            return Err(GpuError::PlanCreationFailed(format!(
                "storage-buffer bindings {actual_bindings:?} do not match shader contract {:?}",
                self.expected_bindings
            )));
        }
        for (_, buffer) in &buffers {
            if !Arc::ptr_eq(&self.device, buffer.device()) {
                return Err(GpuError::PlanCreationFailed(
                    "cannot bind a storage buffer from a different Vulkan device".to_string(),
                ));
            }
            if buffer.scalar() != self.expected_scalar {
                return Err(GpuError::PlanCreationFailed(format!(
                    "storage-buffer scalar {:?} does not match shader scalar {:?}",
                    buffer.scalar(),
                    self.expected_scalar
                )));
            }
        }

        let raw = buffers
            .iter()
            .map(|(binding, buffer)| (*binding, buffer.raw()))
            .collect::<Vec<_>>();
        // SAFETY: each typed slice validated its bounds/alignment during construction, all
        // allocations belong to this pipeline's device, and `_bindings` retains them after
        // the descriptor update. Binding IDs and scalar types match the shader metadata.
        unsafe { self.pipeline.update_storage_buffers(&raw) }
            .map_err(|err| GpuError::PlanCreationFailed(format!("failed to update storage buffers: {err}")))?;
        self._bindings = buffers.into_iter().map(|(_, buffer)| buffer).collect();
        Ok(())
    }

    fn record(&self, command: &RecordingCommandBuffer<'_>) {
        assert!(
            Arc::ptr_eq(&self.device, command.device),
            "attempted to record a compute pipeline on a different Vulkan device"
        );
        // SAFETY: `command` can only be constructed while its matching command buffer is
        // recording, and the device brand was checked above.
        unsafe { self.pipeline.record_dispatch(command.handle) };
    }
}

/// A command buffer known to be recording on one particular [`GpuDevice`].
///
/// Values exist only inside the closure passed to [`GpuDevice::submit_async`], so raw command
/// handles and recording-state obligations do not escape into the rest of the GPU module.
pub(crate) struct RecordingCommandBuffer<'a> {
    device: &'a Arc<GpuDevice>,
    handle: vk::CommandBuffer,
}

impl RecordingCommandBuffer<'_> {
    pub(crate) fn dispatch(&self, pipeline: &GpuComputePipeline) {
        pipeline.record(self);
    }

    pub(crate) fn compute_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: this wrapper exists only while `handle` is recording on `device`.
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                self.handle,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
    }

    pub(crate) fn transfer_to_compute_barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: this wrapper exists only while `handle` is recording on `device`.
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                self.handle,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
    }

    /// # Safety
    ///
    /// `src` and `dst` must be live buffers owned by this command buffer's device, have the
    /// required transfer usage flags, and each cover at least `byte_len` bytes.
    pub(crate) unsafe fn copy_raw_buffer(&self, src: vk::Buffer, dst: vk::Buffer, byte_len: vk::DeviceSize) {
        let region = vk::BufferCopy::default().size(byte_len);
        // SAFETY: the caller supplies the raw-buffer provenance, usage, and bounds guarantees.
        unsafe { self.device.device.cmd_copy_buffer(self.handle, src, dst, &[region]) };
    }
}

impl GpuDevice {
    /// Lists every Vulkan 1.2+ physical device that exposes a compute queue.
    pub fn list() -> Result<Vec<GpuInfo>, GpuError> {
        let (_entry, instance) = Self::create_instance()?;
        let result =
            Self::compatible_devices(&instance).map(|devices| devices.into_iter().map(|device| device.info).collect());
        // SAFETY: enumeration does not create child objects from this instance.
        unsafe { instance.destroy_instance(None) };
        result
    }

    /// Creates a logical device for the physical GPU identified by `id`.
    pub fn new(id: GpuDeviceId) -> Result<Self, GpuError> {
        Self::create(Some(id))
    }

    /// Creates a logical device on the preferred available compute-capable GPU.
    pub fn auto_select() -> Result<Self, GpuError> {
        Self::create(None)
    }

    fn create(requested: Option<GpuDeviceId>) -> Result<Self, GpuError> {
        let (entry, instance) = Self::create_instance()?;
        let selected = match Self::select_physical_device(&instance, requested) {
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

    fn create_instance() -> Result<(ash::Entry, ash::Instance), GpuError> {
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
        Ok((entry, instance))
    }

    fn finish_new(
        entry: ash::Entry,
        instance: ash::Instance,
        selected: SelectedDevice,
    ) -> Result<Self, Box<(ash::Instance, GpuError)>> {
        let SelectedDevice {
            physical_device,
            queue_family,
            info,
            storage_buffer_offset_alignment,
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

        let enabled_features = vk::PhysicalDeviceFeatures::default().shader_float64(info.shader_float64);

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
        // hardware truth (see `GpuInfo::shader_float64`/`GpuDevice::require_f64` for
        // that); it only gates which precision `RealFftIr::build` is willing to plan for, so it
        // must mirror what we actually queried, not the backend-wide default.
        let device_profile = DeviceProfile {
            supports_f64: info.shader_float64,
            ..DeviceProfile::generic(Backend::Vulkan, vendor)
        };

        let device = Arc::new(device);

        Ok(Self {
            entry,
            instance,
            device,
            compute_queue,
            command_pool,
            pipeline_cache,
            pipeline_cache_lock: Mutex::new(()),
            memory_properties,
            storage_buffer_offset_alignment,
            device_profile,
            info,
            execution_lock: Mutex::new(()),
        })
    }

    fn select_physical_device(
        instance: &ash::Instance,
        requested: Option<GpuDeviceId>,
    ) -> Result<SelectedDevice, GpuError> {
        let devices = Self::compatible_devices(instance)?;
        match requested {
            Some(id) => devices
                .into_iter()
                .find(|device| device.info.id == id)
                .ok_or(GpuError::DeviceNotFound(id)),
            None => devices
                .into_iter()
                .max_by_key(|device| device_score(device.info.device_type))
                .ok_or(GpuError::NoCompatibleDevice),
        }
    }

    fn compatible_devices(instance: &ash::Instance) -> Result<Vec<SelectedDevice>, GpuError> {
        // SAFETY: `instance` is valid and was just created successfully by the caller.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|err| GpuError::VulkanInitFailed(format!("failed to enumerate Vulkan devices: {err}")))?;

        let mut devices = Vec::new();
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

            let mut id_properties = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_properties);
            // SAFETY: `physical_device` is valid, from this instance, and the output chain is
            // initialized for Vulkan to populate.
            unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
            let properties = properties2.properties;
            // SAFETY: `physical_device` is valid, from the same instance.
            let features = unsafe { instance.get_physical_device_features(physical_device) };
            let extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap_or_default();
            let supports_portability_subset = extensions
                .iter()
                .any(|ext| ext.extension_name_as_c_str().ok() == Some(ash::khr::portability_subset::NAME));

            let device_name = properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unknown Vulkan device>".to_string());

            let info = GpuInfo {
                id: GpuDeviceId {
                    device_uuid: id_properties.device_uuid,
                },
                pipeline_cache_id: GpuPipelineCacheId {
                    vendor_id: properties.vendor_id,
                    device_id: properties.device_id,
                    pipeline_cache_uuid: properties.pipeline_cache_uuid,
                },
                device_type: match properties.device_type {
                    vk::PhysicalDeviceType::INTEGRATED_GPU => GpuDeviceType::Integrated,
                    vk::PhysicalDeviceType::DISCRETE_GPU => GpuDeviceType::Discrete,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => GpuDeviceType::Virtual,
                    vk::PhysicalDeviceType::CPU => GpuDeviceType::Cpu,
                    _ => GpuDeviceType::Other,
                },
                api_version: properties.api_version,
                driver_version: properties.driver_version,
                shader_float64: features.shader_float64 == vk::TRUE,
                max_storage_buffer_range: properties.limits.max_storage_buffer_range as u64,
                max_compute_workgroup_count: properties.limits.max_compute_work_group_count,
                max_compute_workgroup_size: properties.limits.max_compute_work_group_size,
                max_compute_work_group_invocations: properties.limits.max_compute_work_group_invocations,
                device_name,
            };

            devices.push(SelectedDevice {
                physical_device,
                queue_family,
                info,
                storage_buffer_offset_alignment: properties.limits.min_storage_buffer_offset_alignment.max(1),
                supports_portability_subset,
                vendor: gpu_vendor_from_pci_id(properties.vendor_id),
            });
        }

        Ok(devices)
    }

    /// This device's reported capabilities (precision support, workgroup/buffer limits).
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Human-readable Vulkan device name, for diagnostics/logging.
    pub fn device_name(&self) -> &str {
        &self.info.device_name
    }

    /// Returns `Ok(())` if this device can execute `f64` compute shaders, or
    /// [`GpuError::Fp64Unsupported`] otherwise.
    ///
    /// GPU cores constructed with an `f64` element type must call this (rather than
    /// silently falling back to `f32`) before doing any GPU work.
    pub fn require_f64(&self) -> Result<(), GpuError> {
        if self.info.shader_float64 {
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

    /// This device's memory heaps/types, for choosing a memory type index when allocating
    /// buffers.
    pub(crate) fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    pub(crate) fn storage_buffer_offset_alignment(&self) -> vk::DeviceSize {
        self.storage_buffer_offset_alignment
    }

    /// Builds one live compute pipeline while externally synchronizing the shared Vulkan cache.
    pub(crate) fn create_compute_pipeline(
        self: &Arc<Self>,
        shader: &VulkanSpirvShader,
    ) -> Result<GpuComputePipeline, GpuError> {
        if shader.descriptors.iter().any(|descriptor| descriptor.set != 0) {
            return Err(GpuError::PlanCreationFailed(
                "only descriptor set 0 is supported by the typed pipeline wrapper".to_string(),
            ));
        }
        let _guard = self
            .pipeline_cache_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the device and pipeline cache are owned by `self` and remain live for the
        // returned pipeline's construction. The cache lock provides Vulkan's required external
        // synchronization.
        let pipeline = unsafe {
            VulkanComputePipeline::new_with_pipeline_cache(Arc::clone(&self.device), shader, self.pipeline_cache)
        }
        .map_err(|err| GpuError::PlanCreationFailed(format!("failed to build Vulkan compute pipeline: {err}")))?;
        Ok(GpuComputePipeline {
            pipeline,
            device: Arc::clone(self),
            expected_scalar: shader.scalar,
            expected_bindings: shader.descriptors.iter().map(|descriptor| descriptor.binding).collect(),
            _bindings: Vec::new(),
        })
    }

    pub(crate) fn pipeline_cache_data(&self) -> Result<Vec<u8>, GpuError> {
        let _guard = self
            .pipeline_cache_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the cache belongs to this live device and is externally synchronized.
        unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache) }
            .map_err(|err| GpuError::PipelineCacheFailed(err.to_string()))
    }

    pub(crate) fn merge_pipeline_cache_data(&self, data: &[u8]) -> Result<(), GpuError> {
        if data.is_empty() {
            return Ok(());
        }
        let _guard = self
            .pipeline_cache_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let create_info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        // SAFETY: `data` remains live during creation and the returned cache belongs to this
        // device. It is destroyed before releasing the external synchronization lock.
        let source = unsafe { self.device.create_pipeline_cache(&create_info, None) }
            .map_err(|err| GpuError::PipelineCacheFailed(err.to_string()))?;
        let merge_result = unsafe { self.device.merge_pipeline_caches(self.pipeline_cache, &[source]) };
        unsafe { self.device.destroy_pipeline_cache(source, None) };
        merge_result.map_err(|err| GpuError::PipelineCacheFailed(err.to_string()))
    }

    /// Moves `resources` into a fresh submission and lends both it and a device-branded
    /// recording command buffer to `record`.
    ///
    /// The returned [`GpuSubmission`] owns everything the commands may reference. It releases
    /// those resources only after fence completion, including when dropped during unwinding.
    pub(crate) fn submit_async<R: 'static>(
        self: &Arc<Self>,
        resources: R,
        record: for<'a> fn(&RecordingCommandBuffer<'a>, &R),
    ) -> Result<GpuSubmission<R>, (GpuError, R)> {
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
        let command_buffers = match unsafe { device.allocate_command_buffers(&allocate_info) } {
            Ok(command_buffers) => command_buffers,
            Err(err) => {
                return Err((
                    GpuError::ExecutionFailed(format!("failed to allocate Vulkan command buffer: {err}")),
                    resources,
                ));
            }
        };
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
            return Err((
                GpuError::ExecutionFailed(format!("failed to begin Vulkan command buffer: {err}")),
                resources,
            ));
        }

        let recording = RecordingCommandBuffer {
            device: self,
            handle: command_buffer,
        };
        record(&recording, &resources);

        // SAFETY: `command_buffer` is in the recording state from `begin_command_buffer` above.
        if let Err(err) = unsafe { device.end_command_buffer(command_buffer) } {
            free_command_buffer();
            return Err((
                GpuError::ExecutionFailed(format!("failed to end Vulkan command buffer: {err}")),
                resources,
            ));
        }

        let fence_create_info = vk::FenceCreateInfo::default();
        // SAFETY: `device` is live.
        let fence = match unsafe { device.create_fence(&fence_create_info, None) } {
            Ok(fence) => fence,
            Err(err) => {
                free_command_buffer();
                return Err((
                    GpuError::ExecutionFailed(format!("failed to create Vulkan fence: {err}")),
                    resources,
                ));
            }
        };

        let command_buffers_ref = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers_ref);
        // SAFETY: `command_buffer` finished recording above, `fence` was just created and is
        // unsignaled, and `self.compute_queue` accepts compute/transfer work; submissions to it
        // are serialized by `execution_lock`. Completion is managed by the returned owning
        // submission.
        if let Err(err) = unsafe { device.queue_submit(self.compute_queue, &[submit_info], fence) } {
            // SAFETY: `fence` was never submitted, so it is safe to destroy immediately.
            unsafe { device.destroy_fence(fence, None) };
            free_command_buffer();
            return Err((
                GpuError::ExecutionFailed(format!("failed to submit Vulkan command buffer: {err}")),
                resources,
            ));
        }

        Ok(GpuSubmission {
            device: Some(Arc::clone(self)),
            command_buffer,
            fence,
            resources: Some(resources),
            completed: false,
        })
    }
}

/// One validated, precision-specific GPU resampling geometry and its compiled shaders.
///
/// The underlying Vulkan device may be shared by contexts with different configurations.
/// `group_chunks` belongs here because it changes FFT batch planning and generated shaders;
/// runtime ring depth does not.
pub struct GpuContext<T> {
    device: Arc<GpuDevice>,
    config: Config,
    derived: DerivedConfig<T>,
    group_chunks: usize,
    shaders: RwLock<Option<Arc<GpuShaders>>>,
}

impl<T> GpuContext<T> {
    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }
}

#[allow(private_bounds)]
impl<T> GpuContext<T>
where
    T: Float + GpuScalar + FromF64,
{
    /// Creates a context on the best available Vulkan device.
    pub fn new(config: Config, group_chunks: usize) -> Result<Self, GpuError> {
        Self::with_device(Arc::new(GpuDevice::auto_select()?), config, group_chunks)
    }

    /// Creates a context for an explicitly selected Vulkan device.
    pub fn with_device(device: Arc<GpuDevice>, config: Config, group_chunks: usize) -> Result<Self, GpuError> {
        if group_chunks == 0 {
            return Err(GpuError::InvalidConfig(
                "group_chunks must be greater than zero".to_string(),
            ));
        }
        if T::scalar_type() == vkfft_rs::ScalarType::F64 {
            device.require_f64()?;
        }
        let derived = config
            .derive_config::<T>()
            .map_err(|err| GpuError::InvalidConfig(err.to_string()))?;
        if derived.decimation_stages > 0 {
            return Err(GpuError::DecimationUnsupported);
        }
        if derived.f128 {
            return Err(GpuError::F128UnsupportedOnGpu);
        }
        Ok(Self {
            device,
            config,
            derived,
            group_chunks,
            shaders: RwLock::new(None),
        })
    }

    /// Loads a precompiled shader artifact after validating its device and exact geometry.
    ///
    /// An incompatible artifact is rejected without replacing the currently cached shaders.
    ///
    /// # Safety
    ///
    /// `shaders` must contain trusted SPIR-V and pipeline-cache data produced by this crate for
    /// the reported device. The validation performed here checks archive metadata and geometry,
    /// but it does not prove that the executable SPIR-V obeys its descriptor and buffer-bounds
    /// contracts, nor can it authenticate the opaque pipeline-cache bytes. Loading a forged or
    /// otherwise untrusted artifact can violate Vulkan's safety requirements when pipelines are
    /// created or dispatched.
    pub unsafe fn load_shaders(&self, shaders: GpuShaders) -> Result<(), GpuError> {
        shaders.validate(&self.device, &self.derived, self.config.channels, self.group_chunks)?;
        self.device.merge_pipeline_cache_data(shaders.pipeline_cache())?;
        *self.shaders.write().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(shaders));
        Ok(())
    }

    /// Returns compiled shaders, generating and caching them when necessary.
    ///
    /// The returned owned snapshot includes the device's latest Vulkan pipeline-cache bytes and
    /// can be persisted with [`GpuShaders::to_bytes`].
    pub fn shaders(&self) -> Result<GpuShaders, GpuError> {
        let shaders = self.compiled_shaders()?;
        Ok(shaders
            .as_ref()
            .clone()
            .with_pipeline_cache(self.device.pipeline_cache_data()?))
    }

    pub(crate) fn compiled_shaders(&self) -> Result<Arc<GpuShaders>, GpuError> {
        if let Some(shaders) = self
            .shaders
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .cloned()
            && shaders
                .validate(&self.device, &self.derived, self.config.channels, self.group_chunks)
                .is_ok()
        {
            return Ok(shaders);
        }

        let generated = Arc::new(GpuShaders::compile(
            &self.device,
            &self.derived,
            self.config.channels,
            self.group_chunks,
        )?);
        let mut cached = self.shaders.write().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = cached.as_ref()
            && existing
                .validate(&self.device, &self.derived, self.config.channels, self.group_chunks)
                .is_ok()
        {
            return Ok(Arc::clone(existing));
        }
        *cached = Some(Arc::clone(&generated));
        Ok(generated)
    }

    /// Identity, capabilities, and limits of this context's target GPU.
    pub fn info(&self) -> &GpuInfo {
        self.device.info()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn group_chunks(&self) -> usize {
        self.group_chunks
    }

    pub(crate) fn derived(&self) -> &DerivedConfig<T> {
        &self.derived
    }
}

/// One pending GPU submission together with every resource its commands may access.
///
/// Resolving or dropping this value waits for completion before releasing either the Vulkan
/// synchronization objects or `resources`. If completion cannot be established, all of them
/// are intentionally leaked so safe unwinding cannot destroy GPU-referenced memory.
pub(crate) struct GpuSubmission<R> {
    device: Option<Arc<GpuDevice>>,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    resources: Option<R>,
    completed: bool,
}

impl<R> GpuSubmission<R> {
    pub(crate) fn is_ready(&self) -> Result<bool, GpuError> {
        let device = self.device.as_ref().expect("pending submission retains its device");
        // SAFETY: this fence was created and submitted by `GpuDevice::submit_async` and remains
        // owned by `self`.
        unsafe { device.device.get_fence_status(self.fence) }
            .map_err(|err| GpuError::ExecutionFailed(format!("failed to query Vulkan fence status: {err}")))
    }

    fn wait(&self) -> Result<(), GpuError> {
        let device = self.device.as_ref().expect("pending submission retains its device");
        // SAFETY: this fence was created and submitted by `GpuDevice::submit_async` and remains
        // owned by `self`.
        unsafe { device.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|err| GpuError::ExecutionFailed(format!("failed waiting for Vulkan fence: {err}")))
    }

    fn release_completed(&mut self) {
        let device = Arc::clone(self.device.as_ref().expect("pending submission retains its device"));
        let _guard = device
            .execution_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the fence has reported completion, so neither object is pending.
        unsafe {
            device.device.destroy_fence(self.fence, None);
            device
                .device
                .free_command_buffers(device.command_pool, &[self.command_buffer]);
        }
        self.completed = true;
    }

    fn abandon(&mut self) {
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
        if let Some(device) = self.device.take() {
            std::mem::forget(device);
        }
        self.completed = true;
    }

    pub(crate) fn resolve(mut self) -> Result<R, GpuError> {
        if let Err(err) = self.wait() {
            self.abandon();
            return Err(err);
        }
        self.release_completed();
        Ok(self.resources.take().expect("pending submission retains its resources"))
    }
}

impl<R> Drop for GpuSubmission<R> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if self.wait().is_ok() {
            self.release_completed();
            return;
        }
        self.abandon();
        // The raw command-buffer and fence handles have no Rust destructors. They deliberately
        // remain allocated on the leaked device because completion is unknown.
    }
}

struct SelectedDevice {
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    info: GpuInfo,
    storage_buffer_offset_alignment: vk::DeviceSize,
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

const fn device_score(device_type: GpuDeviceType) -> u8 {
    match device_type {
        GpuDeviceType::Discrete => 4,
        GpuDeviceType::Integrated => 3,
        GpuDeviceType::Virtual => 2,
        GpuDeviceType::Cpu => 1,
        GpuDeviceType::Other => 0,
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // SAFETY: every buffer allocation, bound pipeline, and pending submission owns an
        // `Arc<GpuDevice>`, so the last Arc cannot reach this destructor while a child object or
        // command is still live. Device-level objects are destroyed in reverse creation order.
        unsafe {
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

// SAFETY: `GpuDevice` does not expose interior mutability; all Vulkan handles are safe to
// share across threads per the Vulkan spec as long as external synchronization is applied
// to any single handle's use, which is the caller's responsibility (matching `ash`'s own
// `Send`/`Sync` handle types).
unsafe impl Send for GpuDevice {}
unsafe impl Sync for GpuDevice {}

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
        let devices = match GpuDevice::list() {
            Ok(devices) => devices,
            Err(err) => {
                eprintln!("skipping GPU context test: {err}");
                return;
            }
        };
        let Some(info) = devices.first() else {
            eprintln!("skipping GPU context test: no compatible Vulkan device found");
            return;
        };
        let context = GpuDevice::new(info.id).expect("listed GPU should remain selectable");

        assert!(!context.device_name().is_empty());
        assert_eq!(context.info().id, info.id);
        assert!(context.info().max_compute_workgroup_count.iter().all(|&n| n > 0));

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

    #[test]
    fn shader_cache_round_trips_and_rejects_wrong_geometry() {
        let device = match GpuDevice::auto_select() {
            Ok(device) => Arc::new(device),
            Err(err) => {
                eprintln!("skipping GPU shader cache test: {err}");
                return;
            }
        };
        let config = Config::new(44_100, 48_000, 1);
        let source = GpuContext::<f32>::with_device(Arc::clone(&device), config.clone(), 2).expect("source context");
        let encoded = source.shaders().expect("compile shaders").to_bytes();
        // SAFETY: `encoded` was just produced by this crate from shaders compiled for `device`
        // and has not been modified.
        let decoded = unsafe { GpuShaders::from_bytes(&encoded) }.expect("decode shaders");

        let matching =
            GpuContext::<f32>::with_device(Arc::clone(&device), config.clone(), 2).expect("matching context");
        // SAFETY: `decoded` came from the trusted artifact produced immediately above.
        unsafe { matching.load_shaders(decoded.clone()) }.expect("load matching shaders");
        assert_eq!(matching.shaders().unwrap().group_chunks(), 2);

        let mismatched = GpuContext::<f32>::with_device(device, config, 3).expect("mismatched context");
        assert!(matches!(
            // SAFETY: `decoded` is trusted; this test expects the safe metadata validation to
            // reject its incompatible geometry before use.
            unsafe { mismatched.load_shaders(decoded) },
            Err(GpuError::IncompatibleShaders(_))
        ));
    }
}
