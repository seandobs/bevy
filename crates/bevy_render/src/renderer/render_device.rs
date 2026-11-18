use super::RenderQueue;
use crate::render_resource::{
    BindGroup, BindGroupLayout, BindGroupLayoutId, Buffer, ComputePipeline,
    RawRenderPipelineDescriptor, RenderPipeline, Sampler, Texture,
};
use crate::renderer::WgpuWrapper;
use bevy_ecs::{resource::Resource, system::Res};
use bevy_platform::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BufferAsyncError, BufferBindingType, PollError, PollStatus,
};

/// This GPU device is responsible for the creation of most rendering and compute resources.
#[derive(Resource, Clone)]
pub struct RenderDevice {
    device: WgpuWrapper<wgpu::Device>,
    /// The device's features, read once: on the WebGPU backend each read
    /// rebuilds the set from the browser's own, one string per feature.
    features: wgpu::Features,
    /// The device's limits, read once for the same reason.
    limits: wgpu::Limits,
    /// The bind groups made in the last few frames, by layout and resources.
    /// The engine makes many bind groups again every frame from the same
    /// inputs, and on the WebGPU backend each one is a call into the browser.
    bind_groups: Arc<WgpuWrapper<Mutex<HashMap<BindGroupKey, KeptBindGroup>>>>,
}

/// The frames a kept bind group stays unused before the cache drops it. The
/// engine swaps a current and a previous buffer each frame for some data, so
/// a frame binds the same buffers as the frame two before it.
const KEPT_FRAMES: u8 = 2;

/// A bind group that [`RenderDevice::create_bind_group`] keeps, and the
/// frames since it last returned it.
struct KeptBindGroup {
    group: BindGroup,
    idle: u8,
}

/// What a bind group binds: its layout, and the resource at each binding.
/// wgpu compares its handles by identity, and the key holds a clone of each,
/// so no resource it names is dropped and replaced by another at the same
/// address while the key lives.
#[derive(PartialEq, Eq, Hash)]
struct BindGroupKey {
    layout: BindGroupLayoutId,
    entries: Vec<(u32, BoundResource)>,
}

#[derive(PartialEq, Eq, Hash)]
enum BoundResource {
    Buffer(BoundBuffer),
    Buffers(Vec<BoundBuffer>),
    Sampler(wgpu::Sampler),
    Samplers(Vec<wgpu::Sampler>),
    TextureView(wgpu::TextureView),
    TextureViews(Vec<wgpu::TextureView>),
}

#[derive(PartialEq, Eq, Hash)]
struct BoundBuffer(wgpu::Buffer, wgpu::BufferAddress, Option<wgpu::BufferSize>);

impl BoundBuffer {
    fn of(binding: &wgpu::BufferBinding) -> Self {
        Self(binding.buffer.clone(), binding.offset, binding.size)
    }
}

impl BindGroupKey {
    /// The key of a bind group of `layout` binding `entries`, or `None` if
    /// an entry binds a resource the cache doesn't key: an acceleration
    /// structure or an external texture.
    fn of(layout: &BindGroupLayout, entries: &[BindGroupEntry]) -> Option<Self> {
        use wgpu::BindingResource as R;
        let entries = entries.iter().map(|entry| {
            let resource = match &entry.resource {
                R::Buffer(binding) => BoundResource::Buffer(BoundBuffer::of(binding)),
                R::BufferArray(bindings) => {
                    BoundResource::Buffers(bindings.iter().map(BoundBuffer::of).collect())
                }
                R::Sampler(sampler) => BoundResource::Sampler((*sampler).clone()),
                R::SamplerArray(samplers) => {
                    BoundResource::Samplers(samplers.iter().map(|s| (*s).clone()).collect())
                }
                R::TextureView(view) => BoundResource::TextureView((*view).clone()),
                R::TextureViewArray(views) => {
                    BoundResource::TextureViews(views.iter().map(|v| (*v).clone()).collect())
                }
                _ => return None,
            };
            Some((entry.binding, resource))
        });
        Some(Self {
            layout: layout.id(),
            entries: entries.collect::<Option<_>>()?,
        })
    }
}

/// Drops the bind groups that the frame didn't ask the [`RenderDevice`] for.
pub fn forget_unused_bind_groups(device: Res<RenderDevice>) {
    device.forget_unused_bind_groups();
}

impl From<wgpu::Device> for RenderDevice {
    fn from(device: wgpu::Device) -> Self {
        Self::new(WgpuWrapper::new(device))
    }
}

impl RenderDevice {
    pub fn new(device: WgpuWrapper<wgpu::Device>) -> Self {
        let features = device.features();
        let limits = device.limits();
        Self {
            device,
            features,
            limits,
            bind_groups: Arc::new(WgpuWrapper::new(Mutex::default())),
        }
    }

    /// List all [`Features`](wgpu::Features) that may be used with this device.
    ///
    /// Functions may panic if you use unsupported features.
    #[inline]
    pub fn features(&self) -> wgpu::Features {
        self.features
    }

    /// List all [`Limits`](wgpu::Limits) that were requested of this device.
    ///
    /// If any of these limits are exceeded, functions may panic.
    #[inline]
    pub fn limits(&self) -> wgpu::Limits {
        self.limits.clone()
    }

    /// Creates a [`ShaderModule`](wgpu::ShaderModule) from either SPIR-V or WGSL source code.
    ///
    /// # Safety
    ///
    /// Creates a shader module with user-customizable runtime checks which allows shaders to
    /// perform operations which can lead to undefined behavior like indexing out of bounds,
    /// To avoid UB, ensure any unchecked shaders are sound!
    /// This method should never be called for user-supplied shaders.
    #[inline]
    pub unsafe fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::ShaderModule {
        #[cfg(feature = "spirv_shader_passthrough")]
        match &desc.source {
            wgpu::ShaderSource::SpirV(source)
                if self
                    .features()
                    .contains(wgpu::Features::PASSTHROUGH_SHADERS) =>
            {
                // SAFETY:
                // This call passes binary data to the backend as-is and can potentially result in a driver crash or bogus behavior.
                // No attempt is made to ensure that data is valid SPIR-V.
                unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: desc.label,
                            spirv: Some(source.clone()),
                            ..Default::default()
                        },
                    )
                }
            }
            // SAFETY:
            //
            // This call passes binary data to the backend as-is and can potentially result in a driver crash or bogus behavior.
            // No attempt is made to ensure that data is valid SPIR-V.
            _ => unsafe {
                self.device
                    .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
            },
        }
        #[cfg(not(feature = "spirv_shader_passthrough"))]
        // SAFETY: the caller is responsible for upholding the safety requirements
        unsafe {
            self.device
                .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
        }
    }

    /// Creates and validates a [`ShaderModule`](wgpu::ShaderModule) from either SPIR-V or WGSL source code.
    ///
    /// See [`ValidateShader`](bevy_shader::ValidateShader) for more information on the tradeoffs involved with shader validation.
    #[inline]
    pub fn create_and_validate_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::ShaderModule {
        #[cfg(feature = "spirv_shader_passthrough")]
        match &desc.source {
            wgpu::ShaderSource::SpirV(_source) => panic!("no safety checks are performed for spirv shaders. use `create_shader_module` instead"),
            _ => self.device.create_shader_module(desc),
        }
        #[cfg(not(feature = "spirv_shader_passthrough"))]
        self.device.create_shader_module(desc)
    }

    /// Check for resource cleanups and mapping callbacks.
    ///
    /// Return `true` if the queue is empty, or `false` if there are more queue
    /// submissions still in flight. (Note that, unless access to the [`wgpu::Queue`] is
    /// coordinated somehow, this information could be out of date by the time
    /// the caller receives it. `Queue`s can be shared between threads, so
    /// other threads could submit new work at any time.)
    ///
    /// no-op on the web, device is automatically polled.
    #[inline]
    pub fn poll(&self, maintain: wgpu::PollType) -> Result<PollStatus, PollError> {
        self.device.poll(maintain)
    }

    /// Creates an empty [`CommandEncoder`](wgpu::CommandEncoder).
    #[inline]
    pub fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor,
    ) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(desc)
    }

    /// Creates an empty [`RenderBundleEncoder`](wgpu::RenderBundleEncoder).
    #[inline]
    pub fn create_render_bundle_encoder(
        &self,
        desc: &wgpu::RenderBundleEncoderDescriptor,
    ) -> wgpu::RenderBundleEncoder<'_> {
        self.device.create_render_bundle_encoder(desc)
    }

    /// Creates a new [`BindGroup`](wgpu::BindGroup), or returns a kept one
    /// made from the same layout and resources.
    ///
    /// A bind group stays kept until [`KEPT_FRAMES`] frames in a row leave
    /// it unused.
    pub fn create_bind_group<'a>(
        &self,
        label: impl Into<wgpu::Label<'a>>,
        layout: &'a BindGroupLayout,
        entries: &'a [BindGroupEntry<'a>],
    ) -> BindGroup {
        let label: wgpu::Label<'a> = label.into();
        let create = || {
            BindGroup::from(self.device.create_bind_group(&BindGroupDescriptor {
                label,
                layout,
                entries,
            }))
        };
        let Some(key) = BindGroupKey::of(layout, entries) else {
            return create();
        };
        let lock = || self.bind_groups.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(kept) = lock().get_mut(&key) {
            kept.idle = 0;
            return kept.group.clone();
        }
        // Made outside the lock, so threads that miss don't wait on each
        // other. Two that miss on the same key each make a group, and the
        // cache keeps the second.
        let group = create();
        let kept = KeptBindGroup {
            group: group.clone(),
            idle: 0,
        };
        lock().insert(key, kept);
        group
    }

    /// Drops each kept bind group that [`Self::create_bind_group`] hasn't
    /// returned for [`KEPT_FRAMES`] frames. Runs once a frame.
    pub fn forget_unused_bind_groups(&self) {
        let mut cache = self.bind_groups.lock().unwrap_or_else(PoisonError::into_inner);
        cache.retain(|_, kept| {
            kept.idle += 1;
            kept.idle <= KEPT_FRAMES
        });
    }

    /// Creates a [`BindGroupLayout`](wgpu::BindGroupLayout).
    #[inline]
    pub fn create_bind_group_layout<'a>(
        &self,
        label: impl Into<wgpu::Label<'a>>,
        entries: &'a [BindGroupLayoutEntry],
    ) -> BindGroupLayout {
        BindGroupLayout::from(
            self.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: label.into(),
                    entries,
                }),
        )
    }

    /// Creates a [`PipelineLayout`](wgpu::PipelineLayout).
    #[inline]
    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor,
    ) -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(desc)
    }

    /// Creates a [`RenderPipeline`].
    #[inline]
    pub fn create_render_pipeline(&self, desc: &RawRenderPipelineDescriptor) -> RenderPipeline {
        let wgpu_render_pipeline = self.device.create_render_pipeline(desc);
        RenderPipeline::from(wgpu_render_pipeline)
    }

    /// Creates a [`ComputePipeline`].
    #[inline]
    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor,
    ) -> ComputePipeline {
        let wgpu_compute_pipeline = self.device.create_compute_pipeline(desc);
        ComputePipeline::from(wgpu_compute_pipeline)
    }

    /// Creates a [`Buffer`].
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> Buffer {
        let wgpu_buffer = self.device.create_buffer(desc);
        Buffer::from(wgpu_buffer)
    }

    /// Creates a [`Buffer`] and initializes it with the specified data.
    pub fn create_buffer_with_data(&self, desc: &wgpu::util::BufferInitDescriptor) -> Buffer {
        let wgpu_buffer = self.device.create_buffer_init(desc);
        Buffer::from(wgpu_buffer)
    }

    /// Creates a new [`Texture`] and initializes it with the specified data.
    ///
    /// `desc` specifies the general format of the texture.
    /// `data` is the raw data.
    pub fn create_texture_with_data(
        &self,
        render_queue: &RenderQueue,
        desc: &wgpu::TextureDescriptor,
        order: wgpu::util::TextureDataOrder,
        data: &[u8],
    ) -> Texture {
        let wgpu_texture =
            self.device
                .create_texture_with_data(render_queue.as_ref(), desc, order, data);
        Texture::from(wgpu_texture)
    }

    /// Creates a new [`Texture`].
    ///
    /// `desc` specifies the general format of the texture.
    pub fn create_texture(&self, desc: &wgpu::TextureDescriptor) -> Texture {
        let wgpu_texture = self.device.create_texture(desc);
        Texture::from(wgpu_texture)
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &wgpu::SamplerDescriptor) -> Sampler {
        let wgpu_sampler = self.device.create_sampler(desc);
        Sampler::from(wgpu_sampler)
    }

    /// Initializes [`Surface`](wgpu::Surface) for presentation.
    ///
    /// # Panics
    ///
    /// - A old [`SurfaceTexture`](wgpu::SurfaceTexture) is still alive referencing an old surface.
    /// - Texture format requested is unsupported on the surface.
    pub fn configure_surface(&self, surface: &wgpu::Surface, config: &wgpu::SurfaceConfiguration) {
        surface.configure(&self.device, config);
    }

    /// Returns the wgpu [`Device`](wgpu::Device).
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn map_buffer(
        &self,
        buffer: &wgpu::BufferSlice,
        map_mode: wgpu::MapMode,
        callback: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static,
    ) {
        buffer.map_async(map_mode, callback);
    }

    // Rounds up `row_bytes` to be a multiple of [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`].
    pub const fn align_copy_bytes_per_row(row_bytes: usize) -> usize {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

        // If row_bytes is aligned calculate a value just under the next aligned value.
        // Otherwise calculate a value greater than the next aligned value.
        let over_aligned = row_bytes + align - 1;

        // Round the number *down* to the nearest aligned value.
        (over_aligned / align) * align
    }

    pub fn get_supported_read_only_binding_type(
        &self,
        buffers_per_shader_stage: u32,
    ) -> BufferBindingType {
        if self.limits().max_storage_buffers_per_shader_stage >= buffers_per_shader_stage {
            BufferBindingType::Storage { read_only: true }
        } else {
            BufferBindingType::Uniform
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_copy_bytes_per_row() {
        // Test for https://github.com/bevyengine/bevy/issues/16992
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

        assert_eq!(RenderDevice::align_copy_bytes_per_row(0), 0);
        assert_eq!(RenderDevice::align_copy_bytes_per_row(1), align);
        assert_eq!(RenderDevice::align_copy_bytes_per_row(align + 1), align * 2);
        assert_eq!(RenderDevice::align_copy_bytes_per_row(align), align);
    }
}
