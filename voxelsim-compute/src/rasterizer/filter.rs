use voxelsim::Coord;

pub struct FilterBindings {
    pub layout: wgpu::BindGroupLayout,
    pub group: wgpu::BindGroup,
    pub output_buffer: wgpu::Buffer,
    pub output_buffer_size: wgpu::BufferAddress,
    pub flags_buffer: wgpu::Buffer,
    pub flags_buffer_size: wgpu::BufferAddress,
}

impl FilterBindings {
    pub async fn get_filter_list(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Coord> {
        let data = crate::buf::read_gpu_buffer(device, queue, &self.output_buffer, true).await;
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap());

        let output_data_slice: &[Coord] = bytemuck::cast_slice(&data[4..]);
        let valid = &output_data_slice[..count as usize];

        valid.to_vec()
    }

    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                // External Depth Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler for the depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), // Or NonFiltering
                    count: None,
                },
                // Output Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Atomic Flags Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn create(
        device: &wgpu::Device,
        depth_texture_view: &wgpu::TextureView,
        depth_texture_sampler: &wgpu::Sampler,
        instance_count: usize,
    ) -> Self {
        let output_buffer_size =
            (4 + std::mem::size_of::<Coord>() * instance_count) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            // Needs STORAGE for shader writes, and COPY_SRC to read it back on the CPU.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The atomic flag buffer, one u32 per instance.
        let flags_buffer_size =
            (std::mem::size_of::<u32>() * instance_count) as wgpu::BufferAddress;
        let flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atomic Flags Buffer"),
            size: flags_buffer_size,
            // Needs STORAGE for shader writes, and COPY_DST to clear it each frame.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = Self::layout(device);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(depth_texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flags_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            layout,
            group,
            output_buffer,
            output_buffer_size,
            flags_buffer,
            flags_buffer_size,
        }
    }

    pub async fn clear_buffers(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        crate::buf::clear_gpu_buffer(device, queue, &self.flags_buffer).await
    }
}
