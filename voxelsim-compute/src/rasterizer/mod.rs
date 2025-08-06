pub mod camera;
pub mod culling;
pub mod filter;
pub mod noise;
pub mod texture;

use crate::rasterizer::{
    culling::FrustumCulling,
    filter::FilterBindings,
    noise::{NoiseBuffer, NoiseParams},
    texture::TextureSet,
};
use camera::CameraMatrix;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use voxelsim::{Coord, VoxelGrid, viewport::VirtualGrid};
use wgpu::{
    Buffer,
    util::{BufferInitDescriptor, DeviceExt},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl Vertex {
    const CUBE_VERTEX_BUFFER_LABEL: &'static str = "CUBE_VERTEX_BUFFER_LABEL";
    const CUBE_INDEX_BUFFER_LABEL: &'static str = "CUBE_INDEX_BUFFER_LABEL";

    const CUBE_VERTICES: &[Vertex] = &[
        Vertex {
            position: [-0.5, -0.5, -0.5],
        }, // 0
        Vertex {
            position: [0.5, -0.5, -0.5],
        }, // 1
        Vertex {
            position: [0.5, 0.5, -0.5],
        }, // 2
        Vertex {
            position: [-0.5, 0.5, -0.5],
        }, // 3
        Vertex {
            position: [-0.5, -0.5, 0.5],
        }, // 4
        Vertex {
            position: [0.5, -0.5, 0.5],
        }, // 5
        Vertex {
            position: [0.5, 0.5, 0.5],
        }, // 6
        Vertex {
            position: [-0.5, 0.5, 0.5],
        }, // 7
    ];

    const CUBE_INDICES: &[u16] = &[
        0, 1, 2, 0, 2, 3, // Front
        4, 6, 5, 4, 7, 6, // Back
        4, 0, 3, 4, 3, 7, // Left
        1, 5, 6, 1, 6, 2, // Right
        3, 2, 6, 3, 6, 7, // Top
        4, 5, 1, 4, 1, 0, // Bottom
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }

    pub fn create_cube_buffer(device: &wgpu::Device) -> BufferSet {
        let vbuf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(Self::CUBE_VERTEX_BUFFER_LABEL),
            contents: bytemuck::cast_slice(Self::CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(Self::CUBE_INDEX_BUFFER_LABEL),
            contents: bytemuck::cast_slice(Self::CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        BufferSet {
            vertex: vbuf,
            index: ibuf,
        }
    }
}

pub struct BufferSet {
    vertex: Buffer,
    index: Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellInstance {
    pub position: Coord,
    pub value: u32,
}

impl CellInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CellInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Sint32x3,  // Changed from x4 to x3 to match Vector3<i32>
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[i32; 3]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }

    pub fn create_instance_buffer_uninit(device: &wgpu::Device, len: usize) -> InstanceBuffer {
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<Self>() * len) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        InstanceBuffer {
            buf: instance_buffer,
            len: len as u32,
        }
    }

    pub fn write_instance_buffer(
        queue: &wgpu::Queue,
        instance: &InstanceBuffer,
        world: &VirtualGrid,
    ) {
        let instance_data = world
            .cells()
            .par_iter()
            .map(|r| {
                let (p, v) = r.pair();
                CellInstance {
                    position: *p,
                    value: v.cell.bits(),
                }
            })
            .collect::<Vec<_>>();
        assert!(instance_data.len() <= instance.len as usize);

        queue.write_buffer(&instance.buf, 0, bytemuck::cast_slice(&instance_data));
    }

    pub fn create_instance_buffer(device: &wgpu::Device, world: &VoxelGrid) -> InstanceBuffer {
        let instance_data = world
            .cells()
            .par_iter()
            .map(|r| {
                let (p, v) = r.pair();
                CellInstance {
                    position: *p,
                    value: v.bits(),
                }
            })
            .collect::<Vec<_>>();

        let instance_len = instance_data.len();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        InstanceBuffer {
            buf: instance_buffer,
            len: instance_len as u32,
        }
    }
}

pub struct InstanceBuffer {
    pub buf: Buffer,
    pub len: u32,
}

pub struct RasterizerPipeline {
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl RasterizerPipeline {
    pub fn create_pipeline(device: &wgpu::Device) -> Self {
        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../../shaders/voxelcoord.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&NoiseBuffer::layout(device)], // No longer need a bind group for the camera
            // Add a push constant range for the camera matrix
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<CameraMatrix>() as u32,
            }],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), CellInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Sint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline_layout,
            pipeline,
        }
    }

    pub fn create_filter_pipeline(
        device: &wgpu::Device,
        filter_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/filter.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[filter_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<CameraMatrix>() as u32,
            }],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Filter Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), CellInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // No depth testing - we manually compare depth in fragment shader
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline_layout,
            pipeline,
        }
    }
}

pub struct RasterizerState {
    cube_buffer: BufferSet,
    instances: InstanceBuffer,
    pub filter: FilterBindings,
    pub noise: NoiseBuffer,
    pub frustum_culling: FrustumCulling,
    pub filter_frustum_culling: FrustumCulling, // Separate culling for filter stage
    rasterizer_pipeline: RasterizerPipeline,
    filter_pipeline: RasterizerPipeline,
    depth: TextureSet,
    pub render_target: TextureSet,
    pub filter_render_target: TextureSet,
    pub depth_sampler: wgpu::Sampler,
}

impl RasterizerState {
    const DEPTH_TEXTURE_LABEL: &'static str = "DEPTH_TEXTURE_LABEL";
    const RENDER_TEXTURE_LABEL: &'static str = "RENDER_TEXTURE_LABEL";

    pub fn instance_count(&self) -> u32 {
        self.instances.len
    }
    
    pub fn instances_buffer(&self) -> &wgpu::Buffer {
        &self.instances.buf
    }

    pub fn create(
        device: &wgpu::Device,
        world: &VoxelGrid,
        texture_size: [u32; 2],
        noise: NoiseParams,
    ) -> Self {
        let cube_buffer: BufferSet = Vertex::create_cube_buffer(device);
        let instances: InstanceBuffer = CellInstance::create_instance_buffer(device, world);
        let depth =
            TextureSet::create_depth_texture(device, texture_size, Self::DEPTH_TEXTURE_LABEL);
        // TODO: Here we reserve more memory than we need (assuming insances.len is the max cells in the
        // abstract world, which is not exactly true...!!).
        let depth_sampler = texture::create_depth_sampler(device);
        let filter =
            FilterBindings::create(device, &depth.view, &depth_sampler, instances.len as usize);
        // Camera buffer and binding are no longer needed here
        let rasterizer_pipeline = RasterizerPipeline::create_pipeline(device);
        let filter_pipeline = RasterizerPipeline::create_filter_pipeline(device, &filter.layout);

        let render_target = TextureSet::create_render_target_texture(
            device,
            texture_size,
            wgpu::TextureFormat::Rgba32Sint,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            Self::RENDER_TEXTURE_LABEL,
        );

        let filter_render_target = TextureSet::create_render_target_texture(
            device,
            texture_size,
            wgpu::TextureFormat::R8Uint,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            Self::RENDER_TEXTURE_LABEL,
        );

        let noise_buffer = NoiseBuffer::create_buffer(device, noise);

        // Create frustum culling system with a generous buffer size
        let max_instances = (instances.len * 2).max(10000); // Allow for growth
        let mut frustum_culling = FrustumCulling::new(device, max_instances);
        frustum_culling.create_bind_group(device, &instances.buf);
        
        // Create separate frustum culling for filter stage
        // Filter typically has fewer instances, but use same max for safety
        let mut filter_frustum_culling = FrustumCulling::new(device, max_instances);
        // Create bind group for filter culling (will be updated later in pipeline)
        filter_frustum_culling.create_bind_group(device, &instances.buf); // Temporary bind group

        Self {
            cube_buffer,
            instances,
            filter,
            noise: noise_buffer,
            frustum_culling,
            filter_frustum_culling,
            rasterizer_pipeline,
            filter_pipeline,
            depth,
            render_target,
            depth_sampler,
            filter_render_target,
        }
    }

    pub fn encode_frustum_culling(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera_uniform: &CameraMatrix,
    ) {
        // Pass the camera matrix directly (matches voxelcoord.wgsl approach)
        self.frustum_culling.dispatch_culling(
            encoder,
            queue,
            camera_uniform,
            self.instances.len,
        );
    }

    pub fn encode_filter_frustum_culling(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        camera_uniform: &CameraMatrix,
        filter_instance_buffer: &wgpu::Buffer,
        filter_len: u32,
    ) {
        // Create/update bind group for filter culling (since filter buffer can change)
        self.filter_frustum_culling.create_bind_group(device, filter_instance_buffer);
        
        // Dispatch filter frustum culling
        self.filter_frustum_culling.dispatch_culling(
            encoder,
            queue,
            camera_uniform,
            filter_len,
        );
    }

    pub fn encode_rasterizer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraMatrix, // Pass camera data directly
        use_culled_instances: bool,
        visible_count: Option<u32>,
    ) {
        let view = &self.render_target.view;
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Voxel Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.rasterizer_pipeline.pipeline);
        render_pass.set_bind_group(0, &self.noise.group, &[]);
        // Set the push constant data for the vertex shader
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX,
            0,
            bytemuck::bytes_of(camera_uniform),
        );

        // The rest of the drawing logic - choose between original and culled instances
        render_pass.set_vertex_buffer(0, self.cube_buffer.vertex.slice(..));
        
        let (instance_buffer, instance_count) = if use_culled_instances {
            (&self.frustum_culling.culled_instance_buffer, visible_count.unwrap_or(0))
        } else {
            (&self.instances.buf, self.instances.len)
        };
        
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_buffer.index.slice(..), wgpu::IndexFormat::Uint16);
        
        render_pass.draw_indexed(
            0..Vertex::CUBE_INDICES.len() as u32,
            0,
            0..instance_count,
        );
    }

    pub fn encode_rasterizer_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraMatrix,
    ) {
        let view = &self.render_target.view;
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Voxel Render Pass (Indirect)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.rasterizer_pipeline.pipeline);
        render_pass.set_bind_group(0, &self.noise.group, &[]);
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX,
            0,
            bytemuck::bytes_of(camera_uniform),
        );

        // Set vertex buffers
        render_pass.set_vertex_buffer(0, self.cube_buffer.vertex.slice(..));
        render_pass.set_vertex_buffer(1, self.frustum_culling.culled_instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_buffer.index.slice(..), wgpu::IndexFormat::Uint16);
        
        // Use indirect draw with GPU-generated draw command
        render_pass.draw_indexed_indirect(&self.frustum_culling.indirect_draw_buffer, 0);
    }

    pub fn encode_filter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraMatrix,
        filter_instance_buffer: &wgpu::Buffer,
        filter_len: u32,
        use_culled_instances: bool,
        culled_visible_count: Option<u32>,
    ) {
        let view = &self.filter_render_target.view;
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Filter Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None, // No depth attachment - we only read depth via texture binding
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.filter_pipeline.pipeline);
        render_pass.set_bind_group(0, &self.filter.group, &[]);
        // Set the push constant data for the vertex shader
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX,
            0,
            bytemuck::bytes_of(camera_uniform),
        );

        // The rest of the drawing logic - choose between original and culled instances
        render_pass.set_vertex_buffer(0, self.cube_buffer.vertex.slice(..));
        
        let (instance_buffer, instance_count) = if use_culled_instances {
            (&self.filter_frustum_culling.culled_instance_buffer, culled_visible_count.unwrap_or(0))
        } else {
            (filter_instance_buffer, filter_len)
        };
        
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_buffer.index.slice(..), wgpu::IndexFormat::Uint16);

        render_pass.draw_indexed(0..Vertex::CUBE_INDICES.len() as u32, 0, 0..instance_count);
    }

    pub fn encode_filter_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraMatrix,
    ) {
        let view = &self.filter_render_target.view;
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Filter Render Pass (Indirect)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.filter_pipeline.pipeline);
        render_pass.set_bind_group(0, &self.filter.group, &[]);
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX,
            0,
            bytemuck::bytes_of(camera_uniform),
        );

        // Set vertex buffers
        render_pass.set_vertex_buffer(0, self.cube_buffer.vertex.slice(..));
        render_pass.set_vertex_buffer(1, self.filter_frustum_culling.culled_instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_buffer.index.slice(..), wgpu::IndexFormat::Uint16);

        // Use indirect draw with GPU-generated draw command
        render_pass.draw_indexed_indirect(&self.filter_frustum_culling.indirect_draw_buffer, 0);
    }
}
