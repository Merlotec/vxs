pub mod camera;
pub mod texture;

use nalgebra::Matrix4;
use voxelsim::VoxelGrid;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer,
};

use camera::CameraBinding;

use crate::rasterizer::{camera::CameraUniform, texture::TextureSet};

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

    // Describes how to pass this struct to the vertex shader.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0, // Corresponds to @location(0) in the vertex shader
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
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellInstance {
    pub position: [f32; 3],
    pub value: u32,
}

impl CellInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CellInstance>() as wgpu::BufferAddress,
            // This is the important part for instancing
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1, // Corresponds to @location(1) for position
                    format: wgpu::VertexFormat::Float32x3,
                },
                // NEW: Attribute for metadata
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 2, // Corresponds to @location(2) for metadata
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }

    pub fn create_instance_buffer(device: &wgpu::Device, world: &VoxelGrid) -> InstanceBuffer {
        let instance_data = world
            .cells()
            .iter()
            .map(|(p, v)| CellInstance {
                position: [p[0] as f32, p[1] as f32, p[2] as f32],
                value: v.bits(),
            })
            .collect::<Vec<_>>();

        let instance_len = instance_data.len();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
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
    pub camera_binding: CameraBinding,
}

impl RasterizerPipeline {
    pub fn create_pipeline(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_binding: CameraBinding,
    ) -> Self {
        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../../shaders/voxelview.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_binding.layout],
            push_constant_ranges: &[],
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
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
            // NEW: Add depth stencil state
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // 1.
                stencil: wgpu::StencilState::default(),     // 2.
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
            camera_binding,
        }
    }

    pub fn bind_and_draw(&self, encoder: wgpu::CommandEncoder) {}
}

pub struct RasterizerState {
    cube_buffer: BufferSet,
    instances: InstanceBuffer,
    camera_buffer: wgpu::Buffer,
    rasterizer_pipeline: RasterizerPipeline,
    depth: TextureSet,
    render_target: TextureSet,
}

impl RasterizerState {
    const DEPTH_TEXTURE_LABEL: &'static str = "DEPTH_TEXTURE_LABEL";
    const RENDER_TEXTURE_LABEL: &'static str = "RENDER_TEXTURE_LABEL";

    pub fn create(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        world: &VoxelGrid,
        texture_size: [u32; 2],
    ) -> Self {
        let cube_buffer: BufferSet = Vertex::create_cube_buffer(&device);

        let instances: InstanceBuffer = CellInstance::create_instance_buffer(&device, world);

        let camera_buffer: wgpu::Buffer = camera::CameraUniform::default().create_buffer(&device);

        let camera_binding: CameraBinding = CameraBinding::create(&device, &camera_buffer);

        let rasterizer_pipeline =
            RasterizerPipeline::create_pipeline(&device, config, camera_binding);

        let depth =
            TextureSet::create_depth_texture(device, texture_size, Self::DEPTH_TEXTURE_LABEL);

        let render_target = TextureSet::create_render_target_texture(
            device,
            texture_size,
            wgpu::TextureFormat::Rgba8UnormSrgb, //wgpu::TextureFormat::Rgba32Sint,
            Self::RENDER_TEXTURE_LABEL,
        );

        Self {
            cube_buffer,
            instances,
            camera_buffer,
            rasterizer_pipeline,
            depth,
            render_target,
        }
    }

    pub fn update_camera_buffer(&self, queue: &wgpu::Queue, uniform: CameraUniform) {
        uniform.write_buffer(queue, &self.camera_buffer);
    }

    pub fn encode(
        &self,
        mut encoder: wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        //let view = &self.render_target.view;
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view, // Render to our texture
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None, // May have to set this to something.
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
            render_pass.set_bind_group(0, &self.rasterizer_pipeline.camera_binding.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.cube_buffer.vertex.slice(..));
            render_pass.set_vertex_buffer(1, self.instances.buf.slice(..));
            render_pass
                .set_index_buffer(self.cube_buffer.index.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(
                0..Vertex::CUBE_INDICES.len() as u32,
                0,
                0..self.instances.len,
            );
        }

        encoder.finish()
    }
}
