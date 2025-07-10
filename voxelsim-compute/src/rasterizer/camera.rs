use nalgebra::{Matrix4, Vector3};
use wgpu::util::DeviceExt;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct CameraBinding {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl CameraBinding {
    pub fn create(device: &wgpu::Device, camera_buffer: &wgpu::Buffer) -> Self {
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        Self {
            layout: camera_bind_group_layout,
            bind_group: camera_bind_group,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        let v = nalgebra_glm::look_at_lh(
            &Vector3::<f32>::new(0.0, 10.0, 0.0),
            &Vector3::<f32>::new(-10.0, 0.0, -10.0),
            &Vector3::<f32>::new(0.0, 1.0, 0.0),
        );
        let p = nalgebra_glm::perspective(1.4, 1.3, 1.0, 200.0);
        Self::from_view_proj(p * v)
    }
}

impl CameraUniform {
    const CAMERA_BUFFER_LABEL: &'static str = "CAMERA_BUFFER_LABEL";

    pub fn from_view_proj(camera_view_proj: Matrix4<f32>) -> Self {
        Self {
            view_proj: (camera_view_proj).into(),
        }
    }

    pub fn create_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(Self::CAMERA_BUFFER_LABEL),
            contents: bytemuck::cast_slice(&[*self]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        camera_buffer
    }

    pub fn write_buffer(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[*self]));
    }
}
