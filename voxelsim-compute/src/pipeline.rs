use crate::rasterizer::camera::CameraMatrix;
use crate::rasterizer::{self, CellInstance, InstanceBuffer};
use crate::rasterizer::{BufferSet, RasterizerState};
use nalgebra::Matrix4;
use std::sync::Arc;
use voxelsim::{Cell, Coord, VoxelGrid};
use winit::event_loop::ActiveEventLoop;
use winit::{event::*, event_loop::EventLoop, window::Window};

// Main State struct to hold all wgpu-related objects
pub struct State {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub rasterizer_state: RasterizerState,
    pub filter_buffer: InstanceBuffer,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn create(window: Arc<Window>, world: &VoxelGrid) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The surface is the part of the window that we draw to.
        // This is unsafe because it involves raw window handles.
        let surface = instance.create_surface(window.clone()).unwrap();

        // The adapter is a handle to a physical graphics card.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // The device is the logical connection to the GPU, used to create resources.
        // The queue is used to submit command buffers to the GPU.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::PUSH_CONSTANTS, // Add features you need here
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: wgpu::Limits {
                    max_push_constant_size: 64, // Size of mat4x4<f32>
                    ..wgpu::Limits::default()
                },
                label: None,
                trace: wgpu::Trace::Off,
                memory_hints: wgpu::MemoryHints::Performance,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let rasterizer_state = RasterizerState::create(&device, world, [size.width, size.height]);

        let filter_buffer =
            CellInstance::create_instance_buffer_uninit(&device, world.cells().len());

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            rasterizer_state,
            filter_buffer,
        }
    }

    // Handles window resizing.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    // The main rendering function.
    pub async fn run(
        &mut self,
        camera_matrix: &CameraMatrix,
        filter_world: &VoxelGrid,
    ) -> Result<WorldChangeset, wgpu::SurfaceError> {
        // Get the current texture to render to from the swap chain.
        //let output = self.surface.get_current_texture()?;
        //let view = output
        //    .texture
        //    .create_view(&wgpu::TextureViewDescriptor::default());

        // A command encoder builds a command buffer that we can send to the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Rasterizer Encoder"),
            });

        self.rasterizer_state
            .encode_rasterizer(&mut encoder, camera_matrix);

        self.queue.submit(std::iter::once(encoder.finish()));
        //output.present();

        // Update filter buffer.
        CellInstance::write_instance_buffer(&self.queue, &self.filter_buffer, filter_world); // Run filter stage.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Filter Encoder"),
            });

        self.rasterizer_state.encode_filter(
            &mut encoder,
            camera_matrix,
            &self.filter_buffer.buf,
            self.filter_buffer.len,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let to_insert = rasterizer::texture::extract_cells_from_texture(
            &self.device,
            &self.queue,
            &self.rasterizer_state.render_target.texture,
        )
        .await;

        let to_remove = self
            .rasterizer_state
            .filter
            .get_filter_list(&self.device, &self.queue)
            .await;

        Ok(WorldChangeset {
            to_remove,
            to_insert,
        })
    }
}

pub struct WorldChangeset {
    pub to_insert: Vec<(Coord, Cell)>,
    pub to_remove: Vec<Coord>,
}
