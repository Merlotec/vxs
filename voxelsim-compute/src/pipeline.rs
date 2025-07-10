use crate::rasterizer::camera::{CameraBinding, CameraUniform};
use crate::rasterizer::{self, InstanceBuffer};
use crate::rasterizer::{BufferSet, RasterizerState};
use nalgebra::Matrix4;
use std::sync::Arc;
use voxelsim::VoxelGrid;
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
                required_features: wgpu::Features::empty(), // Add features you need here
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
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

        let rasterizer_state =
            RasterizerState::create(&device, &config, world, [size.width, size.height]);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            rasterizer_state,
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

    // Handles input events. Returns true if the event was fully processed.
    #[allow(unused_variables)]
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        // Return true if you handled the event, false otherwise.
        // For example, camera controls would go here.
        false
    }

    // Update application state (e.g., animations, physics).
    pub fn update(&mut self) {
        // This is where you would update your application logic before rendering.
    }

    // The main rendering function.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Get the current texture to render to from the swap chain.
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // A command encoder builds a command buffer that we can send to the GPU.
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        let cmd_buf = self.rasterizer_state.encode(encoder, &view);
        self.queue.submit(std::iter::once(cmd_buf));
        output.present();

        Ok(())
    }
}
