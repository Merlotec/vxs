use nalgebra::Matrix4;
use voxelsim::VoxelGrid;
use winit::{event::*, event_loop::EventLoop, window::Window};

use crate::rasterizer::camera::{CameraBinding, CameraUniform};
use crate::rasterizer::{self, InstanceBuffer};
use crate::rasterizer::{BufferSet, RasterizerState};

// Main State struct to hold all wgpu-related objects
struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    rasterizer_state: RasterizerState,
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    async fn new(window: &'a Window, world: &VoxelGrid) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The surface is the part of the window that we draw to.
        // This is unsafe because it involves raw window handles.
        let surface = instance.create_surface(window).unwrap();

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

        let cube_buffer: BufferSet = rasterizer::Vertex::create_cube_buffer(&device);

        let instances: InstanceBuffer =
            rasterizer::CellInstance::create_instance_buffer(&device, world);

        let camera_buffer: wgpu::Buffer =
            CameraUniform::create_buffer(&device, Matrix4::identity());

        let camera_binding: CameraBinding = CameraBinding::create(&device, &camera_buffer);

        let rasterizer_pipeline =
            rasterizer::RasterizerPipeline::create_pipeline(&device, &config, camera_binding);

        Self {
            surface,
            device,
            queue,
            config,
            size,
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
    fn input(&mut self, event: &WindowEvent) -> bool {
        // Return true if you handled the event, false otherwise.
        // For example, camera controls would go here.
        false
    }

    // Update application state (e.g., animations, physics).
    fn update(&mut self) {
        // This is where you would update your application logic before rendering.
    }

    // The main rendering function.
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Get the current texture to render to from the swap chain.
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // A command encoder builds a command buffer that we can send to the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        // ====================================================================
        // START OF YOUR IMPLEMENTATION
        // ====================================================================

        // 1. RASTERIZATION PASS (to an off-screen buffer)
        //    - Create a target texture for rasterization output (e.g., with world coordinates).
        //    - Create a render pipeline for drawing your voxels.
        //    - Create a render pass that targets your off-screen texture.
        //    - Bind your voxel vertex buffer and draw all instances.

        // 2. COMPUTE PASS
        //    - Create a compute pipeline.
        //    - Create storage buffers (SSBOs) for the compute shader's input (the texture from step 1)
        //      and output (the new voxel grid).
        //    - Create a bind group for these resources.
        //    - Create a compute pass.
        //    - Set the pipeline and bind group.
        //    - Dispatch the compute shader.

        // 3. (Optional) FINAL RENDER PASS (to the screen)
        //    - You could use another render pass to visualize the output of your compute shader
        //      on the screen for debugging.
        //    - The code below is a simple example of a render pass that clears the screen.

        {
            // This is a placeholder render pass that just clears the screen.
            // You will replace or augment this with your actual rendering logic.
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // In a real application, you would set your pipeline and draw here.
            // render_pass.set_pipeline(&self.render_pipeline);
            // render_pass.draw(0..3, 0..1);
        }

        // ====================================================================
        // END OF YOUR IMPLEMENTATION
        // ====================================================================

        // Submit the command buffer to the GPU's command queue.
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// Main entry point for the application
pub async fn run() {
    // Set up logging
    env_logger::init();

    // Create the event loop and window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    window.set_title("WGPU Voxel Processor");

    // Create the wgpu state
    let mut state = State::new(window).await;

    // Start the event loop
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = winit::event_loop::ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}
