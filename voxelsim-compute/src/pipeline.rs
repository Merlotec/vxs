use crate::rasterizer::camera::CameraMatrix;
use crate::rasterizer::{self, CellInstance, InstanceBuffer};
use crate::rasterizer::RasterizerState;
use nalgebra::Vector2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use voxelsim::viewport::{VirtualCell, VirtualGrid};
use voxelsim::{Cell, Coord, VoxelGrid}; // Main State struct to hold all wgpu-related objects
pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub size: Vector2<u32>,
    pub rasterizer_state: RasterizerState,
    pub filter_buffer: InstanceBuffer,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn create(world: &VoxelGrid, view_size: Vector2<u32>) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The adapter is a handle to a physical graphics card.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
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

        let rasterizer_state = RasterizerState::create(&device, world, [view_size.x, view_size.y]);

        let filter_buffer =
            CellInstance::create_instance_buffer_uninit(&device, world.cells().len());

        Self {
            device,
            queue,
            size: view_size,
            rasterizer_state,
            filter_buffer,
        }
    }

    // The main rendering function.
    pub async fn run(
        &mut self,
        camera_matrix: &CameraMatrix,
        filter_world: &VirtualGrid,
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

        let to_insert = rasterizer::texture::extract_texture_data(
            &self.device,
            &self.queue,
            &self.rasterizer_state.render_target.texture,
        )
        .await
        .unwrap();

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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct WorldChangeset {
    pub to_insert: Vec<(Coord, Cell)>,
    pub to_remove: Vec<Coord>,
}

#[cfg_attr(feature = "python", pyo3::prelude::pymethods)]
impl WorldChangeset {
    pub fn update_world(&self, world: &mut VirtualGrid) {
        self.to_insert.par_iter().for_each(|(coord, cell)| {
            world.cells().insert(
                *coord,
                VirtualCell {
                    cell: *cell,
                    uncertainty: 0.0,
                },
            );
        });
        self.to_remove.par_iter().for_each(|coord| {
            world.cells().remove(coord);
        });
    }
}
