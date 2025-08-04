use crate::rasterizer::RasterizerState;
use crate::rasterizer::camera::CameraMatrix;
use crate::rasterizer::noise::NoiseParams;
use crate::rasterizer::{CellInstance, InstanceBuffer};
use crate::buf::StagingBufferPool;
use nalgebra::Vector2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use voxelsim::viewport::{VirtualCell, VirtualGrid};
use voxelsim::{Cell, Coord, VoxelGrid};

pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub size: Vector2<u32>,
    pub rasterizer_state: RasterizerState,
    pub filter_buffer: InstanceBuffer,
    pub staging_pool: StagingBufferPool,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn create(world: &VoxelGrid, view_size: Vector2<u32>, noise: NoiseParams) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU

        let (queue, device) = {
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
                    required_features: wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::CLEAR_TEXTURE, // Add features you need here
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 76, // Size of mat4x4<f32>
                        ..wgpu::Limits::default()
                    },
                    label: None,
                    trace: wgpu::Trace::Off,
                    memory_hints: wgpu::MemoryHints::Performance,
                })
                .await
                .unwrap();

            (queue, device)
        };
        let rasterizer_state =
            RasterizerState::create(&device, world, [view_size.x, view_size.y], noise);

        let filter_buffer =
            CellInstance::create_instance_buffer_uninit(&device, world.cells().len());

        let staging_pool = StagingBufferPool::new(device.clone());

        Self {
            device,
            queue,
            size: view_size,
            rasterizer_state,
            filter_buffer,
            staging_pool,
        }
    }

    // The main rendering function.
    pub async fn run(
        &mut self,
        camera_matrix: &CameraMatrix,
        filter_world: &VirtualGrid,
    ) -> Result<WorldChangeset, wgpu::SurfaceError> {
        let total_start = std::time::Instant::now();
        // Get the current texture to render to from the swap chain.
        //let output = self.surface.get_current_texture()?;
        //let view = output
        //    .texture
        //    .create_view(&wgpu::TextureViewDescriptor::default());

        // === PHASE 1: GPU Command Encoding ===
        let encoding_start = std::time::Instant::now();
        
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

        // Submit the main rendering commands and capture the submission index
        let render_submission = self.queue.submit(std::iter::once(encoder.finish()));
        
        let encoding_time = encoding_start.elapsed();
        println!("📋 GPU Command Encoding: {:.2}ms", encoding_time.as_secs_f64() * 1000.0);

        // === PHASE 2: Prepare Read Operations ===
        let prepare_start = std::time::Instant::now();
        
        // Prepare both operations without polling
        let texture_op = self.staging_pool.prepare_texture_extraction(
            &self.queue,
            &self.rasterizer_state.render_target.texture,
        );

        let filter_op = self.staging_pool.prepare_buffer_read(
            &self.queue,
            &self.rasterizer_state.filter.output_buffer,
            true,
        );
        
        let prepare_time = prepare_start.elapsed();
        println!("⚙️  Read Operations Setup: {:.2}ms", prepare_time.as_secs_f64() * 1000.0);

        // Collect submission indices before moving the operations
        let texture_submission = texture_op.submission_index.clone();
        let filter_submission = filter_op.submission_index.clone();

        // === PHASE 3: GPU Polling ===
        let polling_start = std::time::Instant::now();
        
        // Wait for all specific submissions to complete
        self.staging_pool.wait_for_submissions(&[
            render_submission,
            texture_submission,
            filter_submission,
        ]);
        
        let polling_time = polling_start.elapsed();
        println!("⏳ GPU Submission Polling: {:.2}ms", polling_time.as_secs_f64() * 1000.0);

        // === PHASE 4: Parallel Buffer Reads ===
        let parallel_read_start = std::time::Instant::now();
        
        // Execute both read operations in parallel
        let results = self.staging_pool.execute_batched_reads_parallel(vec![texture_op, filter_op]).await;
        
        let parallel_read_time = parallel_read_start.elapsed();
        println!("🚀 Parallel Buffer Reads: {:.2}ms", parallel_read_time.as_secs_f64() * 1000.0);
        
        // === PHASE 5: Data Processing ===
        let processing_start = std::time::Instant::now();
        
        // Extract results
        let to_insert_raw = results[0].as_ref()
            .unwrap_or_else(|e| panic!("Texture extraction failed: {}", e))
            .clone();
        let filter_data = results[1].as_ref()
            .unwrap_or_else(|e| panic!("Filter buffer read failed: {}", e))
            .clone();

        // Process the raw texture data
        let to_insert: Vec<(voxelsim::Coord, voxelsim::Cell)> = unsafe {
            let data_slice = std::slice::from_raw_parts::<(voxelsim::Coord, voxelsim::Cell)>(
                to_insert_raw.as_ptr() as *const (voxelsim::Coord, voxelsim::Cell),
                to_insert_raw.len() / std::mem::size_of::<(voxelsim::Coord, voxelsim::Cell)>(),
            );
            data_slice.to_vec()
        };

        // Process the filter data
        let count = u32::from_le_bytes(filter_data[0..4].try_into().unwrap());
        let output_data_slice: &[voxelsim::Coord] = bytemuck::cast_slice(&filter_data[4..]);
        let to_remove = output_data_slice[..count as usize].to_vec();
        
        let processing_time = processing_start.elapsed();
        println!("🔄 Data Processing: {:.2}ms", processing_time.as_secs_f64() * 1000.0);

        // === PHASE 6: Cleanup ===
        let cleanup_start = std::time::Instant::now();

        self.rasterizer_state
            .filter
            .clear_buffers(&self.device, &self.queue)
            .await;
            
        let cleanup_time = cleanup_start.elapsed();
        println!("🧹 Buffer Cleanup: {:.2}ms", cleanup_time.as_secs_f64() * 1000.0);

        // === TOTAL PIPELINE TIME ===
        let total_time = total_start.elapsed();
        println!("🏁 TOTAL PIPELINE: {:.2}ms", total_time.as_secs_f64() * 1000.0);
        println!("📊 Items: {} to_insert, {} to_remove", to_insert.len(), to_remove.len());
        println!("═══════════════════════════════════════");

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
            if !cell.is_empty() {
                world.cells().insert(
                    *coord,
                    VirtualCell {
                        cell: *cell,
                        uncertainty: 0.0,
                    },
                );
            }
        });
        self.to_remove.par_iter().for_each(|coord| {
            world.cells().remove(coord);
        });
    }
}
