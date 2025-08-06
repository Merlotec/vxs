use crate::buf::StagingBufferPool;
use crate::rasterizer::RasterizerState;
use crate::rasterizer::camera::CameraMatrix;
use crate::rasterizer::noise::NoiseParams;
use crate::rasterizer::{CellInstance, InstanceBuffer};
use bytemuck::{Pod, Zeroable};
use nalgebra::Vector2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use voxelsim::viewport::{VirtualCell, VirtualGrid};
use voxelsim::{Cell, Coord, VoxelGrid};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Default, Pod, Zeroable)]
pub struct FilterCoord {
    coord: Coord,
    _p0: i32,
}

// FilterWorld with optional pre-uploaded buffer
#[derive(Debug, Clone)]
pub struct FilterWorld {
    pub grid: VirtualGrid,
    pub buffer: Option<Vec<CellInstance>>, // Pre-transformed GPU data
}

impl FilterWorld {
    pub fn new(grid: VirtualGrid) -> Self {
        Self { grid, buffer: None }
    }
    
    pub fn from_grid(grid: VirtualGrid) -> Self {
        Self { grid, buffer: None }
    }
}

// Enum for passing either VirtualGrid or FilterWorld to run()
#[derive(Debug)]
pub enum FilterInput<'a> {
    VirtualGrid(&'a VirtualGrid),
    FilterWorld(&'a FilterWorld),
}

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
                        max_push_constant_size: 80, // Size of mat4x4<f32>
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

    // Async function to prepare a FilterWorld buffer in background
    pub async fn prepare_filter_world(filter_world: &mut FilterWorld) -> Option<wgpu::SubmissionIndex> {
        if filter_world.buffer.is_some() {
            return None; // Already prepared
        }
        
        // Transform VirtualGrid to GPU format asynchronously
        let instance_data = tokio::task::spawn_blocking({
            let grid = filter_world.grid.clone();
            move || {
                grid.cells()
                    .par_iter()
                    .map(|r| {
                        let (p, v) = r.pair();
                        CellInstance {
                            position: *p,
                            value: v.cell.bits(),
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }).await.unwrap();
        
        filter_world.buffer = Some(instance_data);
        None // No GPU submission happened here, just CPU work
    }

    // Async function to prepare FilterWorld buffer with GPU upload and return submission index
    pub async fn prepare_filter_world_with_upload(
        &self, 
        filter_world: &mut FilterWorld
    ) -> Option<wgpu::SubmissionIndex> {
        if filter_world.buffer.is_some() {
            return None; // Already prepared
        }
        
        // Transform VirtualGrid to GPU format asynchronously
        let instance_data = tokio::task::spawn_blocking({
            let grid = filter_world.grid.clone();
            move || {
                grid.cells()
                    .par_iter()
                    .map(|r| {
                        let (p, v) = r.pair();
                        CellInstance {
                            position: *p,
                            value: v.cell.bits(),
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }).await.unwrap();
        
        // Upload to GPU buffer and get submission index
        self.queue.write_buffer(&self.filter_buffer.buf, 0, bytemuck::cast_slice(&instance_data));
        filter_world.buffer = Some(instance_data);
        
        // Submit an empty command buffer to get a submission index for synchronization
        let submission_index = self.queue.submit(std::iter::empty());
        Some(submission_index)
    }

    // The main rendering function.
    pub async fn run(
        &mut self,
        camera_matrix: &CameraMatrix,
        filter_input: FilterInput<'_>,
    ) -> Result<WorldChangeset, wgpu::SurfaceError> {
        let profile_enabled = std::env::var("VXS_COMPUTE_PROFILE").is_ok();
        let total_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        // Get the current texture to render to from the swap chain.
        //let output = self.surface.get_current_texture()?;
        //let view = output
        //    .texture
        //    .create_view(&wgpu::TextureViewDescriptor::default());

        // === PHASE 1: PARALLEL FRUSTUM CULLING ===
        let culling_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };

        // Clear filter buffers and prepare filter buffer FIRST (needed for filter culling)
        let _buffer_prep_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        
        self.rasterizer_state
            .filter
            .clear_buffers(&self.device, &self.queue)
            .await;
        
        let buffer_write_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        
        // Handle different input types and track buffer upload submission if needed
        let (_filter_grid, buffer_len, filter_buffer_submission) = match &filter_input {
            FilterInput::VirtualGrid(grid) => {
                // Regular upload from VirtualGrid
                CellInstance::write_instance_buffer(&self.queue, &self.filter_buffer, grid);
                let submission = self.queue.submit(std::iter::empty()); // Get submission index for sync
                if profile_enabled {
                    println!("💾 Regular filter buffer upload from VirtualGrid");
                }
                (*grid, grid.cells().len() as u32, Some(submission))
            }
            FilterInput::FilterWorld(filter_world) => {
                if let Some(ref buffer_data) = filter_world.buffer {
                    // Use pre-prepared buffer - buffer upload may have happened earlier
                    self.queue.write_buffer(&self.filter_buffer.buf, 0, bytemuck::cast_slice(buffer_data));
                    let submission = self.queue.submit(std::iter::empty()); // Get submission index for sync
                    if profile_enabled {
                        println!("🚀 Used pre-prepared FilterWorld buffer");
                    }
                    (&filter_world.grid, buffer_data.len() as u32, Some(submission))
                } else {
                    // Fall back to regular upload
                    CellInstance::write_instance_buffer(&self.queue, &self.filter_buffer, &filter_world.grid);
                    let submission = self.queue.submit(std::iter::empty());
                    if profile_enabled {
                        println!("💾 Regular filter buffer upload from FilterWorld (no buffer prepared)");
                    }
                    (&filter_world.grid, filter_world.grid.cells().len() as u32, Some(submission))
                }
            }
        };
        
        if let Some(start) = buffer_write_start {
            let buffer_write_time = start.elapsed();
            println!(
                "💾 Filter Buffer Write: {:.2}ms",
                buffer_write_time.as_secs_f64() * 1000.0
            );
        }

        // Create TWO culling encoders to run in parallel
        let mut main_cull_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Main Frustum Culling Encoder"),
        });

        let mut filter_cull_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Filter Frustum Culling Encoder"),
        });

        // Encode main frustum culling
        self.rasterizer_state.encode_frustum_culling(&mut main_cull_encoder, &self.queue, camera_matrix);

        // Encode filter frustum culling in parallel
        self.rasterizer_state.encode_filter_frustum_culling(
            &mut filter_cull_encoder,
            &self.queue,
            &self.device,
            camera_matrix,
            &self.filter_buffer.buf,
            buffer_len,
        );

        // Submit BOTH culling operations together for maximum parallelism
        let culling_submissions = self.queue.submit([
            main_cull_encoder.finish(),
            filter_cull_encoder.finish(),
        ].into_iter());

        if let Some(start) = culling_start {
            let culling_time = start.elapsed();
            println!(
                "✂️  Parallel Culling (Main + Filter): {:.2}ms",
                culling_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 2: GPU Command Encoding ===
        
        // Wait for filter buffer upload AND both culling operations to complete
        let mut submissions_to_wait = vec![culling_submissions.clone()];
        if let Some(filter_submission) = &filter_buffer_submission {
            submissions_to_wait.push(filter_submission.clone());
            if profile_enabled {
                println!("⏳ Waiting for filter buffer upload before main pipeline...");
            }
        }
        
        for submission in submissions_to_wait {
            self.device
                .poll(wgpu::wgt::PollType::WaitForSubmissionIndex(submission))
                .unwrap();
        }

        // Initialize indirect draw buffers with cube index count
        let cube_index_count = 36u32; // 6 faces * 6 indices per face
        self.rasterizer_state
            .frustum_culling
            .initialize_indirect_draw_buffer(&self.queue, cube_index_count);
        self.rasterizer_state
            .filter_frustum_culling
            .initialize_indirect_draw_buffer(&self.queue, cube_index_count);

        if profile_enabled {
            println!("    🎯 Using GPU-driven indirect draws (no readbacks needed)");
            println!("    📊 Main instances: {}", self.rasterizer_state.instance_count());
            println!("    🎯 Filter instances: {}", buffer_len);
        }

        // Debug: Check indirect draw commands AFTER culling completes
        if profile_enabled {
            println!("🔍 AFTER CULLING - Main indirect draw command:");
            self.rasterizer_state
                .frustum_culling
                .debug_read_indirect_draw_command(&self.device, &self.queue);
            println!("🔍 AFTER CULLING - Filter indirect draw command:");
            self.rasterizer_state
                .filter_frustum_culling
                .debug_read_indirect_draw_command(&self.device, &self.queue);
        }

        // Start GPU Command Encoding timer right before actual encoding begins
        let encoding_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };

        // A command encoder builds a command buffer that we can send to the GPU.
        let main_encode_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Rasterizer Encoder"),
            });

        // Back to GPU-driven indirect rendering 
        self.rasterizer_state.encode_rasterizer_indirect(
            &mut encoder,
            camera_matrix,
        );

        let main_submit_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        self.queue.submit(std::iter::once(encoder.finish()));
        
        if let Some(start) = main_submit_start {
            let main_submit_time = start.elapsed();
            println!(
                "📤 Main Submission: {:.2}ms",
                main_submit_time.as_secs_f64() * 1000.0
            );
        }
        
        if let Some(start) = main_encode_start {
            let main_encode_time = start.elapsed();
            println!(
                "🎨 Main Render Encoding: {:.2}ms",
                main_encode_time.as_secs_f64() * 1000.0
            );
        }
        
        //output.present();

        // Create filter encoder with culling results
        let filter_encode_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        
        let mut filter_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Filter Encoder"),
            });

        // Use GPU-driven indirect filter rendering (fixed implementation)
        self.rasterizer_state.encode_filter_indirect(
            &mut filter_encoder,
            camera_matrix,
        );

        // Submit the filter rendering commands and capture the submission index
        let filter_submit_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        let render_submission = self.queue.submit(std::iter::once(filter_encoder.finish()));
        
        if let Some(start) = filter_submit_start {
            let filter_submit_time = start.elapsed();
            println!(
                "📤 Filter Submission: {:.2}ms",
                filter_submit_time.as_secs_f64() * 1000.0
            );
        }
        
        if let Some(start) = filter_encode_start {
            let filter_encode_time = start.elapsed();
            println!(
                "🔍 Filter Render Encoding: {:.2}ms",
                filter_encode_time.as_secs_f64() * 1000.0
            );
        }

        if let Some(start) = encoding_start {
            let encoding_time = start.elapsed();
            println!(
                "📋 GPU Command Encoding: {:.2}ms",
                encoding_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 2: Prepare Read Operations ===
        let prepare_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };

        // Prepare both operations without polling
        let texture_op = self
            .staging_pool
            .prepare_texture_extraction(&self.queue, &self.rasterizer_state.render_target.texture);

        let filter_op = self.staging_pool.prepare_buffer_read(
            &self.queue,
            &self.rasterizer_state.filter.output_buffer,
            true,
        );

        if let Some(start) = prepare_start {
            let prepare_time = start.elapsed();
            println!(
                "⚙️  Read Operations Setup: {:.2}ms",
                prepare_time.as_secs_f64() * 1000.0
            );
        }

        // Collect submission indices before moving the operations
        let texture_submission = texture_op.submission_index.clone();
        let filter_submission = filter_op.submission_index.clone();

        // === PHASE 3: GPU Polling ===
        let polling_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };

        // Wait for all specific submissions to complete
        let mut all_submissions = vec![
            culling_submissions,
            render_submission,
            texture_submission,
            filter_submission,
        ];
        
        // Add filter buffer submission if it exists
        if let Some(filter_buffer_sub) = filter_buffer_submission {
            all_submissions.push(filter_buffer_sub);
        }
        
        self.staging_pool.wait_for_submissions(&all_submissions);

        if let Some(start) = polling_start {
            let polling_time = start.elapsed();
            println!(
                "⏳ GPU Submission Polling: {:.2}ms",
                polling_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 4: Parallel Buffer Reads ===
        let parallel_read_start = if profile_enabled { Some(std::time::Instant::now()) } else { None };

        // Execute both read operations in parallel
        let results = self
            .staging_pool
            .execute_batched_reads_parallel(vec![texture_op, filter_op])
            .await;

        if let Some(start) = parallel_read_start {
            let parallel_read_time = start.elapsed();
            println!(
                "🚀 Parallel Buffer Reads: {:.2}ms",
                parallel_read_time.as_secs_f64() * 1000.0
            );
        }

        // === PHASE 5: Data Processing ===
        let processing_start = std::time::Instant::now();

        // Extract results
        let to_insert_raw = results[0]
            .as_ref()
            .unwrap_or_else(|e| panic!("Texture extraction failed: {}", e))
            .clone();
        let filter_data = results[1]
            .as_ref()
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
        let count = u32::from_le_bytes(filter_data[0..4].try_into().unwrap()) as usize;
        // Due to std140 we must start offset at 16
        let base = 16usize;
        let output_data_slice: &[FilterCoord] = bytemuck::cast_slice(
            &filter_data[base..base + count * std::mem::size_of::<FilterCoord>()],
        );
        let to_remove = output_data_slice.to_vec();

        if profile_enabled {
            let processing_time = processing_start.elapsed();
            println!(
                "🔄 Data Processing: {:.2}ms",
                processing_time.as_secs_f64() * 1000.0
            );
        }

        if profile_enabled {
            if let Some(start) = total_start {
                let total_time = start.elapsed();
                println!(
                    "🏁 TOTAL PIPELINE: {:.2}ms",
                    total_time.as_secs_f64() * 1000.0
                );
                println!(
                    "📊 Items: {} to_insert, {} to_remove",
                    to_insert.len(),
                    to_remove.len()
                );
                println!("═══════════════════════════════════════");
            }
        }

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
    pub to_remove: Vec<FilterCoord>,
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
            world.cells().remove(&coord.coord);
        });
    }
}
