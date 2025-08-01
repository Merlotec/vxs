use crate::rasterizer::RasterizerState;
use crate::rasterizer::camera::CameraMatrix;
use crate::rasterizer::{self, CellInstance, InstanceBuffer};
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
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn create(world: &VoxelGrid, view_size: Vector2<u32>) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU

        #[cfg(feature = "cuda-octree")]
        let (queue, device) = {
            use ash::khr::{
                external_memory, external_memory_fd, external_semaphore, external_semaphore_fd,
            };
            use wgpu::hal::api::Vulkan;
            use wgpu::{
                Backends, DeviceDescriptor, Features, Instance, InstanceFlags, Limits, MemoryHints,
                PowerPreference, RequestAdapterOptions, Trace,
            };

            // 1) Create instance & pick adapter
            let instance = Instance::new(&wgpu::InstanceDescriptor {
                backends: Backends::all(),
                flags: InstanceFlags::VALIDATION | InstanceFlags::GPU_BASED_VALIDATION,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .unwrap();

            // 2) Make sure the adapter supports push-constants & clear-texture:
            let needed = Features::PUSH_CONSTANTS | Features::CLEAR_TEXTURE;
            let supported = adapter.features();
            if !supported.contains(needed) {
                panic!(
                    "Adapter is missing required features: {:?}",
                    (needed - supported)
                );
            }

            // 3) Build your DeviceDescriptor
            let desc = DeviceDescriptor {
                label: None,
                required_features: needed,
                required_limits: Limits {
                    max_push_constant_size: 64,
                    ..Limits::default()
                },
                trace: Trace::Off,
                memory_hints: Default::default(),
            };

            // 4) Open as Vulkan, injecting external-memory-fd
            let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }.unwrap();
            let open_dev = unsafe {
                hal_adapter
                    .open_with_callback(
                        desc.required_features,
                        &MemoryHints::Performance,
                        Some(Box::new(|args| {
                            // enable the Vulkan extension for exporting FDs
                            args.extensions.push(external_memory::NAME);
                            args.extensions.push(external_memory_fd::NAME);
                            args.extensions.push(external_semaphore::NAME);
                            args.extensions.push(external_semaphore_fd::NAME);
                        })),
                    )
                    .unwrap()
            };
            println!("Created vulkan device with external_memory_fd extension.");
            // 5) Turn it back into a wgpu::Device + Queue
            let (device, queue) =
                unsafe { adapter.create_device_from_hal(open_dev, &desc).unwrap() };
            (queue, device)
        };

        #[cfg(not(feature = "cuda-octree"))]
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
                        max_push_constant_size: 64, // Size of mat4x4<f32>
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

        #[cfg(feature = "cuda-octree")]
        let sem_fd = {
            use ash::vk;

            use wgpu::wgc::api::Vulkan; // Main State struct to hold all wgpu-related objects
            let hal_queue_guard = unsafe { self.queue.as_hal::<Vulkan>() }
                .expect("Must be running on the Vulkan backend to extract cuda texture!"); // 
            let hal_queue: &wgpu_hal::vulkan::Queue = &*hal_queue_guard;

            // Create an exportable (opaque-FD) VkSemaphore
            let mut export_info = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
            let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
            let vk_semaphore = unsafe {
                hal_queue
                    .raw_device()
                    .create_semaphore(&sem_info, None)
                    .expect("Unable to create semaphore!")
            }; // 

            // Instruct wgpu-hal to signal that semaphore at queue submit
            hal_queue.add_signal_semaphore(vk_semaphore, None);
            println!("Created vulkan semaphore!");
            vk_semaphore
        };

        self.queue.submit(std::iter::once(encoder.finish()));

        let to_insert = rasterizer::texture::extract_texture_data(
            &self.device,
            &self.queue,
            &self.rasterizer_state.render_target.texture,
        )
        .await
        .unwrap();

        // Try sending the texture to cuda
        #[cfg(feature = "cuda-octree")]
        unsafe {
            use ash::vk;
            use ash::vk::Handle;
            use std::ffi::c_int;

            use wgpu::wgc::api::Vulkan; // Main State struct to hold all wgpu-related objects

            use std::ops::Deref;

            let tex_fd = self
                .rasterizer_state
                .render_target
                .texture
                .as_hal::<Vulkan>()
                .expect("Must be using a vulkan backend to extract cuda texture!");

            // let vk_device_memory: vk::DeviceMemory = *tex_fd.external_memory().unwrap();

            let vk_device_memory: vk::DeviceMemory = if let Some(raw_block) = tex_fd.raw_block() {
                *raw_block.memory()
            } else if let Some(mem) = tex_fd.external_memory() {
                mem
            } else {
                panic!("Texture does not have any memory bound to it!")
            };

            let dev = self.device.as_hal::<Vulkan>().unwrap();
            let hal_device = dev.raw_device();
            let hal_instance: &ash::Instance = dev.shared_instance().raw_instance();

            let ex_mem_dev = ash::khr::external_memory_fd::Device::new(hal_instance, hal_device);

            // Build your get-FD info
            let get_fd_info = vk::MemoryGetFdInfoKHR::default()
                .memory(vk_device_memory) // the VkDeviceMemory you obtained from your patched HAL
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD_KHR);

            // Now call the method you saw in `crate::khr::external_memory_fd::Device`:
            let mem_fd = unsafe {
                ex_mem_dev
                    .get_memory_fd(&get_fd_info)
                    .expect("Could not get external memory.")
            };

            let mut props = vk::MemoryFdPropertiesKHR::default();
            ex_mem_dev.get_memory_fd_properties_khr(
                vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD_KHR,
                mem_fd,
                &mut props,
            );

            println!("memory info: {:?}", &props);

            // 4 i32s per texel.
            println!("memory handle: {}", mem_fd);
            let res = octree_gpu::test_vk_texture(
                mem_fd,
                sem_fd.as_raw() as c_int,
                self.size.x as c_int,
                self.size.y as c_int,
            );
            println!("Successfully uploaded cuda texture!");
        }
        let to_remove = self
            .rasterizer_state
            .filter
            .get_filter_list(&self.device, &self.queue)
            .await;

        self.rasterizer_state
            .filter
            .clear_buffers(&self.device, &self.queue)
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
