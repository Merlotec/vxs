use std::{
    ops::DerefMut,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
    time::SystemTime,
};
use voxelsim::{
    agent::Agent,
    env::VoxelGrid,
    viewport::{CameraProjection, CameraView, VirtualCell, VirtualGrid},
};

use nalgebra::{Matrix4, Vector2};

use pyo3::prelude::*;

// Define a dummy struct here since it sits behind an opaque pointer.
#[repr(transparent)]
pub struct InnerState;

unsafe extern "C" {
    fn render_borrowed(
        camera_view_proj: Matrix4<f32>,
        filter_world: *mut VirtualGrid,
        render_state: *mut InnerState,
    );

    fn render_shared(
        camera_view_proj: Matrix4<f32>,
        filter_world: Arc<Mutex<VirtualGrid>>,
        render_state: *mut InnerState,
    );
    fn create_renderer(world: *const VoxelGrid, view_size: Vector2<u32>) -> *mut InnerState;

    fn destroy_renderer(state: *mut InnerState);
}

pub struct State(*mut InnerState);

// Implement for now because it saves us from having to copy the world between threads.
unsafe impl Send for State {}
unsafe impl Sync for State {}

impl State {
    pub fn create(world: &VoxelGrid, view_size: Vector2<u32>) -> Self {
        Self(unsafe { create_renderer(world as *const VoxelGrid, view_size) })
    }

    pub fn inner(&self) -> *const InnerState {
        self.0
    }

    pub fn inner_mut(&mut self) -> *mut InnerState {
        self.0
    }

    pub fn render_shared(
        &mut self,
        camera_view_proj: Matrix4<f32>,
        filter_world: Arc<Mutex<VirtualGrid>>,
    ) {
        unsafe { render_shared(camera_view_proj, filter_world, self.inner_mut()) }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        unsafe { destroy_renderer(self.0) }
    }
}

#[pymodule]
pub fn voxelsim_compute_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FilterWorld>()?;
    m.add_class::<AgentVisionRenderer>()?;
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FilterWorld {
    filter_world: Arc<Mutex<VirtualGrid>>,
    // The timestamp of the current filter world.
    timestamp: Arc<Mutex<SystemTime>>,
}

#[pymethods]
impl FilterWorld {
    #[new]
    pub fn new() -> Self {
        Self {
            filter_world: Arc::new(Mutex::new(VirtualGrid::with_capacity(10000))),
            timestamp: Arc::new(Mutex::new(SystemTime::now())),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AgentVisionRenderer {
    sender: SyncSender<RenderCommand>,
}

pub enum RenderCommand {
    Exit,
    Render {
        view_proj: Matrix4<f32>,
        filter_world: FilterWorld,
    },
}

#[pymethods]
impl AgentVisionRenderer {
    #[new]
    pub fn init(world: &VoxelGrid, view_size: [u32; 2]) -> Self {
        let (tx, rx) = mpsc::sync_channel(1000);

        let state = State::create(world, view_size.into());

        std::thread::spawn(move || Self::start_render_thread(state, rx));
        Self { sender: tx }
    }

    pub fn render(&self, camera: CameraView, proj: CameraProjection, filter_world: FilterWorld) {
        let view_proj = proj.projection_matrix() * camera.view_matrix();
        self.sender
            .send(RenderCommand::Render {
                view_proj,
                filter_world,
            })
            .unwrap()
    }
}

impl AgentVisionRenderer {
    fn start_render_thread(mut state: State, receiver: Receiver<RenderCommand>) {
        while let Ok(rc) = receiver.recv() {
            match rc {
                RenderCommand::Exit => break,
                RenderCommand::Render {
                    view_proj,
                    filter_world,
                } => {
                    state.render_shared(view_proj, filter_world.filter_world.clone());
                    *filter_world.timestamp.lock().unwrap().deref_mut() = SystemTime::now();
                }
            }
        }
    }
}
