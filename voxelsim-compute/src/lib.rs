pub mod buf;
//pub mod compute;
pub mod pipeline;
pub mod rasterizer;

#[cfg(feature = "python")]
pub mod py;

use nalgebra::{Matrix4, Vector2};
use pipeline::State;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::Deref;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use voxelsim::VoxelGrid;
use voxelsim::viewport::{VirtualCell, VirtualGrid};

use crate::{pipeline::WorldChangeset, rasterizer::camera::CameraMatrix};

#[cfg(feature = "clib")]
mod ffi {
    use crate::CameraMatrix;
    use crate::pipeline::State;
    use nalgebra::{Matrix4, Vector2};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use std::ops::Deref;
    use std::sync::{Arc, Mutex};
    use voxelsim::agent::viewport::{VirtualCell, VirtualGrid};
    use voxelsim::env::VoxelGrid;

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn render_borrowed(
        camera_view_proj: Matrix4<f32>,
        filter_world: *mut VirtualGrid,
        render_state: *mut State,
    ) {
        let render_state: &mut State = std::mem::transmute(render_state);
        let filter_world: &mut VirtualGrid = std::mem::transmute(filter_world);
        let camera_matrix = CameraMatrix::from_view_proj(camera_view_proj);
        let change =
            futures::executor::block_on(render_state.run(&camera_matrix, &filter_world)).unwrap();

        change.to_insert.into_par_iter().for_each(|(coord, cell)| {
            filter_world.cells().insert(
                coord,
                VirtualCell {
                    cell,
                    uncertainty: 0.0,
                },
            );
        });
        change.to_remove.into_par_iter().for_each(|coord| {
            filter_world.cells().remove(&coord);
        });
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn render_shared(
        camera_view_proj: Matrix4<f32>,
        filter_world: Arc<Mutex<VirtualGrid>>,
        render_state: *mut State,
    ) {
        let render_state: &mut State = std::mem::transmute(render_state);
        let camera_matrix = CameraMatrix::from_view_proj(camera_view_proj);
        let change = futures::executor::block_on(
            render_state.run(&camera_matrix, filter_world.lock().unwrap().deref()),
        )
        .unwrap();

        let filter_world = filter_world.lock().unwrap();
        change.to_insert.into_par_iter().for_each(|(coord, cell)| {
            filter_world.cells().insert(
                coord,
                VirtualCell {
                    cell,
                    uncertainty: 0.0,
                },
            );
        });
        change.to_remove.into_par_iter().for_each(|coord| {
            filter_world.cells().remove(&coord);
        });
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn create_renderer(
        world: *const VoxelGrid,
        view_size: Vector2<u32>,
    ) -> *mut State {
        let world: &VoxelGrid = std::mem::transmute(world);
        let state = Box::new(futures::executor::block_on(State::create(world, view_size)));
        Box::into_raw(state)
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn destroy_renderer(state: *mut State) {
        let _state = Box::from_raw(state);
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AgentVisionRenderer {
    sender: SyncSender<RenderCommand>,
}

pub enum RenderCommand {
    Exit,
    Render {
        view_proj: Matrix4<f32>,
        filter_world: Arc<Mutex<VirtualGrid>>,
        sender: SyncSender<WorldChangeset>,
    },
}

impl AgentVisionRenderer {
    pub fn init(world: &VoxelGrid, view_size: Vector2<u32>) -> Self {
        let (tx, rx) = mpsc::sync_channel(1000);

        let state = futures::executor::block_on(State::create(world, view_size));
        std::thread::spawn(move || {
            futures::executor::block_on(Self::start_render_thread(state, rx))
        });
        Self { sender: tx }
    }

    async fn start_render_thread(mut state: State, receiver: Receiver<RenderCommand>) {
        while let Ok(rc) = receiver.recv() {
            match rc {
                RenderCommand::Exit => break,
                RenderCommand::Render {
                    view_proj,
                    filter_world,
                    sender,
                } => {
                    let camera_matrix = CameraMatrix::from_view_proj(view_proj);
                    if let Ok(changeset) = state
                        .run(&camera_matrix, filter_world.lock().unwrap().deref())
                        .await
                    {
                        sender.send(changeset).unwrap();
                    }
                }
            }
        }
    }

    pub fn update_world(&self, view_proj: Matrix4<f32>, filter_world: Arc<Mutex<VirtualGrid>>) {
        let rx = self.render(view_proj, filter_world.clone());
        std::thread::spawn(move || {
            if let Ok(change) = rx.recv() {
                let vgrid = filter_world.lock().unwrap();
                change.to_insert.into_par_iter().for_each(|(coord, cell)| {
                    vgrid.cells().insert(
                        coord,
                        VirtualCell {
                            cell,
                            uncertainty: 0.0,
                        },
                    );
                });
                change.to_remove.into_par_iter().for_each(|coord| {
                    vgrid.cells().remove(&coord);
                });
            }
        });
    }

    pub fn render(
        &self,
        view_proj: Matrix4<f32>,
        filter_world: Arc<Mutex<VirtualGrid>>,
    ) -> Receiver<WorldChangeset> {
        let (tx, rx) = mpsc::sync_channel(1000);
        self.sender
            .send(RenderCommand::Render {
                view_proj,
                filter_world,
                sender: tx,
            })
            .unwrap();
        rx
    }
}
