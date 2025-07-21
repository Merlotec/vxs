pub mod buf;
//#[cfg(feature = "python")]
pub mod py;
//pub mod compute;
pub mod pipeline;
pub mod rasterizer;
use nalgebra::{Matrix4, Vector2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::Deref;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use voxelsim::VoxelGrid;

use pipeline::State;

use crate::{pipeline::WorldChangeset, rasterizer::camera::CameraMatrix};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::attributes::pyclass)]
pub struct AgentVisionRenderer {
    sender: SyncSender<RenderCommand>,
}

pub enum RenderCommand {
    Exit,
    Render {
        view_proj: Matrix4<f32>,
        filter_world: Arc<Mutex<VoxelGrid>>,
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

    pub fn update_world(&self, view_proj: Matrix4<f32>, filter_world: Arc<Mutex<VoxelGrid>>) {
        let rx = self.render(view_proj, filter_world.clone());
        std::thread::spawn(move || {
            if let Ok(change) = rx.recv() {
                let vgrid = filter_world.lock().unwrap();
                change.to_insert.into_par_iter().for_each(|(coord, cell)| {
                    vgrid.cells().insert(coord, cell);
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
        filter_world: Arc<Mutex<VoxelGrid>>,
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
