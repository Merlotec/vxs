use voxelsim::{Action, Agent, VoxelGrid};

pub mod manual;
pub mod python;
pub use python::PythonBackend;

pub trait ControlBackend {
    fn update_action(&mut self, agent: &Agent, fw: &VoxelGrid) -> Option<Action>;
}
