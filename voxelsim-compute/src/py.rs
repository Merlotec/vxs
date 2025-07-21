use crate::AgentVisionRenderer;
use std::sync::{Arc, Mutex};
use voxelsim::{Agent, VoxelGrid};
#[derive(Debug, Clone)]
#[pyclass]
pub struct FilterWorld {
    filter_world: Arc<Mutex<VoxelGrid>>,
}

#[pymethods]
impl AgentVisionRenderer {
    pub fn update_py(&self, agent: &Agent, filter_world: &FilterWorld) {
        self.update_world(
            agent.camera().view_proj(),
            filter_world.filter_world.clone(),
        );
    }
}
