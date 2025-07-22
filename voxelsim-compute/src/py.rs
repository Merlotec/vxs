use crate::AgentVisionRenderer;
use crate::FilterWorld;
use crate::WorldChangeset;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use voxelsim::RendererClient;
use voxelsim::env::VoxelGrid;
use voxelsim::viewport::{CameraProjection, CameraView};

#[pymodule]
pub fn voxelsim_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FilterWorld>()?;
    m.add_class::<WorldChangeset>()?;
    m.add_class::<AgentVisionRenderer>()?;
    Ok(())
}

#[pymethods]
impl FilterWorld {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_updating_py(&self, timestamp: f64) -> bool {
        self.is_updating(timestamp)
    }

    pub fn send_pov_py(
        &self,
        client: &mut RendererClient,
        stream_idx: usize,
        agent_id: usize,
        proj: CameraProjection,
    ) -> PyResult<()> {
        self.send_pov(client, stream_idx, agent_id, proj)
            .map_err(|e| PyException::new_err(format!("Could not send pov: {}", e)))
    }
}

#[pymethods]
impl AgentVisionRenderer {
    #[new]
    pub fn init_py(world: &VoxelGrid, view_size: [u32; 2]) -> Self {
        Self::init(world, view_size.into())
    }
    pub fn update_filter_world_py(
        &self,
        camera: CameraView,
        proj: CameraProjection,
        filter_world: FilterWorld,
        timestamp: f64,
    ) {
        let view_proj = proj.projection_matrix() * camera.view_matrix();
        self.update_filter_world(view_proj, filter_world, timestamp)
    }
}
