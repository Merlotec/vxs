use crate::AgentVisionRenderer;
use crate::FilterWorld;
use crate::WorldChangeset;
use crate::rasterizer::noise::NoiseParams;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use voxelsim::RendererClient;
use voxelsim::env::VoxelGrid;
use voxelsim::viewport::CameraOrientation;
use voxelsim::viewport::{CameraProjection, CameraView};

#[pymodule]
pub fn voxelsim_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FilterWorld>()?;
    m.add_class::<WorldChangeset>()?;
    m.add_class::<AgentVisionRenderer>()?;
    m.add_class::<NoiseParams>()?;
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
        orientation: CameraOrientation,
    ) -> PyResult<()> {
        self.send_pov(client, stream_idx, agent_id, proj, orientation)
            .map_err(|e| PyException::new_err(format!("Could not send pov: {}", e)))
    }
}

#[pymethods]
impl AgentVisionRenderer {
    #[new]
    pub fn init_py(world: &VoxelGrid, view_size: [u32; 2], noise: NoiseParams) -> Self {
        Self::init(world, view_size.into(), noise)
    }
    pub fn update_filter_world_py(
        &self,
        camera: CameraView,
        proj: CameraProjection,
        filter_world: FilterWorld,
        timestamp: f64,
        callback: PyObject,
    ) {
        self.update_filter_world(
            camera.view_matrix(),
            proj.projection_matrix(),
            filter_world,
            timestamp,
            move |fw| {
                Python::with_gil(|py| {
                    let _ = callback.call1(py, (fw.clone(),));
                });
            },
        )
    }
}

#[pymethods]
impl NoiseParams {
    #[staticmethod]
    pub fn default_with_seed_py(seed: [f32; 3]) -> Self {
        Self::default_with_seed(seed.into())
    }

    #[staticmethod]
    pub fn none_py() -> Self {
        Self::none()
    }
}
