use crate::AgentVisionRenderer;
use crate::WorldChangeset;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use voxelsim::viewport::{CameraProjection, CameraView, VirtualGrid};

use pyo3::prelude::*;

#[pymodule]
pub fn voxelsim_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FilterWorld>()?;
    m.add_class::<WorldChangeset>()?;
    m.add_class::<AgentVisionRenderer>()?;
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FilterWorld {
    filter_world: Arc<Mutex<VirtualGrid>>,
}

#[pymethods]
impl FilterWorld {
    #[new]
    pub fn new() -> Self {
        Self {
            filter_world: Arc::new(Mutex::new(VirtualGrid::with_capacity(10000))),
        }
    }

    pub fn update(&self, changeset: WorldChangeset) {
        changeset.update_world(self.filter_world.lock().unwrap().deref_mut())
    }
}

#[pymethods]
impl AgentVisionRenderer {
    pub fn update_filter_world(
        &self,
        camera: CameraView,
        proj: CameraProjection,
        filter_world: &FilterWorld,
    ) {
        self.update_world(
            proj.projection_matrix() * camera.view_matrix(),
            filter_world.filter_world.clone(),
        );
    }
}
