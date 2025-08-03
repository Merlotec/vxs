use voxelsim::env::CollisionShell;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Collision {
    pub agent_id: usize,
    pub shell: CollisionShell,
}
