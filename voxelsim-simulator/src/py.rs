use nalgebra::Vector3;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::collections::HashMap;
use voxelsim::chase::ChaseTarget;
use voxelsim::py::PyCoord;
use voxelsim::{Agent, Coord, MoveCommand, RendererClient, VoxelGrid};

use crate::dynamics::peng::PengQuadDynamics;
use crate::dynamics::{AgentDynamics, EnvState};
use crate::{
    dynamics::standard::StandardDynamics,
    sim::Collision,
    terrain::{TerrainConfig, TerrainGenerator},
};

#[pymodule]
pub fn voxelsim_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Collision>()?;
    m.add_class::<StandardDynamics>()?;
    m.add_class::<PengQuadDynamics>()?;
    m.add_class::<TerrainGenerator>()?;
    m.add_class::<TerrainConfig>()?;
    m.add_class::<EnvState>()?;
    Ok(())
}

#[pymethods]
impl PengQuadDynamics {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
    pub fn update_agent_dynamics_py(
        &mut self,
        agent: &mut Agent,
        env: &EnvState,
        chaser: &ChaseTarget,
        delta: f64,
    ) {
        self.update_agent_dynamics(agent, env, chaser, delta);
    }
}

#[pymethods]
impl StandardDynamics {
    #[new]
    pub fn new(
        air_resistance: f64,
        gravity: (f64, f64, f64),
        thrust: f64,
        thrust_urgency_ceiling: f64,
        bounding_box: (f64, f64, f64),
        jerk_coeff: f64,
        yaw_rate: f64,
    ) -> Self {
        Self {
            air_resistance,
            g: Vector3::new(gravity.0, gravity.1, gravity.2),
            thrust,
            thrust_urgency_ceiling,
            bounding_box: Vector3::new(bounding_box.0, bounding_box.1, bounding_box.2),
            jerk_coeff,
            yaw_rate,
        }
    }

    /// Create default drone dynamics (commonly used settings)
    #[classmethod]
    pub fn default_drone(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            air_resistance: 0.5,
            g: Vector3::new(0.0, -9.8, 0.0),
            thrust: 25.0,
            thrust_urgency_ceiling: 0.7,
            bounding_box: Vector3::new(0.5, 0.2, 0.5),
            jerk_coeff: 0.4,
            yaw_rate: 4.6,
        }
    }

    // Getters for all properties
    pub fn get_air_resistance(&self) -> f64 {
        self.air_resistance
    }

    pub fn get_gravity(&self) -> (f64, f64, f64) {
        (self.g.x, self.g.y, self.g.z)
    }

    pub fn get_thrust(&self) -> f64 {
        self.thrust
    }

    pub fn get_thrust_urgency_ceiling(&self) -> f64 {
        self.thrust_urgency_ceiling
    }

    pub fn get_bounding_box(&self) -> (f64, f64, f64) {
        (
            self.bounding_box.x,
            self.bounding_box.y,
            self.bounding_box.z,
        )
    }

    // Setters for all properties
    pub fn set_air_resistance(&mut self, value: f64) {
        self.air_resistance = value;
    }

    pub fn set_gravity(&mut self, x: f64, y: f64, z: f64) {
        self.g = Vector3::new(x, y, z);
    }

    pub fn set_thrust(&mut self, value: f64) {
        self.thrust = value;
    }

    pub fn set_thrust_urgency_ceiling(&mut self, value: f64) {
        self.thrust_urgency_ceiling = value;
    }

    pub fn set_bounding_box(&mut self, x: f64, y: f64, z: f64) {
        self.bounding_box = Vector3::new(x, y, z);
    }

    /// Get a string representation of the dynamics
    pub fn to_string(&self) -> String {
        format!(
            "AgentDynamics(air_resistance={:.2}, gravity=({:.1}, {:.1}, {:.1}), thrust={:.1}, thrust_urgency_ceiling={:.2}, bounding_box=({:.2}, {:.2}, {:.2}))",
            self.air_resistance,
            self.g.x,
            self.g.y,
            self.g.z,
            self.thrust,
            self.thrust_urgency_ceiling,
            self.bounding_box.x,
            self.bounding_box.y,
            self.bounding_box.z
        )
    }

    /// Check if this configuration represents a fast/agile drone
    pub fn is_agile(&self) -> bool {
        self.air_resistance < 0.4 && self.thrust > 25.0
    }

    /// Check if this configuration represents a stable/heavy drone
    pub fn is_stable(&self) -> bool {
        self.air_resistance > 0.6 && self.thrust < 25.0
    }

    /// Get the effective gravity magnitude
    pub fn gravity_magnitude(&self) -> f64 {
        self.g.norm()
    }

    /// Get the bounding box volume
    pub fn bounding_box_volume(&self) -> f64 {
        self.bounding_box.x * self.bounding_box.y * self.bounding_box.z
    }
}
#[pymethods]
impl Collision {
    pub fn agent_id(&self) -> usize {
        self.agent_id
    }

    pub fn shell_coords(&self) -> Vec<PyCoord> {
        self.shell
            .iter()
            .map(|(coord, _)| (*coord).into())
            .collect()
    }
}

#[pymethods]
impl EnvState {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
}

#[pymethods]
impl TerrainConfig {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }
}

#[pymethods]
impl TerrainGenerator {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    pub fn generate_terrain_py(&mut self, cfg: &TerrainConfig) {
        self.generate_terrain(cfg);
    }
    pub fn generate_world_py(&self) -> VoxelGrid {
        self.clone().generate_world()
    }
}
