use nalgebra::Vector3;
use pyo3::exceptions::PyException;
use pyo3::{IntoPyObjectExt, prelude::*};
use std::collections::HashMap;
use tinyvec::ArrayVec;
use voxelsim::{Agent, Coord, MoveCommand, RendererClient, VoxelGrid};

use crate::dynamics::EnvState;
use crate::sim::GlobalEnv;
use crate::{
    dynamics::standard::StandardDynamics,
    sim::Collision,
    terrain::{TerrainConfig, TerrainGenerator},
};

#[pymodule]
pub fn voxelsim_simulator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GlobalEnv>()?;
    m.add_class::<Collision>()?;
    m.add_class::<StandardDynamics>()?;
    m.add_class::<TerrainGenerator>()?;
    m.add_class::<EnvState>()?;
    Ok(())
}

#[pymethods]
impl GlobalEnv {
    #[new]
    pub fn new(world: VoxelGrid, agents: HashMap<usize, Agent>) -> Self {
        Self { world, agents }
    }

    pub fn update_py(
        &mut self,
        py: Python,
        dynamics: &StandardDynamics,
        env: &EnvState,
        delta: f32,
    ) -> Vec<Collision> {
        self.update(dynamics, env, delta)
    }

    pub fn send_agents(&self, client: &mut RendererClient) -> PyResult<()> {
        client
            .send_agents(&self.agents)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    pub fn send_world(&self, client: &mut RendererClient) -> PyResult<()> {
        client
            .send_world(&self.world)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    pub fn perform_sequence_on_agent(
        &mut self,
        agent_id: usize,
        cmds: Vec<MoveCommand>,
    ) -> PyResult<()> {
        self.agent_mut(&agent_id)
            .ok_or_else(|| PyException::new_err(format!("no agent with id: {}", agent_id)))?
            .perform_sequence(cmds);
        Ok(())
    }

    pub fn get_agent(&self, agent_id: usize) -> PyResult<Agent> {
        Ok(self
            .agents
            .get(&agent_id)
            .ok_or(PyException::new_err("Invalid agent id!"))?
            .clone())
    }

    pub fn get_agent_pos(&self, agent_id: usize) -> Option<[f32; 3]> {
        self.agents.get(&agent_id).map(|x| x.pos.into())
    }

    pub fn set_agent_pos(&mut self, agent_id: usize, pos: [f32; 3]) -> PyResult<()> {
        self.agents
            .get_mut(&agent_id)
            .ok_or(PyException::new_err("Invalid agent id!"))?
            .pos = pos.into();
        Ok(())
    }

    /// Simple Python wrapper (no complex args).
    #[staticmethod]
    pub fn new_default_terrain(seed: u32, agents: HashMap<usize, Agent>) -> Self {
        let mut world = TerrainGenerator::new();
        world.generate_terrain(&TerrainConfig {
            seed,
            ..Default::default()
        });
        Self {
            world: world.generate_world(),
            agents,
        }
    }
}
#[pymethods]
impl StandardDynamics {
    #[new]
    pub fn new(
        air_resistance: f32,
        gravity: (f32, f32, f32),
        thrust: f32,
        thrust_urgency_ceiling: f32,
        bounding_box: (f32, f32, f32),
        jerk_coeff: f32,
        yaw_rate: f32,
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
    pub fn get_air_resistance(&self) -> f32 {
        self.air_resistance
    }

    pub fn get_gravity(&self) -> (f32, f32, f32) {
        (self.g.x, self.g.y, self.g.z)
    }

    pub fn get_thrust(&self) -> f32 {
        self.thrust
    }

    pub fn get_thrust_urgency_ceiling(&self) -> f32 {
        self.thrust_urgency_ceiling
    }

    pub fn get_bounding_box(&self) -> (f32, f32, f32) {
        (
            self.bounding_box.x,
            self.bounding_box.y,
            self.bounding_box.z,
        )
    }

    // Setters for all properties
    pub fn set_air_resistance(&mut self, value: f32) {
        self.air_resistance = value;
    }

    pub fn set_gravity(&mut self, x: f32, y: f32, z: f32) {
        self.g = Vector3::new(x, y, z);
    }

    pub fn set_thrust(&mut self, value: f32) {
        self.thrust = value;
    }

    pub fn set_thrust_urgency_ceiling(&mut self, value: f32) {
        self.thrust_urgency_ceiling = value;
    }

    pub fn set_bounding_box(&mut self, x: f32, y: f32, z: f32) {
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
    pub fn gravity_magnitude(&self) -> f32 {
        self.g.norm()
    }

    /// Get the bounding box volume
    pub fn bounding_box_volume(&self) -> f32 {
        self.bounding_box.x * self.bounding_box.y * self.bounding_box.z
    }
}
#[pymethods]
impl Collision {
    pub fn agent_id(&self) -> usize {
        self.agent_id
    }

    pub fn shell_coords(&self) -> Vec<Coord> {
        self.shell.iter().map(|(coord, _)| *coord).collect()
    }
}
