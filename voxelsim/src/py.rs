use nalgebra::Vector3;
use pyo3::exceptions::PyException;
use pyo3::{IntoPyObjectExt, prelude::*};
use std::collections::HashMap;
use tinyvec::ArrayVec;

use crate::{
    agent::{
        Action, Agent, AgentDynamics, CmdSequence, MoveCommand, MoveDir,
        viewport::{CameraProjection, CameraView, IntersectionMap},
    },
    env::{CollisionShell, Coord, GlobalEnv, VoxelGrid},
    network::RendererClient,
    sim::Collision,
    terrain::TerrainConfig,
};

#[pymodule]
pub fn voxelsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VoxelGrid>()?;
    m.add_class::<GlobalEnv>()?;
    m.add_class::<Agent>()?;
    m.add_class::<MoveDir>()?;
    m.add_class::<Action>()?;
    m.add_class::<MoveCommand>()?;
    m.add_class::<AgentDynamics>()?;
    m.add_class::<RendererClient>()?;
    m.add_class::<IntersectionMap>()?;
    m.add_class::<CameraProjection>()?;
    m.add_class::<Collision>()?;
    Ok(())
}

#[pymethods]
impl RendererClient {
    /// Creates a new Client.
    #[new]
    fn new_py(
        host: String,
        world_port: u16,
        agent_port: u16,
        pov_start_port: u16,
        pov_agent_start_port: u16,
    ) -> Self {
        Self::new(
            &host,
            world_port,
            agent_port,
            pov_start_port,
            pov_agent_start_port,
        )
    }

    /// Connects both world and agent ports.
    ///
    /// Raises a Python exception on error.
    fn connect_py(&mut self, pov_count: u16) -> PyResult<()> {
        self.connect(pov_count)
            .map_err(|e| PyException::new_err(e.to_string()))
    }
}

#[pymethods]
impl GlobalEnv {
    #[new]
    pub fn new(world: VoxelGrid, agents: HashMap<usize, Agent>) -> Self {
        Self {
            world,
            agents,
            povs: HashMap::new(),
        }
    }

    pub fn update_py(
        &mut self,
        py: Python,
        dynamics: &AgentDynamics,
        delta: f32,
    ) -> Vec<Collision> {
        self.update(dynamics, delta)
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

    pub fn send_pov(
        &self,
        client: &mut RendererClient,
        stream_idx: usize,
        agent_id: usize,
    ) -> PyResult<()> {
        if let Some(pov) = self.povs.get(&agent_id) {
            client.send_pov(stream_idx, pov);
            Ok(())
        } else {
            Err(PyException::new_err(format!(
                "No pov with agent_id {}",
                agent_id
            )))
        }
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
}

// --------------- PYTHON-EXPOSED IMPL -----------------------------
#[pymethods]
impl VoxelGrid {
    #[new]
    fn new_py() -> Self {
        Self::new()
    }
    /// Simple Python wrapper (no complex args).
    pub fn generate_default_terrain(&mut self, seed: u32) {
        self.generate_terrain(&TerrainConfig {
            seed,
            ..Default::default()
        });
    }

     pub fn get_cell(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        self.cells().get(&[x, y, z]).map(|cell| cell.bits())
    }


}

#[pymethods]
impl Agent {
    #[new]
    pub fn new_py(id: usize) -> Self {
        Self {
            id,
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
            thrust: Vector3::zeros(),
            action: None,
        }
    }

    /// Python-friendly method to perform a sequence of move commands
    pub fn perform_dyn_sequence(&mut self, commands: Vec<MoveCommand>) {
        let mut cmd_sequence = ArrayVec::new();

        // Convert Vec to ArrayVec, respecting the MAX_ACTIONS limit
        for (i, cmd) in commands.into_iter().enumerate() {
            if i >= crate::agent::MAX_ACTIONS {
                break;
            }
            cmd_sequence.push(cmd);
        }

        self.perform(cmd_sequence);
    }

    pub fn camera_view_py(&self) -> CameraView {
        self.camera_view()
    }
}

#[pymethods]
impl MoveCommand {
    #[new]
    pub fn new(dir: MoveDir, urgency: f32) -> Self {
        Self { dir, urgency }
    }

    /// Create a move command with default urgency (0.5)
    #[classmethod]
    pub fn with_direction(_cls: &Bound<'_, pyo3::types::PyType>, dir: MoveDir) -> Self {
        Self::new(dir, 0.5)
    }

    /// Create a move command for moving up
    #[classmethod]
    pub fn up(_cls: &Bound<'_, pyo3::types::PyType>, urgency: f32) -> Self {
        Self::new(MoveDir::Up, urgency)
    }

    /// Create a move command for moving down
    #[classmethod]
    pub fn down(_cls: &Bound<'_, pyo3::types::PyType>, urgency: f32) -> Self {
        Self::new(MoveDir::Down, urgency)
    }

    /// Create a move command for moving left
    #[classmethod]
    pub fn left(_cls: &Bound<'_, pyo3::types::PyType>, urgency: f32) -> Self {
        Self::new(MoveDir::Left, urgency)
    }

    /// Create a move command for moving right
    #[classmethod]
    pub fn right(_cls: &Bound<'_, pyo3::types::PyType>, urgency: f32) -> Self {
        Self::new(MoveDir::Right, urgency)
    }

    /// Create a move command for moving forward
    #[classmethod]
    pub fn forward(_cls: &Bound<'_, pyo3::types::PyType>, urgency: f32) -> Self {
        Self::new(MoveDir::Forward, urgency)
    }

    /// Create a move command for moving backward
    #[classmethod]
    pub fn back(_cls: &Bound<'_, pyo3::types::PyType>, urgency: f32) -> Self {
        Self::new(MoveDir::Back, urgency)
    }

    /// Get the direction
    pub fn get_direction(&self) -> MoveDir {
        self.dir
    }

    /// Get the urgency value
    pub fn get_urgency(&self) -> f32 {
        self.urgency
    }
}

#[pymethods]
impl Action {
    #[new]
    pub fn new() -> Self {
        Self {
            cmd_sequence: ArrayVec::new(),
            origin: Vector3::zeros(),
        }
    }

    /// Add a move command to the action sequence
    pub fn add_command(&mut self, command: MoveCommand) -> bool {
        if self.cmd_sequence.len() < crate::agent::MAX_ACTIONS {
            self.cmd_sequence.push(command);
            true
        } else {
            false
        }
    }

    /// Get the number of commands in the sequence
    pub fn len(&self) -> usize {
        self.cmd_sequence.len()
    }

    /// Check if the action sequence is empty
    pub fn is_empty(&self) -> bool {
        self.cmd_sequence.is_empty()
    }

    /// Clear all commands from the sequence
    pub fn clear(&mut self) {
        self.cmd_sequence.clear();
    }

    /// Get the origin position as a tuple
    pub fn get_origin(&self) -> (i32, i32, i32) {
        (self.origin.x, self.origin.y, self.origin.z)
    }

    /// Set the origin position
    pub fn set_origin(&mut self, x: i32, y: i32, z: i32) {
        self.origin = Vector3::new(x, y, z);
    }
}

#[pymethods]
impl MoveDir {
    /// Create a None/stationary direction
    #[classmethod]
    pub fn none(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::None
    }

    /// Create an Up direction
    #[classmethod]
    pub fn up(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Up
    }

    /// Create a Down direction
    #[classmethod]
    pub fn down(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Down
    }

    /// Create a Left direction
    #[classmethod]
    pub fn left(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Left
    }

    /// Create a Right direction
    #[classmethod]
    pub fn right(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Right
    }

    /// Create a Forward direction
    #[classmethod]
    pub fn forward(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Forward
    }

    /// Create a Back direction
    #[classmethod]
    pub fn back(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Back
    }

    /// Create an Undecided direction
    #[classmethod]
    pub fn undecided(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        MoveDir::Undecided
    }

    /// Get the direction as a string representation
    pub fn to_string(&self) -> String {
        match self {
            MoveDir::None => "None".to_string(),
            MoveDir::Up => "Up".to_string(),
            MoveDir::Down => "Down".to_string(),
            MoveDir::Left => "Left".to_string(),
            MoveDir::Right => "Right".to_string(),
            MoveDir::Forward => "Forward".to_string(),
            MoveDir::Back => "Back".to_string(),
            MoveDir::Undecided => "Undecided".to_string(),
        }
    }

    /// Get the direction vector as a tuple (x, y, z) or None if undecided
    pub fn get_direction_vector(&self) -> Option<(i32, i32, i32)> {
        self.dir_vec().map(|v| (v.x, v.y, v.z))
    }

    /// Check if this direction represents movement
    pub fn is_movement(&self) -> bool {
        !matches!(self, MoveDir::None | MoveDir::Undecided)
    }

    /// Check if this is a vertical movement
    pub fn is_vertical(&self) -> bool {
        matches!(self, MoveDir::Up | MoveDir::Down)
    }

    /// Check if this is a horizontal movement
    pub fn is_horizontal(&self) -> bool {
        matches!(
            self,
            MoveDir::Left | MoveDir::Right | MoveDir::Forward | MoveDir::Back
        )
    }

    /// Get the opposite direction
    pub fn opposite(&self) -> Self {
        match self {
            MoveDir::Up => MoveDir::Down,
            MoveDir::Down => MoveDir::Up,
            MoveDir::Left => MoveDir::Right,
            MoveDir::Right => MoveDir::Left,
            MoveDir::Forward => MoveDir::Back,
            MoveDir::Back => MoveDir::Forward,
            MoveDir::None => MoveDir::None,
            MoveDir::Undecided => MoveDir::Undecided,
        }
    }
}

#[pymethods]
impl IntersectionMap {
    pub fn hit_count(&self) -> usize {
        self.intersections
            .iter()
            .map(|x| if x.is_some() { 1usize } else { 0usize })
            .sum()
    }
}

#[pymethods]
impl AgentDynamics {
    #[new]
    pub fn new(
        air_resistance: f32,
        gravity: (f32, f32, f32),
        thrust: f32,
        thrust_urgency_ceiling: f32,
        bounding_box: (f32, f32, f32),
    ) -> Self {
        Self {
            air_resistance,
            g: Vector3::new(gravity.0, gravity.1, gravity.2),
            thrust,
            thrust_urgency_ceiling,
            bounding_box: Vector3::new(bounding_box.0, bounding_box.1, bounding_box.2),
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
        }
    }

    /// Create lightweight drone dynamics (faster, more agile)
    #[classmethod]
    pub fn lightweight_drone(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            air_resistance: 0.3,
            g: Vector3::new(0.0, -9.8, 0.0),
            thrust: 30.0,
            thrust_urgency_ceiling: 0.8,
            bounding_box: Vector3::new(0.3, 0.15, 0.3),
        }
    }

    /// Create heavy drone dynamics (slower, more stable)
    #[classmethod]
    pub fn heavy_drone(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            air_resistance: 0.8,
            g: Vector3::new(0.0, -9.8, 0.0),
            thrust: 20.0,
            thrust_urgency_ceiling: 0.6,
            bounding_box: Vector3::new(0.7, 0.3, 0.7),
        }
    }

    /// Create zero gravity dynamics (space-like environment)
    #[classmethod]
    pub fn zero_gravity(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            air_resistance: 0.1,
            g: Vector3::new(0.0, 0.0, 0.0),
            thrust: 15.0,
            thrust_urgency_ceiling: 1.0,
            bounding_box: Vector3::new(0.5, 0.2, 0.5),
        }
    }

    /// Create underwater dynamics (high resistance, different gravity)
    #[classmethod]
    pub fn underwater(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self {
            air_resistance: 2.0,
            g: Vector3::new(0.0, -2.0, 0.0), // Reduced effective gravity due to buoyancy
            thrust: 40.0,                    // Need more thrust to overcome water resistance
            thrust_urgency_ceiling: 0.9,
            bounding_box: Vector3::new(0.6, 0.25, 0.6),
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

    /// Create a copy with modified air resistance
    pub fn with_air_resistance(&self, air_resistance: f32) -> Self {
        Self {
            air_resistance,
            g: self.g,
            thrust: self.thrust,
            thrust_urgency_ceiling: self.thrust_urgency_ceiling,
            bounding_box: self.bounding_box,
        }
    }

    /// Create a copy with modified gravity
    pub fn with_gravity(&self, x: f32, y: f32, z: f32) -> Self {
        Self {
            air_resistance: self.air_resistance,
            g: Vector3::new(x, y, z),
            thrust: self.thrust,
            thrust_urgency_ceiling: self.thrust_urgency_ceiling,
            bounding_box: self.bounding_box,
        }
    }

    /// Create a copy with modified thrust
    pub fn with_thrust(&self, thrust: f32) -> Self {
        Self {
            air_resistance: self.air_resistance,
            g: self.g,
            thrust,
            thrust_urgency_ceiling: self.thrust_urgency_ceiling,
            bounding_box: self.bounding_box,
        }
    }

    /// Create a copy with modified bounding box
    pub fn with_bounding_box(&self, x: f32, y: f32, z: f32) -> Self {
        Self {
            air_resistance: self.air_resistance,
            g: self.g,
            thrust: self.thrust,
            thrust_urgency_ceiling: self.thrust_urgency_ceiling,
            bounding_box: Vector3::new(x, y, z),
        }
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
impl CameraProjection {
    #[new]
    pub fn new(aspect: f32, fov_vertical: f32, max_distance: f32, near_distance: f32) -> Self {
        Self {
            aspect,
            fov_vertical,
            max_distance,
            near_distance,
        }
    }
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
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
