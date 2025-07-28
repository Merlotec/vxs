use std::collections::HashMap;

use dashmap::DashMap;
use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelDrainFull, ParallelIterator};

use crate::{
    Cell, Coord,
    agent::{
        Action, Agent, MoveCommand, MoveDir,
        viewport::{CameraProjection, CameraView},
    },
    chase::{ChaseTarget, FixedLookaheadChaser, TrajectoryChaser},
    env::VoxelGrid,
    network::RendererClient,
    viewport::CameraOrientation,
};

#[pymodule]
pub fn voxelsim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VoxelGrid>()?;
    m.add_class::<Agent>()?;
    m.add_class::<MoveDir>()?;
    m.add_class::<Action>()?;
    m.add_class::<MoveCommand>()?;
    m.add_class::<RendererClient>()?;
    m.add_class::<CameraProjection>()?;
    m.add_class::<FixedLookaheadChaser>()?;
    m.add_class::<ChaseTarget>()?;
    m.add_class::<CameraOrientation>()?;
    Ok(())
}

pub type PyCoord = [i32; 3];

#[pymethods]
impl VoxelGrid {
    #[staticmethod]
    pub fn from_dict_py(dict: HashMap<PyCoord, Cell>) -> Self {
        let cells: DashMap<Coord, Cell> = DashMap::with_capacity(dict.len());

        dict.par_iter().for_each(|(coord, value)| {
            cells.insert((*coord).into(), *value);
        });

        Self::from_cells(cells)
    }
}

#[pymethods]
impl Cell {
    #[staticmethod]
    pub fn filled() -> Self {
        Self::FILLED
    }

    #[staticmethod]
    pub fn sparse() -> Self {
        Self::SPARSE
    }
}

#[pymethods]
impl Agent {
    #[new]
    pub fn new_py(id: usize) -> Self {
        Self::new(id)
    }

    /// Python-friendly method to perform a sequence of move commands
    pub fn perform_sequence_py(&mut self, commands: Vec<MoveCommand>) -> PyResult<()> {
        self.perform_sequence(commands)
            .map_err(|e| PyException::new_err(format!("Could not perform sequence: {}", e)))
    }

    pub fn camera_view_py(&self, orientation: &CameraOrientation) -> CameraView {
        self.camera_view(orientation)
    }

    pub fn get_action(&self) -> Option<Action> {
        self.action.clone()
    }

    pub fn set_pos(&mut self, pos: [f64; 3]) {
        self.pos = Vector3::from(pos)
    }

    pub fn get_pos(&self) -> [f64; 3] {
        self.pos.into()
    }
}

#[pymethods]
impl MoveCommand {
    #[new]
    pub fn new(dir: MoveDir, urgency: f64, yaw_delta: f64) -> Self {
        Self {
            dir,
            urgency,
            yaw_delta,
        }
    }
    /// Get the direction
    pub fn get_direction(&self) -> MoveDir {
        self.dir
    }

    /// Get the urgency value
    pub fn get_urgency(&self) -> f64 {
        self.urgency
    }
}

#[pymethods]
impl Action {
    /// Add a move command to the action sequence
    // pub fn add_command(&mut self, command: MoveCommand) -> bool {
    //     if self.cmd_sequence.len() < crate::agent::MAX_ACTIONS {
    //         self.cmd_sequence.push(command);
    //         true
    //     } else {
    //         false
    //     }
    // }

    pub fn get_commands(&self) -> Vec<MoveCommand> {
        self.cmd_sequence.clone().to_vec()
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
        self.dir_vector().map(|v| (v.x, v.y, v.z))
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
impl RendererClient {
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

    fn connect_py(&mut self, pov_count: u16) -> PyResult<()> {
        self.connect(pov_count)
            .map_err(|e| PyException::new_err(e.to_string()))
    }
    pub fn send_agents_py(&mut self, agents: HashMap<usize, Agent>) -> PyResult<()> {
        self.send_agents(&agents)
            .map_err(|e| PyException::new_err(format!("Failed to send agents: {}", e)))
    }
    pub fn send_world_py(&mut self, world: &VoxelGrid) -> PyResult<()> {
        self.send_world(world)
            .map_err(|e| PyException::new_err(format!("Failed to send world: {}", e)))
    }
}

#[pymethods]
impl CameraProjection {
    #[new]
    pub fn new(aspect: f64, fov_vertical: f64, max_distance: f64, near_distance: f64) -> Self {
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
impl FixedLookaheadChaser {
    #[staticmethod]
    pub fn default_py() -> Self {
        Self::default()
    }

    pub fn step_chase_py(&self, agent: &Agent, delta: f64) -> ChaseTarget {
        self.step_chase(agent, delta)
    }
}

#[pymethods]
impl CameraOrientation {
    #[staticmethod]
    pub fn vertical_tilt_py(tilt: f64) -> Self {
        Self::vertical_tilt(tilt)
    }
}
