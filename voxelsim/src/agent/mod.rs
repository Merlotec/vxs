use std::{collections::HashSet, error::Error, fmt::Display};

use dashmap::DashSet;
use nalgebra::{Unit, Vector3};
use serde::{Deserialize, Serialize};
use tinyvec::ArrayVec;

use crate::{Coord, trajectory::Trajectory, viewport::CameraView};

pub mod chase;
pub mod trajectory;
pub mod viewport;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Agent {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,

    // Thrust is specified in units of ACCELERATION!
    // It is also assumed that if there is no change from previous action then the thrust will
    // remain the same.
    pub thrust: Vector3<f64>,

    // Given in units of radians around the thrust axis.
    pub yaw: f64,

    pub id: usize,

    // Current action being processed by the agent.
    pub action: Option<Action>,
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[repr(i32)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub enum MoveDir {
    #[default]
    None = 0,
    Forward = 1,
    Back = 2,
    Left = 3,
    Right = 4,
    Up = 5,
    Down = 6,
    Undecided = 7,
}

impl MoveDir {
    pub fn dir_vector(&self) -> Option<Coord> {
        match self {
            MoveDir::None => Some(Vector3::zeros()),
            MoveDir::Up => Some(Vector3::new(0, 0, 1)),
            MoveDir::Down => Some(Vector3::new(0, 0, -1)),
            MoveDir::Left => Some(Vector3::new(-1, 0, 0)),
            MoveDir::Right => Some(Vector3::new(1, 0, 0)),
            MoveDir::Forward => Some(Vector3::new(0, 1, 0)),
            MoveDir::Back => Some(Vector3::new(0, -1, 0)),
            MoveDir::Undecided => None,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct MoveCommand {
    pub dir: MoveDir,
    pub urgency: f64,
    pub yaw_delta: f64,
}

pub type CmdSequence = ArrayVec<[MoveCommand; MAX_ACTIONS]>;

/// When we carry out an action we need to pyo3ensure that we remove subsequent steps when we enter the
/// relevant voxels.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Action {
    pub cmd_sequence: CmdSequence,
    pub trajectory: Trajectory,
    pub origin: Vector3<i32>,
}

impl Action {
    // Gets the relative position of the centroid of the grid at position t (unscaled) from the
    // centroid of the starting cell.
    pub fn relative_centroid_pos(&self, t: usize) -> Option<Coord> {
        if self.cmd_sequence.len() >= t {
            Some(
                self.cmd_sequence
                    .iter()
                    .take(t)
                    .filter_map(|x| x.dir.dir_vector())
                    .sum(),
            )
        } else {
            None
        }
    }

    pub fn centroids(
        cmd_sequence: &[MoveCommand],
        origin: Vector3<i32>,
    ) -> Result<Vec<(Coord, f64, f64)>, ActionError> {
        let mut total: Coord = origin;
        let mut centroids = Vec::new();
        let set: DashSet<Coord> = DashSet::new();
        // Make sure we dont duplicate origin either.
        set.insert(origin);
        for seq in cmd_sequence.iter() {
            if let Some(dir_vector) = seq.dir.dir_vector() {
                let centroid = total + dir_vector;
                centroids.push((centroid, seq.urgency, seq.yaw_delta));
                total = centroid;
                if !set.insert(centroid) {
                    return Err(ActionError::DuplicateCentroid);
                }
            }
        }

        Ok(centroids)
    }
}

pub const MAX_ACTIONS: usize = 6;

impl Agent {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
            thrust: Vector3::zeros(),
            yaw: 0.0,
            action: None,
        }
    }

    // TODO: fix to use yaw.
    pub fn camera_view(&self) -> CameraView {
        let forward_h = self.vel.normalize();
        let orthogonal = (forward_h + Vector3::new(0.0, 0.0, -0.4)).normalize();

        // // Axis to pitch about = camera-right (world-up × forward).
        let world_up = Vector3::z_axis(); // Y is up
        let right = Unit::new_normalize(world_up.cross(&forward_h));

        // Rotate the horizontal-only forward vector downward by `pitch`.
        let forward = orthogonal;
        // Re-derive an exact orthonormal basis.
        let camera_right = right.into_inner(); // already unit
        let camera_up = camera_right.cross(&forward).normalize();

        CameraView {
            camera_pos: self.pos,    // assumed Point3<f64>
            camera_forward: forward, // Vector3<f64>
            camera_up,               // Vector3<f64>
            camera_right,            // Vector3<f64>
        }
    }

    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.pos = Vector3::new(x, y, z);
    }

    pub fn get_position(&self) -> (f64, f64, f64) {
        (self.pos.x, self.pos.y, self.pos.z)
    }

    pub fn set_velocity(&mut self, x: f64, y: f64, z: f64) {
        self.vel = Vector3::new(x, y, z);
    }

    pub fn perform_sequence(&mut self, commands: Vec<MoveCommand>) -> Result<(), ActionError> {
        let mut cmd_sequence = ArrayVec::new();
        for cmd in commands.into_iter().take(MAX_ACTIONS) {
            cmd_sequence.push(cmd);
        }
        self.perform(cmd_sequence)
    }

    pub fn perform(&mut self, cmd_sequence: CmdSequence) -> Result<(), ActionError> {
        if cmd_sequence.is_empty() {
            return Err(ActionError::NoCommands);
        }
        let origin = self.pos.map(|e| e.round() as i32);
        let cells = Action::centroids(&cmd_sequence, origin)?;
        let action = Action {
            cmd_sequence,
            origin,
            trajectory: Trajectory::generate(origin, &cells),
        };
        self.action = Some(action);
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ActionError {
    DuplicateCentroid,
    NoCommands,
}

impl Error for ActionError {}

impl Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCentroid => f.write_str("Duplicate centroid"),
            Self::NoCommands => f.write_str("No commands"),
        }
    }
}
