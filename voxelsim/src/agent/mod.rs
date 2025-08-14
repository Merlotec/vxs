use std::{error::Error, fmt::Display};

use dashmap::DashSet;
use nalgebra::{Unit, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use tinyvec::ArrayVec;

use crate::{
    ActionPlanner, Coord,
    trajectory::Trajectory,
    viewport::{CameraOrientation, CameraView},
};

pub mod chase;
pub mod trajectory;
pub mod viewport;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Agent {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub thrust: Vector3<f64>,
    /// Angular velocity of rotation around own CoM.
    pub rate: Vector3<f64>,

    pub attitude: UnitQuaternion<f64>,

    pub id: usize,

    // Current action being processed by the agent.
    pub action: Option<Action>,
}

#[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, num_enum::TryFromPrimitive,
)]
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
            MoveDir::Up => Some(Vector3::new(0, 0, -1)),
            MoveDir::Down => Some(Vector3::new(0, 0, 1)),
            MoveDir::Left => Some(Vector3::new(0, -1, 0)),
            MoveDir::Right => Some(Vector3::new(0, 1, 0)),
            MoveDir::Forward => Some(Vector3::new(1, 0, 0)),
            MoveDir::Back => Some(Vector3::new(-1, 0, 0)),
            MoveDir::Undecided => None,
        }
    }
}

pub type MoveSequence = Vec<MoveDir>;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ActionIntent {
    pub move_sequence: MoveSequence,
    pub urgency: f64,
    pub yaw: f64,
}

/// When we carry out an action we need to pyo3ensure that we remove subsequent steps when we enter the
/// relevant voxels.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Action {
    pub intent: ActionIntent,
    pub trajectory: Trajectory,
    pub origin: Vector3<i32>,
}

impl ActionIntent {
    pub fn new(urgency: f64, yaw: f64, move_sequence: MoveSequence) -> Self {
        Self {
            urgency,
            yaw,
            move_sequence,
        }
    }

    // Gets the relative position of the centroid of the grid at position t (unscaled) from the
    // centroid of the starting cell.
    pub fn relative_centroid_pos(&self, t: usize) -> Option<Coord> {
        if self.move_sequence.len() >= t {
            Some(
                self.move_sequence
                    .iter()
                    .take(t)
                    .filter_map(|x| x.dir_vector())
                    .sum(),
            )
        } else {
            None
        }
    }
}

impl Action {
    pub fn centroids<'a, I: IntoIterator<Item = &'a MoveDir>>(
        move_sequence: I,
        origin: Vector3<i32>,
    ) -> Result<Vec<Coord>, ActionError> {
        let mut total: Coord = origin;
        let mut centroids = Vec::new();
        let set: DashSet<Coord> = DashSet::new();
        // Make sure we dont duplicate origin either.
        set.insert(origin);
        for seq in move_sequence {
            if let Some(dir_vector) = seq.dir_vector() {
                let centroid = total + dir_vector;
                centroids.push(centroid);
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
            rate: Vector3::zeros(),
            action: None,
            attitude: UnitQuaternion::identity(),
        }
    }

    pub fn camera_view(&self, orientation: &CameraOrientation) -> CameraView {
        CameraView::from_pos_quat(self.pos, orientation.quat * self.attitude)
    }

    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.pos = Vector3::new(x, y, z);
    }

    pub fn get_position(&self) -> (f64, f64, f64) {
        (self.pos.x, self.pos.y, self.pos.z)
    }

    pub fn get_coord(&self) -> Coord {
        self.pos.map(|e| e.round() as i32)
    }

    pub fn set_velocity(&mut self, x: f64, y: f64, z: f64) {
        self.vel = Vector3::new(x, y, z);
    }

    pub fn perform(&mut self, intent: ActionIntent) -> Result<(), ActionError> {
        if intent.move_sequence.is_empty() {
            return Err(ActionError::NoCommands);
        }
        let origin = self.get_coord();
        let cells = Action::centroids(&intent.move_sequence, origin)?;
        let action = Action {
            intent,
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
