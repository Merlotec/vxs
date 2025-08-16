use std::{error::Error, fmt::Display};

use dashmap::DashSet;
use nalgebra::{Unit, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::{
    Coord,
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
    pub state: AgentState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Action(Action),
    Hold { coord: Coord, yaw: f64 },
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

    pub next: Option<Box<Self>>,
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
    pub fn new(
        urgency: f64,
        yaw: f64,
        move_sequence: MoveSequence,
        next: Option<Box<Self>>,
    ) -> Self {
        Self {
            urgency,
            yaw,
            move_sequence,
            next,
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

    pub fn relative_end_coord(&self) -> Coord {
        self.relative_centroid_pos(self.move_sequence.len())
            .unwrap()
    }

    pub fn set_next(&mut self, next: Option<Box<Self>>) -> Option<Box<Self>> {
        let old = self.next.take();
        self.next = next;
        old
    }

    pub fn next(&self) -> Option<&Self> {
        self.next.as_deref()
    }

    pub fn next_mut(&mut self) -> Option<&mut Self> {
        self.next.as_deref_mut()
    }

    /// Can be used to feed a buffer of previous actions into an ML model.
    pub fn sequence_desc(&self) -> Vec<ActionDesc> {
        let mut a_buf = Some(self);
        let mut descs = Vec::new();
        while let Some(a) = a_buf {
            a_buf = a.next();
            descs.push(ActionDesc::from_intent(a));
        }

        descs
    }

    pub fn clone_next(&self) -> Option<Self> {
        self.next.clone().map(|x| *x)
    }

    pub fn head_intent_mut<'a>(&'a mut self) -> &'a mut Self {
        let mut a_buf: Option<&'a mut Self> = Some(self);
        loop {
            let next = a_buf.take().unwrap().next_mut();
            if let Some(next) = next {
                a_buf = Some(next);
            } else {
                return a_buf.take().unwrap();
            }
        }
    }

    pub fn push_intent(&mut self, intent: Box<ActionIntent>) {
        self.head_intent_mut().next = Some(intent);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ActionDesc {
    pub target: Coord,
    pub urgency: f64,
    pub yaw: f64,
}

impl ActionDesc {
    pub fn from_intent(intent: &ActionIntent) -> Self {
        Self {
            target: intent.relative_end_coord(),
            urgency: intent.urgency,
            yaw: intent.yaw,
        }
    }
}

impl Action {
    pub fn new(intent: ActionIntent, origin: Coord) -> Result<Self, ActionError> {
        if intent.move_sequence.is_empty() {
            return Err(ActionError::NoCommands);
        }
        let cells = Action::centroids(&intent.move_sequence, origin)?;
        let action = Action {
            intent,
            origin,
            trajectory: Trajectory::new(origin, &cells),
        };
        Ok(action)
    }

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

    pub fn end_coord(&self) -> Coord {
        self.origin + self.intent.relative_end_coord()
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
            state: AgentState::Hold {
                coord: Vector3::zeros(),
                yaw: 0.0,
            },
            attitude: UnitQuaternion::identity(),
        }
    }

    pub fn camera_view(&self, orientation: &CameraOrientation) -> CameraView {
        CameraView::from_pos_quat(self.pos, self.attitude * orientation.quat)
    }

    pub fn get_action(&self) -> Option<&Action> {
        if let AgentState::Action(action) = &self.state {
            Some(action)
        } else {
            None
        }
    }

    pub fn get_position(&self) -> Vector3<f64> {
        self.pos
    }

    pub fn get_coord(&self) -> Coord {
        self.pos.map(|e| e.round() as i32)
    }

    pub fn set_hold(&mut self, coord: Coord, yaw: f64) {
        self.pos = coord.cast();
        self.state = AgentState::Hold { coord, yaw };
    }

    pub fn set_velocity(&mut self, x: f64, y: f64, z: f64) {
        self.vel = Vector3::new(x, y, z);
    }

    pub fn perform(&mut self, intent: ActionIntent) -> Result<(), ActionError> {
        let action = Action::new(intent, self.get_coord())?;
        self.state = AgentState::Action(action);
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
