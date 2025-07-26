use nalgebra::Vector3;

use crate::Agent;

pub trait TrajectoryChaser {
    /// It is the chaser's job to remove/update the move sequence of the agent as it moves along its spline.
    fn step_chase(&self, agent: &mut Agent) -> ChaseTarget;
}

/// This is the data sent to ArduPilot.
pub struct ChaseTarget {
    pos: Vector3<f64>,
    vel: Vector3<f64>,
}

pub struct SimpleChaser;

impl SimpleChaser {}

impl TrajectoryChaser for SimpleChaser {
    fn step_chase(&self, agent: &mut Agent) -> ChaseTarget {
        // Get next
    }
}
