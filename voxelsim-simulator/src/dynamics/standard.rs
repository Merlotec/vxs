use nalgebra::Vector3;

use crate::dynamics::{AgentDynamics, EnvState};

const STABLE_VEL: f32 = 4.0;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct StandardDynamics {
    // In terms of acceleration NOT force.
    pub air_resistance: f32,
    pub g: Vector3<f32>,
    pub thrust: f32,                 // Acelleration due to thrust.
    pub thrust_urgency_ceiling: f32, // Usually set to g.
    pub bounding_box: Vector3<f32>,
    pub jerk_coeff: f32,
    pub yaw_rate: f32,
}

impl AgentDynamics for StandardDynamics {
    fn update_agent(&self, agent: &mut voxelsim::Agent, env: &EnvState, delta: f32) {}

    fn bounding_box(&self) -> Vector3<f32> {
        self.bounding_box
    }
}
