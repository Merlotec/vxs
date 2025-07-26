use nalgebra::Vector3;

use crate::dynamics::{AgentDynamics, EnvState};

const STABLE_VEL: f64 = 4.0;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct StandardDynamics {
    // In terms of acceleration NOT force.
    pub air_resistance: f64,
    pub g: Vector3<f64>,
    pub thrust: f64,                 // Acelleration due to thrust.
    pub thrust_urgency_ceiling: f64, // Usually set to g.
    pub bounding_box: Vector3<f64>,
    pub jerk_coeff: f64,
    pub yaw_rate: f64,
}

impl AgentDynamics for StandardDynamics {
    fn update_agent(&mut self, agent: &mut voxelsim::Agent, env: &EnvState, delta: f64) {}

    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
