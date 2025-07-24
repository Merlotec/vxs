pub mod standard;

use nalgebra::Vector3;

use voxelsim::Agent;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct EnvState {
    pub g: Vector3<f32>,
    pub wind: Vector3<f32>,
}

// Completely controls how the agent's position is updated in the world through time.
pub trait AgentDynamics {
    // Responsible for updating agent physics and action state (i.e. removing actions when they have been completed).
    fn update_agent(&self, agent: &mut Agent, env: &EnvState, delta: f32);
    fn bounding_box(&self) -> Vector3<f32>;
}

// Determines external factors e.g. wind.
pub trait EnvDynamics {
    fn agent_env(&self, agent: &Agent) -> EnvState;
}
