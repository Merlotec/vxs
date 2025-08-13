pub mod pid;
pub mod px4;
pub mod quad;

use nalgebra::Vector3;

use voxelsim::{Agent, chase::ChaseTarget};

use crate::dynamics::quad::QuadParams;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct EnvState {
    pub g: Vector3<f64>,
    pub wind: Vector3<f64>,
}

impl Default for EnvState {
    fn default() -> Self {
        Self {
            g: Vector3::new(0.0, -9.8, 0.0),
            wind: Vector3::zeros(),
        }
    }
}

// Completely controls how the agent's position is updated in the world through time.
pub trait AgentDynamics {
    // Responsible for updating agent physics and action state (i.e. removing actions when they have been completed).
    fn update_agent_dynamics(
        &mut self,
        agent: &mut Agent,
        env: &EnvState,
        chaser: &ChaseTarget,
        delta: f64,
    );

    fn bounding_box(&self) -> Vector3<f64>;
}

// Determines external factors e.g. wind.
pub trait EnvDynamics {
    fn agent_env(&self, agent: &Agent) -> EnvState;
}

pub fn run_simulation_tick_rk4(
    agent: &mut Agent,
    quad: &mut peng_quad::Quadrotor,
    control_thrust_ned: f64,
    control_torque_ned: Vector3<f64>,
    delta: f64,
) {
    quad.position = agent.pos.cast();
    quad.velocity = agent.vel.cast();
    quad.orientation = agent.attitude.cast();
    quad.angular_velocity = agent.rate.cast();

    quad.time_step = delta as f32;

    // Due to NED, we enter negative control thrust along the vertical (z) axis to represent forward movement.
    // quad.update_dynamics_with_controls_rk4(control_thrust as f32, &control_torque.cast());

    agent.pos = quad.position.cast();
    agent.vel = quad.velocity.cast();
    agent.attitude = quad.orientation.cast();
    agent.rate = quad.angular_velocity.cast();
}
