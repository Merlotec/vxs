use nalgebra::{Matrix3, Rotation3, Vector3};
use peng_quad::{PlannerManager, QPpolyTrajPlanner, Quadrotor, config::SimulationConfig};
use voxelsim::{Agent, chase::ChaseTarget};

use crate::dynamics::{
    AgentDynamics,
    drone::{self, PosPIDParams, PosPIDState, RatePIDParams, RatePIDState},
};

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PengQuadDynamics {
    quad: Quadrotor,
    bounding_box: Vector3<f64>,
    pos_params: PosPIDParams,
    pos_state: PosPIDState,
    rate_params: RatePIDParams,
    rate_state: RatePIDState,
}

impl Default for PengQuadDynamics {
    fn default() -> Self {
        Self::new(
            1.0,
            9.81,
            0.1,
            Matrix3::new(0.0347563, 0.0, 0.0, 0.0, 0.0458929, 0.0, 0.0, 0.0, 0.0977),
            Vector3::new(0.5, 0.1, 0.5),
        )
    }
}

impl PengQuadDynamics {
    pub fn switch_vec<T: Copy>(v: Vector3<T>) -> Vector3<T> {
        Vector3::new(v[0], v[2], v[1])
    }

    pub fn new(
        mass: f64,
        gravity: f64,
        drag_coefficient: f64,
        inertia_matrix: Matrix3<f64>,
        bounding_box: Vector3<f64>,
    ) -> Self {
        let quad = Quadrotor::new(
            0.01,
            mass as f32,
            gravity as f32,
            drag_coefficient as f32,
            inertia_matrix.cast::<f32>().as_slice().try_into().unwrap(),
        )
        .unwrap();
        Self {
            quad,
            bounding_box: Self::switch_vec(bounding_box),
            pos_params: Default::default(),
            pos_state: Default::default(),
            rate_params: Default::default(),
            rate_state: Default::default(),
        }
    }
}

impl AgentDynamics for PengQuadDynamics {
    fn update_agent_dynamics(
        &mut self,
        agent: &mut Agent,
        env: &super::EnvState,
        chaser: &ChaseTarget,
        delta: f64,
    ) {
        self.quad.time_step = delta as f32;
        self.quad.position = Self::switch_vec(agent.pos.cast::<f32>());
        self.quad.velocity = Self::switch_vec(agent.pos.cast::<f32>());
        let a_cmd = drone::compute_accel_cmd(
            self.quad.position.cast::<f64>(),
            self.quad.velocity.cast::<f64>(),
            chaser.pos,
            chaser.vel,
            chaser.acc,
            &self.pos_params,
            &mut self.pos_state,
            delta,
        );

        let a_total = a_cmd + Vector3::new(0.0, 0.0, self.quad.gravity as f64);
        let thrust = self.quad.mass as f64 * a_total.norm();

        let z_b_cmd = -a_total.normalize();
        let yaw_cmd = chaser.yaw;
        let r_cmd = Rotation3::from_matrix_unchecked(drone::build_body_rotation(&z_b_cmd, yaw_cmd));

        let yaw_rate_ff = 0.0; // zero if no feed-forward
        let rate_sp = drone::attitude_to_bodyrate(
            &r_cmd,
            &self.quad.orientation.to_rotation_matrix().cast::<f64>(),
            yaw_rate_ff,
        );

        let rate_error = rate_sp - self.quad.angular_velocity.cast::<f64>();
        let torque =
            drone::control_torque(rate_error, &mut self.rate_state, &self.rate_params, delta);
        // TODO: should really sync everything, but for speed will just update from this.
        self.quad
            .update_dynamics_with_controls_rk4(thrust as f32, &torque.cast::<f32>());

        // Update the agent from the quad.
        agent.pos = Self::switch_vec(self.quad.position.cast::<f64>());
        agent.vel = Self::switch_vec(self.quad.velocity.cast::<f64>());
    }
    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
