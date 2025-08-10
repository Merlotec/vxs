use nalgebra::{Matrix3, Rotation3, Vector3};
use peng_quad::Quadrotor;
use voxelsim::{
    Agent,
    chase::{ActionProgress, ChaseTarget},
};

use crate::dynamics::{
    AgentDynamics,
    drone::{self, PosPIDParams, PosPIDState, RatePIDParams, RatePIDState},
};

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct QuadParams {
    mass: f64,
    gravity: f64,
    drag_coefficient: f64,
    inertia_matrix: Matrix3<f64>,
    bounding_box: Vector3<f64>,
    pos_params: PosPIDParams,
    moving_pos_params: PosPIDParams,
    rate_params: RatePIDParams,
    moving_rate_params: RatePIDParams,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct QuadDynamics {
    quad: Quadrotor,
    params: QuadParams,
    pos_state: PosPIDState,
    rate_state: RatePIDState,
}

impl Default for QuadParams {
    fn default() -> Self {
        Self::new(
            1.0,
            9.81,
            0.1,
            Matrix3::new(0.0347563, 0.0, 0.0, 0.0, 0.0458929, 0.0, 0.0, 0.0, 0.0977),
            Vector3::new(0.5, 0.5, 0.1),
        )
    }
}

impl QuadDynamics {
    pub fn new(settings: QuadParams) -> Self {
        let quad = Quadrotor::new(
            0.01,
            settings.mass as f32,
            settings.gravity as f32,
            settings.drag_coefficient as f32,
            settings
                .inertia_matrix
                .cast::<f32>()
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();

        Self {
            quad,
            pos_state: Default::default(),
            rate_state: Default::default(),
            params: settings,
        }
    }
}

impl QuadParams {
    pub fn new(
        mass: f64,
        gravity: f64,
        drag_coefficient: f64,
        inertia_matrix: Matrix3<f64>,
        bounding_box: Vector3<f64>,
    ) -> Self {
        Self {
            mass,
            gravity,
            drag_coefficient,
            inertia_matrix,
            bounding_box: bounding_box,
            pos_params: Default::default(),
            moving_pos_params: PosPIDParams::default_moving(),
            rate_params: Default::default(),
            moving_rate_params: RatePIDParams::default_moving(),
        }
    }
}

impl AgentDynamics for QuadDynamics {
    fn update_agent_dynamics(
        &mut self,
        agent: &mut Agent,
        _env: &super::EnvState,
        chaser: &ChaseTarget,
        delta: f64,
    ) {
        let (t_pos, t_vel, t_acc, pos_params, rate_params) = match chaser.progress {
            ActionProgress::ProgressTo(p) => {
                if let Some(action) = &mut agent.action {
                    action.trajectory.progress = p;
                }
                (
                    chaser.pos,
                    chaser.vel,
                    Vector3::zeros(),
                    self.params.moving_pos_params,
                    self.params.moving_rate_params,
                )
            }
            ActionProgress::Complete => {
                agent.action = None;
                (
                    agent.pos,
                    Vector3::zeros(),
                    Vector3::zeros(),
                    self.params.pos_params,
                    self.params.rate_params,
                )
            }
            ActionProgress::Hold => (
                agent.pos,
                Vector3::zeros(),
                Vector3::zeros(),
                self.params.pos_params,
                self.params.rate_params,
            ),
        };
        self.quad.position = agent.pos.cast::<f32>();
        self.quad.velocity = agent.vel.cast::<f32>();

        let a_cmd = drone::compute_accel_cmd(
            self.quad.position.cast::<f64>(),
            self.quad.velocity.cast::<f64>(),
            t_pos,
            t_vel,
            t_acc,
            &pos_params,
            &mut self.pos_state,
            delta,
        );

        let a_total = a_cmd + Vector3::new(0.0, 0.0, self.quad.gravity as f64);
        let thrust = self.quad.mass as f64 * a_total.norm();

        let norm = a_total.norm();
        let z_b_cmd = if norm > 1e-6 {
            a_total / norm
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        let r_cmd =
            Rotation3::from_matrix_unchecked(drone::build_body_rotation(&z_b_cmd, chaser.yaw));
        let rate_sp = drone::attitude_to_bodyrate(
            &r_cmd,
            &self.quad.orientation.to_rotation_matrix().cast::<f64>(),
            0.0, // no yaw feed‐forward
        );

        let rate_error = rate_sp - self.quad.angular_velocity.cast::<f64>();
        let raw_torque =
            drone::control_torque(rate_error, &mut self.rate_state, &rate_params, delta);

        let mi = self.params.rate_params.max_integral;
        self.rate_state.integral.x = self.rate_state.integral.x.clamp(-mi.x, mi.x);
        self.rate_state.integral.y = self.rate_state.integral.y.clamp(-mi.y, mi.y);
        self.rate_state.integral.z = self.rate_state.integral.z.clamp(-mi.z, mi.z);

        let mt = self.params.rate_params.max_torque;
        let torque = Vector3::new(
            raw_torque.x.clamp(-mt.x, mt.x),
            raw_torque.y.clamp(-mt.y, mt.y),
            raw_torque.z.clamp(-mt.z, mt.z),
        );
        self.quad.time_step = delta as f32;

        self.quad
            .update_dynamics_with_controls_rk4(thrust as f32, &torque.cast::<f32>());

        agent.pos = self.quad.position.cast::<f64>();
        agent.vel = self.quad.velocity.cast::<f64>();
        agent.attitude = self.quad.orientation.cast::<f64>();
    }
    fn bounding_box(&self) -> Vector3<f64> {
        self.params.bounding_box
    }
}
