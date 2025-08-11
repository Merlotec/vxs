use nalgebra::{Matrix3, Rotation3, Vector3};
use peng_quad::Quadrotor;
use voxelsim::{
    Agent,
    chase::{ActionProgress, ChaseTarget},
};

use crate::dynamics::{
    AgentDynamics,
    pid::{self, AttitudePParams, Px4PosVelParams, PosVelState, RatePIDParams, RatePIDState},
};

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct QuadParams {
    mass: f64,
    gravity: f64,
    drag_coefficient: f64,
    inertia_matrix: Matrix3<f64>,
    bounding_box: Vector3<f64>,
    // PX4-like pos/vel params (hover and moving)
    pos_params: Px4PosVelParams,
    moving_pos_params: Px4PosVelParams,
    rate_params: RatePIDParams,
    moving_rate_params: RatePIDParams,
    att_p: AttitudePParams,
    thrust_min: f64,
    thrust_max: f64,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct QuadDynamics {
    quad: Quadrotor,
    params: QuadParams,
    pos_state: PosVelState,
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
            pos_params: Px4PosVelParams::default(),
            moving_pos_params: Px4PosVelParams::default(),
            rate_params: Default::default(),
            moving_rate_params: RatePIDParams::default_moving(),
            att_p: AttitudePParams::default(),
            thrust_min: 0.05,
            thrust_max: 2.5,
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
                    chaser.acc,
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

        // PX4 pos->vel->accel cascade (see pid::px4_accel_sp for mapping to MPC_* params)
        let (a_sp, _sat) = pid::px4_accel_sp(
            self.quad.position.cast::<f64>(),
            self.quad.velocity.cast::<f64>(),
            t_pos,
            t_vel,
            t_acc,
            &pos_params,
            &mut self.pos_state,
            delta,
        );
        // Total accel includes gravity compensation in ENU (Z-up)
        let a_total = a_sp + Vector3::new(0.0, 0.0, self.quad.gravity as f64);
        let mass = self.params.mass;
        let hover_thrust = mass * self.params.gravity;
        let thrust_ideal = mass * a_total.norm();
        let thrust = thrust_ideal
            .clamp(self.params.thrust_min * hover_thrust, self.params.thrust_max * hover_thrust);

        let norm = a_total.norm();
        let z_b_cmd = if norm > 1e-6 {
            a_total / norm
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        // PX4 AttitudeControl equivalent: attitude error -> body rate setpoint
        let r_cmd = Rotation3::from_matrix_unchecked(pid::build_body_rotation(&z_b_cmd, chaser.yaw));
        let rate_sp = pid::attitude_to_bodyrate(
            &r_cmd,
            &self.quad.orientation.to_rotation_matrix().cast::<f64>(),
            0.0, // no yaw FF
            &self.params.att_p.p,
        );
        // PX4 RateControl equivalent: PID on body rates
        let rate_error = rate_sp - self.quad.angular_velocity.cast::<f64>();
        let raw_torque = pid::control_torque(rate_error, &mut self.rate_state, &rate_params, delta);
        let mi_active = rate_params.max_integral;
        self.rate_state.integral.x = self.rate_state.integral.x.clamp(-mi_active.x, mi_active.x);
        self.rate_state.integral.y = self.rate_state.integral.y.clamp(-mi_active.y, mi_active.y);
        self.rate_state.integral.z = self.rate_state.integral.z.clamp(-mi_active.z, mi_active.z);
        let mt_active = rate_params.max_torque;
        let torque = Vector3::new(
            raw_torque.x.clamp(-mt_active.x, mt_active.x),
            raw_torque.y.clamp(-mt_active.y, mt_active.y),
            raw_torque.z.clamp(-mt_active.z, mt_active.z),
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
