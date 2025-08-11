use nalgebra::{Matrix3, Rotation3, Vector3};
use peng_quad::Quadrotor;
use voxelsim::{
    Agent,
    chase::{ActionProgress, ChaseTarget},
};

use crate::dynamics::{
    AgentDynamics,
    pid::{self, AttitudePParams, PosVelState, Px4PosVelParams, RatePIDParams, RatePIDState},
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
    // Control axis sign mapping: multiply setpoints by this before control
    // Use (-1,-1,-1) to invert FB/LR/UD to match external input convention
    control_sign: Vector3<f64>,
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
            moving_pos_params: Px4PosVelParams::default_moving(),
            rate_params: Default::default(),
            moving_rate_params: RatePIDParams::default_moving(),
            att_p: AttitudePParams::default(),
            thrust_min: 0.05,
            thrust_max: 2.5,
            // Use ENU (X-east, Y-north, Z-up) throughout; no inversion by default
            control_sign: Vector3::new(1.0, 1.0, 1.0),
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
        let (mut t_pos, mut t_vel, mut t_acc, pos_params, rate_params) = match chaser.progress {
            ActionProgress::ProgressTo(p) => {
                if let Some(action) = &mut agent.action {
                    action.trajectory.progress = p;
                }
                // println!(
                //     "t_pos: {}, t_vel: {}, t_acc:{}",
                //     chaser.pos, chaser.vel, chaser.acc
                // );
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
        // Apply control axis sign mapping to setpoints (PX4 uses NED; external may be ENU)
        // We flip relative directions by applying sign to deltas for position and directly to vel/acc.
        let sign = self.params.control_sign;
        t_pos = self.quad.position.cast::<f64>()
            + sign.component_mul(&(t_pos - self.quad.position.cast::<f64>()));
        t_vel = sign.component_mul(&t_vel);
        t_acc = sign.component_mul(&t_acc);
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
        let thrust = thrust_ideal.clamp(
            self.params.thrust_min * hover_thrust,
            self.params.thrust_max * hover_thrust,
        );

        let norm = a_total.norm();
        let z_b_cmd = if norm > 1e-6 {
            a_total / norm
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        // PX4 AttitudeControl equivalent: attitude error -> body rate setpoint
        let r_cmd =
            Rotation3::from_matrix_unchecked(pid::build_body_rotation(&z_b_cmd, chaser.yaw));
        let rate_sp = pid::attitude_to_bodyrate(
            &r_cmd,
            &self.quad.orientation.to_rotation_matrix().cast::<f64>(),
            0.0, // no yaw FF
            &self.params.att_p.p,
        );
        // PX4 RateControl equivalent: PID on body rates
        // PX4-like RateControl with simple per-axis anti-windup (freeze on same-direction saturation)
        let rate_error = rate_sp - self.quad.angular_velocity.cast::<f64>();
        let p_term = Vector3::new(
            rate_params.kp.x * rate_error.x,
            rate_params.kp.y * rate_error.y,
            rate_params.kp.z * rate_error.z,
        );
        // Derivative with simple low-pass filtering to reduce oscillation (approx PX4 gyro filter)
        let raw_d = (rate_error - self.rate_state.last_error) / delta;
        let fc = rate_params.d_lpf_hz.max(0.1);
        let tau = 1.0 / (2.0 * std::f64::consts::PI * fc);
        let alpha = tau / (tau + delta);
        self.rate_state.d_filt = self.rate_state.d_filt * alpha + raw_d * (1.0 - alpha);
        let d_err = self.rate_state.d_filt;
        let d_term = Vector3::new(
            rate_params.kd.x * d_err.x,
            rate_params.kd.y * d_err.y,
            rate_params.kd.z * d_err.z,
        );
        let integral_candidate = self.rate_state.integral + rate_error * delta;
        let i_term_candidate = Vector3::new(
            rate_params.ki.x * integral_candidate.x,
            rate_params.ki.y * integral_candidate.y,
            rate_params.ki.z * integral_candidate.z,
        );
        let raw_candidate = p_term + i_term_candidate + d_term;
        let mt_active = rate_params.max_torque;
        // Anti-windup masks per axis
        let allow_x = !((raw_candidate.x > mt_active.x && rate_error.x > 0.0)
            || (raw_candidate.x < -mt_active.x && rate_error.x < 0.0));
        let allow_y = !((raw_candidate.y > mt_active.y && rate_error.y > 0.0)
            || (raw_candidate.y < -mt_active.y && rate_error.y < 0.0));
        let allow_z = !((raw_candidate.z > mt_active.z && rate_error.z > 0.0)
            || (raw_candidate.z < -mt_active.z && rate_error.z < 0.0));
        if allow_x {
            self.rate_state.integral.x = integral_candidate.x;
        }
        if allow_y {
            self.rate_state.integral.y = integral_candidate.y;
        }
        if allow_z {
            self.rate_state.integral.z = integral_candidate.z;
        }
        let mi_active = rate_params.max_integral;
        self.rate_state.integral.x = self.rate_state.integral.x.clamp(-mi_active.x, mi_active.x);
        self.rate_state.integral.y = self.rate_state.integral.y.clamp(-mi_active.y, mi_active.y);
        self.rate_state.integral.z = self.rate_state.integral.z.clamp(-mi_active.z, mi_active.z);
        let raw_torque = Vector3::new(
            p_term.x + rate_params.ki.x * self.rate_state.integral.x + d_term.x,
            p_term.y + rate_params.ki.y * self.rate_state.integral.y + d_term.y,
            p_term.z + rate_params.ki.z * self.rate_state.integral.z + d_term.z,
        );
        let torque = Vector3::new(
            raw_torque.x.clamp(-mt_active.x, mt_active.x),
            raw_torque.y.clamp(-mt_active.y, mt_active.y),
            raw_torque.z.clamp(-mt_active.z, mt_active.z),
        );
        self.rate_state.last_error = rate_error;
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
