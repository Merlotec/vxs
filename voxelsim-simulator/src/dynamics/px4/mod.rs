use nalgebra::{UnitQuaternion, Vector3};
use voxelsim::{
    Agent,
    chase::{ActionProgress, ChaseTarget},
};

use crate::dynamics::{
    AgentDynamics,
    px4::{
        attitude::AttitudeControl,
        position::{PIDState, PositionControl},
        rate::{RateControl, RateState, SaturationFlags},
    },
};

pub mod attitude;
pub mod position;
pub mod rate;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PX4Dynamics {
    position_control: PositionControl,
    attitude_control: AttitudeControl,
    rate_control: RateControl,

    rate_state: RateState,
    saturation_flags: SaturationFlags,

    bounding_box: Vector3<f64>,
    pid: PIDState,

    quad: peng_quad::Quadrotor,
}

impl Default for PX4Dynamics {
    fn default() -> Self {
        // Reasonable starting values; matches QuadParams::default-like settings
        let quad = peng_quad::Quadrotor::new(
            0.01,
            0.3,
            -9.81,
            0.01,
            [0.0347563, 0.0, 0.0, 0.0, 0.0458929, 0.0, 0.0, 0.0, 0.0977],
        )
        .unwrap();
        Self {
            position_control: PositionControl::default(),
            attitude_control: AttitudeControl::default(),
            rate_control: RateControl::default(),
            rate_state: RateState::default(),
            saturation_flags: SaturationFlags::default(),
            bounding_box: Vector3::new(0.3, 0.3, 0.1),
            pid: PIDState::default(),
            quad,
        }
    }
}

impl PX4Dynamics {
    pub fn next_body_rate_sp(
        &mut self,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        rate: Vector3<f64>,
        q: UnitQuaternion<f64>,
        t_pos: Vector3<f64>,
        t_vel: Vector3<f64>,
        t_yaw: f64,
        dt: f64,
    ) -> (f64, Vector3<f64>) {
        let vel_sp = self.position_control.position_controller(pos, t_pos, t_vel);

        let mut pid = self.pid;
        // Tell PX4 to fill in these values.
        pid.thr_sp.x = f64::NAN;
        pid.thr_sp.y = f64::NAN;

        let new_pid = self
            .position_control
            .velocity_controller(vel, vel_sp, pid, dt);

        self.pid = new_pid;

        println!("Thrust setpoint: {:?}", new_pid.thr_sp);

        let att_sp = attitude::thrust_to_attitude(new_pid.thr_sp, t_yaw);

        let body_dir = att_sp.q_d * (-Vector3::z_axis());
        println!("Body target dir: {:?}", body_dir);

        let rate_sp = self.attitude_control.update(q, att_sp.q_d, 0.0);
        println!("Rate sp: {:?}", rate_sp);

        let (torque, new_rate_state) = self.rate_control.update(
            rate,
            rate_sp,
            self.rate_state,
            self.saturation_flags,
            dt,
            false,
        );
        println!("Body torque: {:?}", torque);

        self.rate_state = new_rate_state;

        // Do something with rate_sp.
        (att_sp.thrust_body.z, torque)
    }

    pub fn mass(&self) -> f64 {
        self.quad.mass as f64
    }
}

impl AgentDynamics for PX4Dynamics {
    fn update_agent_dynamics(
        &mut self,
        agent: &mut Agent,
        _env: &super::EnvState,
        chaser: &ChaseTarget,
        delta: f64,
    ) {
        let (t_pos, t_vel, t_acc) = match chaser.progress {
            ActionProgress::ProgressTo(p) => {
                if let Some(action) = &mut agent.action {
                    action.trajectory.progress = p;
                }
                // Debug print removed to avoid runtime jitter
                (chaser.pos, chaser.vel, chaser.acc)
            }
            ActionProgress::Complete => {
                agent.action = None;
                (agent.pos, Vector3::zeros(), Vector3::zeros())
            }
            ActionProgress::Hold => (agent.pos, Vector3::zeros(), Vector3::zeros()),
        };

        let (control_thrust, control_torque) = self.next_body_rate_sp(
            agent.pos,
            agent.vel,
            agent.rate,
            agent.attitude,
            t_pos,
            t_vel,
            chaser.yaw,
            delta,
        );

        // Now enter RX4 inputs.
        self.quad.position = agent.pos.cast();
        self.quad.velocity = agent.vel.cast();
        self.quad.orientation = agent.attitude.cast();
        self.quad.angular_velocity = agent.rate.cast();

        self.quad.time_step = delta as f32;

        // Due to NED, we enter negative control thrust along the vertical (z) axis to represent forward movement.
        self.quad
            .update_dynamics_with_controls_rk4(control_thrust as f32, &control_torque.cast());

        agent.pos = self.quad.position.cast();
        agent.vel = self.quad.velocity.cast();
        agent.attitude = self.quad.orientation.cast();
        agent.rate = self.quad.angular_velocity.cast();
    }

    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
