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
        let new_pid = self
            .position_control
            .velocity_controller(vel, vel_sp, self.pid, dt);

        self.pid = new_pid;

        let att_sp = attitude::thrust_to_attitude(new_pid.thr_sp, t_yaw);

        let rate_sp = self.attitude_control.update(q, att_sp.q_d, 0.0);

        let (rate_sp, new_rate_state) =
            self.rate_control
                .update(rate, rate_sp, self.rate_state, self.saturation_flags, dt);

        self.rate_state = new_rate_state;

        // Do something with rate_sp.
        (att_sp.thrust_body.z, rate_sp)
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

        self.quad
            .update_dynamics_with_controls_rk4(control_thrust as f32, &control_torque.cast());

        agent.pos = self.quad.position.cast();
        agent.vel = self.quad.velocity.cast();
        agent.attitude = self.quad.orientation.cast();
    }

    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
