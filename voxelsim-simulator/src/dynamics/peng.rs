use nalgebra::{Matrix3, Rotation3, Vector3};
use peng_quad::{PlannerManager, QPpolyTrajPlanner, Quadrotor, config::SimulationConfig};
use voxelsim::{
    Agent,
    chase::{ActionProgress, ChaseTarget},
};

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
            Vector3::new(0.5, 0.5, 0.1),
        )
    }
}

impl PengQuadDynamics {
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
            bounding_box: bounding_box,
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
        // 1) Advance trajectory progress

        let (t_pos, t_vel, t_acc) = match chaser.progress {
            ActionProgress::ProgressTo(p) => {
                if let Some(action) = &mut agent.action {
                    action.trajectory.progress = p;
                }
                (chaser.pos, chaser.vel, Vector3::zeros())
            }
            ActionProgress::Complete => {
                agent.action = None;
                (agent.pos, Vector3::zeros(), Vector3::zeros())
            }
            ActionProgress::Hold => (agent.pos, Vector3::zeros(), Vector3::zeros()),
        };
        // 2) Sync quad state to our Agent
        self.quad.position = agent.pos.cast::<f32>();
        self.quad.velocity = agent.vel.cast::<f32>();

        // 3) Compute desired acceleration to chase the target
        println!("chaser: {:?}", chaser);
        println!(
            "targetting_delta: {:?}",
            t_pos - self.quad.position.cast::<f64>()
        );

        let a_cmd = drone::compute_accel_cmd(
            self.quad.position.cast::<f64>(),
            self.quad.velocity.cast::<f64>(),
            t_pos,
            t_vel,
            t_acc,
            &self.pos_params,
            &mut self.pos_state,
            delta,
        );
        println!("a_cmd = {:?}", a_cmd);

        // 4) Add gravity, compute thrust magnitude
        let a_total = a_cmd + Vector3::new(0.0, 0.0, self.quad.gravity as f64);
        println!("a_total = {:?}", a_total);
        let thrust = self.quad.mass as f64 * a_total.norm();

        // 5) Compute desired body-z axis (with zero‐vector guard)
        let norm = a_total.norm();
        let z_b_cmd = if norm > 1e-6 {
            a_total / norm
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        // 6) Build attitude setpoint
        let r_cmd =
            Rotation3::from_matrix_unchecked(drone::build_body_rotation(&z_b_cmd, chaser.yaw));
        let rate_sp = drone::attitude_to_bodyrate(
            &r_cmd,
            &self.quad.orientation.to_rotation_matrix().cast::<f64>(),
            0.0, // no yaw feed‐forward
        );

        // 7) Rate error & raw torque from PID
        let rate_error = rate_sp - self.quad.angular_velocity.cast::<f64>();
        let raw_torque =
            drone::control_torque(rate_error, &mut self.rate_state, &self.rate_params, delta);
        println!("raw torque = {:?}", raw_torque);

        // 8) Anti‐windup: clamp the integral state
        let mi = self.rate_params.max_integral;
        self.rate_state.integral.x = self.rate_state.integral.x.clamp(-mi.x, mi.x);
        self.rate_state.integral.y = self.rate_state.integral.y.clamp(-mi.y, mi.y);
        self.rate_state.integral.z = self.rate_state.integral.z.clamp(-mi.z, mi.z);

        // 9) Torque clamp: enforce physical max torque
        let mt = self.rate_params.max_torque;
        let torque = Vector3::new(
            raw_torque.x.clamp(-mt.x, mt.x),
            raw_torque.y.clamp(-mt.y, mt.y),
            raw_torque.z.clamp(-mt.z, mt.z),
        );
        println!("clamped torque = {:?}", torque);
        println!("thrust: {}", thrust);
        // 10) Optionally cap the integration timestep
        self.quad.time_step = delta as f32;

        // 11) Integrate dynamics via RK4
        self.quad
            .update_dynamics_with_controls_rk4(thrust as f32, &torque.cast::<f32>());

        // 12) Write back into Agent
        agent.pos = self.quad.position.cast::<f64>();
        agent.vel = self.quad.velocity.cast::<f64>();
        println!("outpos: {}", agent.pos);
    }
    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
