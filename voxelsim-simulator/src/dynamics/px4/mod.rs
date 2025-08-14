use nalgebra::{UnitQuaternion, Vector3};
use voxelsim::{
    Agent,
    chase::{ActionProgress, ChaseTarget},
};

use crate::dynamics::{AgentDynamics, run_simulation_tick_rk4};
use px4_mc::Px4McController;

pub mod attitude;
pub mod position;
pub mod rate;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PX4Dynamics {
    controller: Px4McController,
    bounding_box: Vector3<f64>,
    quad: peng_quad::Quadrotor,
    hover_thrust: f32, // normalized [0,1] used by PX4 PositionControl
}

impl Default for PX4Dynamics {
    fn default() -> Self {
        // Reasonable starting values; matches QuadParams::default-like settings
        let quad = peng_quad::Quadrotor::new(
            0.01,
            0.3,
            9.8,
            0.01,
            [0.0347563, 0.0, 0.0, 0.0, 0.0458929, 0.0, 0.0, 0.0, 0.0977],
        )
        .unwrap();
        let controller = Px4McController::new(0.01, quad.mass as f32);
        let hover_thrust = 0.5_f32; // keep in sync with px4-mc default
        // Configure controller hover thrust
        let mut controller = controller;
        controller.set_hover_thrust(hover_thrust);
        Self {
            controller,
            bounding_box: Vector3::new(0.3, 0.3, 0.1),
            quad,
            hover_thrust,
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
        // Update controller dt in case it differs
        self.controller.set_dt(dt as f32);

        // Convert inputs to f32 as expected by px4_mc
        let posf = pos.cast::<f32>();
        let velf = vel.cast::<f32>();
        let ratesf = rate.cast::<f32>();
        let t_posf = t_pos.cast::<f32>();
        let t_velf = t_vel.cast::<f32>();
        let qf = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            q.w as f32, q.i as f32, q.j as f32, q.k as f32,
        ));

        let (torque, thrust_norm_z) = self
            .controller
            .update(&posf, &velf, &qf, &ratesf, &t_posf, &t_velf);
        // Map PX4 normalized thrust (FRD z-component, negative for lift) to Newtons magnitude for Peng quad
        // F = (-thr_z / hover_thrust) * m * g
        let thrust_newtons = (-(thrust_norm_z) / self.hover_thrust) as f64
            * (self.quad.mass as f64) * 9.81_f64;
        (-thrust_newtons, torque.cast::<f64>())
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

        println!("thrust: {:?}, torque: {:?}", control_thrust, control_torque);

        // Now enter RX4 inputs.
        run_simulation_tick_rk4(agent, &mut self.quad, control_thrust, control_torque, delta);

        // run_simulation_tick_rk4(
        //     agent,
        //     &mut self.quad,
        //     control_thrust,
        //     Vector3::new(0.0, 0.0, 0.0),
        //     delta,
        // );

        // agent.attitude = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0);
    }

    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
