use nalgebra::{Matrix3, Vector3};
use peng_quad::{PlannerManager, QPpolyTrajPlanner, Quadrotor, config::SimulationConfig};
use voxelsim::Agent;

use crate::dynamics::AgentDynamics;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct PengQuadDynamics {
    quad: Quadrotor,
    planner_manager: PlannerManager,
    internal_time: f64,
    bounding_box: Vector3<f64>,
}

impl PengQuadDynamics {
    pub fn new(
        pos: Vector3<f32>,
        time_step: f32,
        mass: f32,
        gravity: f32,
        drag_coefficient: f32,
        inertia_matrix: Matrix3<f32>,
        bounding_box: Vector3<f64>,
    ) -> Self {
        let mut quad = Quadrotor::new(
                time_step,
                mass,
                gravity,
                drag_coefficient,
                inertia_matrix.as_slice().try_into().unwrap(),
            )
            .unwrap();
        quad.position = pos;
        Self {
            quad,
            internal_time: 0.0,
            planner_manager: PlannerManager::new(Vector3::zeros(), 0.0),
            bounding_box,
        }
    }

    pub fn next_state(
        &mut self,
        agent: &mut Agent,
        delta: f64,
    ) -> (Vector3<f64>, Vector3<f32>, f64) {
        self.internal_time += delta;
        let (pos, vel, yaw) = self.planner_manager.current_planner.plan(
            agent.pos.cast(),
            agent.vel.cast(),
            self.internal_time as f32,
        );

        (pos.cast(), vel.cast(), yaw as f64)
    }

    pub fn update_planner(&mut self, agent: &Agent) {
        if let Some(waypoints) = agent.trajectory.map(|x| x.waypoints(30)) {
        let planner = QPpolyTrajPlanner::new(waypoints,)
        }
    }
}

impl AgentDynamics for PengQuadDynamics {
    fn update_agent(&mut self, agent: &mut Agent, env: &super::EnvState, delta: f64) {
        self.next_state(agent, delta);
    }

    fn bounding_box(&self) -> Vector3<f64> {
        self.bounding_box
    }
}
