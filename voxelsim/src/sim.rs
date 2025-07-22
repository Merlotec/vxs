use std::time::{Duration, SystemTime};

use crate::{
    agent::{Agent, AgentDynamics},
    env::{CollisionShell, GlobalEnv},
    viewport::{self, CameraProjection, IntersectionMap, VirtualGrid},
};

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Collision {
    pub agent_id: usize,
    pub shell: CollisionShell,
}

impl GlobalEnv {
    pub fn n_step(&mut self, dynamics: &AgentDynamics, delta: f32, n: usize) {
        for _ in 0..n {
            self.step(dynamics, delta);
        }
    }

    pub fn step(&mut self, dynamics: &AgentDynamics, delta: f32) {
        for (_id, a) in self.agents.iter_mut() {
            // Move the agent according to its current action.
            a.step(dynamics, delta);
        }
    }

    pub fn update(&mut self, dynamics: &AgentDynamics, delta: f32) -> Vec<Collision> {
        self.step(dynamics, delta);
        self.agents
            .iter()
            .map(|(id, agent)| Collision {
                agent_id: *id,
                shell: self.world.collisions(agent.pos, dynamics.bounding_box),
            })
            .collect()
    }

    pub fn run_sim<F, C>(
        mut self,
        dynamics: &AgentDynamics,
        delta: f32,
        realtime: f32,
        mut step_f: F,
        mut collide_f: C,
    ) where
        F: FnMut(&mut GlobalEnv),
        C: FnMut(&mut GlobalEnv, Agent, CollisionShell),
    {
        // Tick then compare the difference between time elapsed and realtime.
        let target_time = delta * realtime;

        loop {
            let ts = SystemTime::now();
            self.step(dynamics, delta);

            // Handle any collisions first (agent <-> environment)
            let mut cols: Vec<(Agent, CollisionShell)> = Vec::new();
            for (_id, agent) in self.agents.clone().iter() {
                let collisions = self.world.collisions(agent.pos, dynamics.bounding_box);
                if !collisions.is_empty() {
                    cols.push((agent.clone(), collisions));
                }
            }

            for (agent, cols) in cols {
                collide_f(&mut self, agent, cols);
            }
            step_f(&mut self);

            let elapsed = ts.elapsed().unwrap().as_secs_f32();

            let wait = target_time - elapsed;
            if wait > 0.0 {
                std::thread::sleep(Duration::from_secs_f32(wait));
            }
        }
    }

    pub fn update_povs(&mut self, proj: &CameraProjection) {
        let povs = self.agents.iter().map(|(id, a)| {
            let camera = a.camera_view();
            let im = viewport::raycast_slice(&self.world, &camera, proj, 16, 9);
            let vw = VirtualGrid::world_from_intersection_map(&im);
            (*id, vw, camera)
        });

        for (id, vw, camera) in povs {
            if let Some(pov) = self.povs.get_mut(&id) {
                pov.virtual_world.merge(vw, &camera, proj);
            } else {
                self.povs.insert(
                    id,
                    crate::PovData {
                        virtual_world: vw,
                        agent_id: id,
                        proj: *proj,
                    },
                );
            }
        }
    }
    pub fn update_pov(
        &mut self,
        proj: &CameraProjection,
        agent_id: &usize,
    ) -> Option<IntersectionMap> {
        let (vw, im, camera) = if let Some(agent) = self.agent(agent_id) {
            let camera = agent.camera_view();
            let im = viewport::raycast_slice(&self.world, &camera, proj, 16, 9);
            let vw = VirtualGrid::world_from_intersection_map(&im);
            (vw, im, camera)
        } else {
            return None;
        };

        if let Some(pov) = self.povs.get_mut(agent_id) {
            pov.virtual_world.merge(vw, &camera, proj);
        } else {
            self.povs.insert(
                *agent_id,
                crate::PovData {
                    virtual_world: vw,
                    agent_id: *agent_id,
                    proj: *proj,
                },
            );
        }
        Some(im)
    }
}
