use std::time::{Duration, SystemTime};

use crate::{
    agent::{Agent, AgentDynamics},
    env::{CollisionShell, GlobalEnv},
    viewport::{self, CameraView, VirtualGrid},
};

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

    pub fn update<F, C>(
        &mut self,
        dynamics: &AgentDynamics,
        delta: f32,
        mut step_f: F,
        mut collide_f: C,
    ) where
        F: FnMut(&mut GlobalEnv),
        C: FnMut(&mut GlobalEnv, Agent, CollisionShell),
    {
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
            collide_f(self, agent, cols);
        }
        step_f(self);
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

    pub fn update_povs(&mut self) {
        println!("preupdate");
        let povs = self.agents.iter().map(|(id, a)| {
            let camera = a.camera();
            let im = viewport::raycast_slice(&self.world, camera, 16, 9);
            let vw = VirtualGrid::world_from_intersection_map(im);
            (*id, vw, camera)
        });
        println!("postupdate");

        for (id, vw, camera) in povs {
            if let Some(pov) = self.povs.get_mut(&id) {
                println!("merge");

                pov.virtual_world.merge(vw, &camera);
            } else {
                self.povs.insert(
                    id,
                    crate::PovData {
                        virtual_world: vw,
                        agent_id: id,
                    },
                );
            }
        }
        println!("endd");
    }
}
