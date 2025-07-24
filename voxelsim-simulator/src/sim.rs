use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};

use serde::{Deserialize, Serialize};
use voxelsim::{
    VoxelGrid,
    agent::Agent,
    env::CollisionShell,
    viewport::{self, CameraProjection, IntersectionMap, VirtualGrid},
};

use crate::dynamics::{AgentDynamics, EnvDynamics, EnvState};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct GlobalEnv {
    pub world: VoxelGrid,
    pub agents: HashMap<usize, Agent>,
}

impl GlobalEnv {
    pub fn agent(&self, id: &usize) -> Option<&Agent> {
        self.agents.get(id)
    }

    pub fn agent_mut(&mut self, id: &usize) -> Option<&mut Agent> {
        self.agents.get_mut(id)
    }
}
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Collision {
    pub agent_id: usize,
    pub shell: CollisionShell,
}

impl GlobalEnv {
    pub fn update<D: AgentDynamics + ?Sized>(
        &mut self,
        dynamics: &D,
        env: &EnvState,
        delta: f32,
    ) -> Vec<Collision> {
        self.agents
            .iter_mut()
            .map(|(id, agent)| {
                dynamics.update_agent(agent, env, delta);
                Collision {
                    agent_id: *id,
                    shell: self.world.collisions(agent.pos, dynamics.bounding_box()),
                }
            })
            .collect()
    }
}
