use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};

use serde::{Deserialize, Serialize};
use voxelsim::{
    VoxelGrid,
    agent::Agent,
    env::CollisionShell,
    viewport::{self, CameraProjection, VirtualGrid},
};

use crate::dynamics::{AgentDynamics, EnvDynamics, EnvState};

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Collision {
    pub agent_id: usize,
    pub shell: CollisionShell,
}
