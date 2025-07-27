use nalgebra::Vector3;

use crate::Agent;

pub trait TrajectoryChaser {
    /// It is the chaser's job to remove/update the move sequence of the agent as it moves along its spline.
    fn step_chase(&self, agent: &Agent, delta: f64) -> ChaseTarget;
}

/// This is the data sent to ArduPilot.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ChaseTarget {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub acc: Vector3<f64>,

    pub yaw: f64,

    pub progress: ActionProgress,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum ActionProgress {
    /// Progress the current action to the specified progress.
    ProgressTo(f64),
    /// End the current action.
    Complete,
    /// Do not update progress (this can be used if there is no currently active action).
    #[default]
    Hold,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct FixedLookaheadChaser {
    pub v_max_base: f64,
    pub s_lookahead_base: f64,
}

impl Default for FixedLookaheadChaser {
    fn default() -> Self {
        Self {
            v_max_base: 3.0,
            s_lookahead_base: 0.3,
        }
    }
}

impl TrajectoryChaser for FixedLookaheadChaser {
    fn step_chase(&self, agent: &Agent, delta: f64) -> ChaseTarget {
        // Get next chase.
        if let Some(action) = &agent.action {
            let x_act = agent.pos;

            let s_cur = action.trajectory.progress;
            let s_end = action.trajectory.length();
            if let (Some(urgency), Some(yaw)) = (
                action.trajectory.sample_urgencies(s_cur),
                action.trajectory.sample_yaw(s_cur),
            ) {
                let v_max_cur = self.v_max_base * urgency;
                let ds_max = v_max_cur * delta;
                let s_lookahead = self.s_lookahead_base * urgency;

                let s_star = action
                    .trajectory
                    .find_nearest_param_in_range(s_cur, s_end, &x_act)
                    .expect(&format!("No nearest param in range: {}, {}", s_cur, s_end));

                let s_updated = s_star.clamp(s_cur, s_cur + ds_max);

                let s_tgt = (s_updated + s_lookahead).min(s_end);
                if let (Some(p_tgt), Some(v_tgt_nominal), Some(a_tgt_nominal)) = (
                    action.trajectory.position(s_tgt),
                    action.trajectory.velocity(s_tgt),
                    action.trajectory.acceleration(s_tgt),
                ) {
                    // 6) Scale velocity and accel targets by urgency
                    let v_tgt = v_tgt_nominal * urgency;
                    let a_tgt = a_tgt_nominal * urgency;

                    let progress = if s_updated < s_end {
                        ActionProgress::ProgressTo(s_updated)
                    } else {
                        ActionProgress::Complete
                    };

                    return ChaseTarget {
                        pos: p_tgt,
                        vel: v_tgt,
                        acc: a_tgt,
                        yaw,
                        progress,
                    };
                } else {
                    return ChaseTarget {
                        progress: ActionProgress::Complete,
                        ..Default::default()
                    };
                }
            }
        }
        ChaseTarget::default()
    }
}
