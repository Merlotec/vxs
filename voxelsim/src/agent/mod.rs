use nalgebra::{Rotation3, Unit, Vector3};
use serde::{Deserialize, Serialize};
use tinyvec::{array_vec, ArrayVec};

#[cfg(feature = "python")]
use pyo3::pyclass;

use crate::viewport::CameraView;

pub mod viewport;

const STABLE_VEL: f32 = 4.0;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AgentDynamics {
    // In terms of acceleration NOT force.
    pub air_resistance: f32,
    pub g: Vector3<f32>,
    pub thrust: f32,                 // Acelleration due to thrust.
    pub thrust_urgency_ceiling: f32, // Usually set to g.
    pub bounding_box: Vector3<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Agent {
    pub pos: Vector3<f32>,
    pub vel: Vector3<f32>,

    // Thrust is specified in units of ACCELERATION!
    // It is also assumed that if there is no change from previous action then the thrust will
    // remain the same.
    pub thrust: Vector3<f32>,

    pub id: usize,

    // Current action being processed by the agent.
    pub action: Option<Action>,
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[repr(i32)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub enum MoveDir {
    #[default]
    None = 0,
    Forward = 1,
    Back = 2,
    Left = 3,
    Right = 4,
    Up = 5,
    Down = 6,
    Undecided = 7,
}

impl MoveDir {
    pub fn dir_vec(&self) -> Option<Vector3<i32>> {
        match self {
            MoveDir::None => Some(Vector3::zeros()),
            MoveDir::Up => Some(Vector3::new(0, 1, 0)),
            MoveDir::Down => Some(Vector3::new(0, -1, 0)),
            MoveDir::Left => Some(Vector3::new(-1, 0, 0)),
            MoveDir::Right => Some(Vector3::new(1, 0, 0)),
            MoveDir::Forward => Some(Vector3::new(0, 0, -1)),
            MoveDir::Back => Some(Vector3::new(0, 0, 1)),
            MoveDir::Undecided => None,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct MoveCommand {
    pub dir: MoveDir,
    pub urgency: f32,
}

pub type CmdSequence = ArrayVec<[MoveCommand; MAX_ACTIONS]>;

/// When we carry out an action we need to pyo3ensure that we remove subsequent steps when we enter the
/// relevant voxels.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Action {
    pub cmd_sequence: CmdSequence,
    pub origin: Vector3<i32>,
}

pub const MAX_ACTIONS: usize = 6;

impl Agent {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
            thrust: Vector3::zeros(),
            action: None,
        }
    }

    pub fn camera(&self) -> CameraView {
        /// Tunables ─ feel free to move these to a `config.rs` or similar.
        const FOV_HORIZONTAL: f32 = 70.0_f32.to_radians();
        const FOV_VERTICAL: f32 = 40.0_f32.to_radians();
        const MAX_DISTANCE: f32 = 1_000.0;

        /// Maximum downward tilt (in radians) reached as thrust → ∞.
        const MAX_PITCH: f32 = 45.0_f32.to_radians();
        /// “Speed” of approach to `MAX_PITCH`.  Bigger → faster.
        const PITCH_RESPONSE: f32 = 0.75;
        // ------------------------------------------------------------------
        // 1. Horizontal orientation (yaw) – look where we’re thrusting.
        // ------------------------------------------------------------------
        let horiz = Vector3::new(self.thrust.x, 0.0, self.thrust.z);
        let forward_h = if horiz.magnitude_squared() > 1.0e-6 {
            Unit::new_normalize(horiz)
        } else {
            // Degeneracy: no thrust -> look along world −Z
            Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0))
        };

        // ------------------------------------------------------------------
        // 2. Downward tilt (pitch) – asymptotic to MAX_PITCH
        //    pitch(t) = MAX_PITCH · (1 − e^(−k·|t|))
        // ------------------------------------------------------------------
        let pitch = MAX_PITCH * (1.0 - (-PITCH_RESPONSE * self.thrust.magnitude()).exp());

        // Axis to pitch about = camera-right (world-up × forward).
        let world_up = Vector3::y_axis(); // Y is up
        let right = Unit::new_normalize(world_up.cross(&forward_h));

        // Rotate the horizontal-only forward vector downward by `pitch`.
        let forward = Rotation3::from_axis_angle(&right, -pitch) // negative ==> downward
                         * forward_h.into_inner();

        // Re-derive an exact orthonormal basis.
        let camera_right = right.into_inner(); // already unit
        let camera_up = camera_right.cross(&forward).normalize();

        CameraView {
            camera_pos: self.pos,    // assumed Point3<f32>
            camera_forward: forward, // Vector3<f32>
            camera_up,               // Vector3<f32>
            camera_right,            // Vector3<f32>
            fov_horizontal: FOV_HORIZONTAL,
            fov_vertical: FOV_VERTICAL,
            max_distance: MAX_DISTANCE,
        }
    }

    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.pos = Vector3::new(x, y, z);
    }

    pub fn get_position(&self) -> (f32, f32, f32) {
        (self.pos.x, self.pos.y, self.pos.z)
    }

    pub fn set_velocity(&mut self, x: f32, y: f32, z: f32) {
        self.vel = Vector3::new(x, y, z);
    }

    pub fn perform_sequence(&mut self, commands: Vec<MoveCommand>) {
        let mut cmd_sequence = ArrayVec::new();
        for cmd in commands.into_iter().take(MAX_ACTIONS) {
            cmd_sequence.push(cmd);
        }
        self.perform(cmd_sequence);
    }

    pub fn perform(&mut self, cmd_sequence: CmdSequence) {
        self.action = Some(Action {
            cmd_sequence,
            origin: self.pos.map(|e| e.round() as i32),
        })
    }

    /// Sets the current thrust according to the action set provided.
    fn next_thrust(&self, dynamics: &AgentDynamics) -> Vector3<f32> {
        if let Some(thrust) = self.action.as_ref().and_then(|x| {
            x.next_target_thrust(self.pos - x.origin.cast::<f32>(), self.vel, dynamics)
        }) {
            thrust
        } else {
            // We dont want to move - just thrust to oppose g (if we can).
            -dynamics.g.normalize() * f32::min(dynamics.g.norm(), dynamics.thrust)
        }
    }

    pub fn step(&mut self, dynamics: &AgentDynamics, delta: f32) {
        // Apply action first then check to see if we should remove.
        let thrust = self.next_thrust(dynamics);
        // Apply forces.
        self.thrust = thrust;
        // Air resistance.
        let air_res = dynamics.air_resistance * -self.vel;

        let net_acc = thrust + dynamics.g + air_res;
        self.pos += delta * self.vel + 0.5 * net_acc * delta * delta;
        self.vel += delta * net_acc;

        // Now we want to remove old commands if they have been 'passed'.
        if let Some(action) = &mut self.action {
            action.remove_expired_commands(self.pos)
        }
    }
}

impl Action {
    pub fn next_target_dir(&self, local_pos: Vector3<f32>) -> Option<Vector3<f32>> {
        let primary: Vector3<f32> = self.relative_centroid_pos(1)?.cast::<f32>() - local_pos;

        //return Some(primary);
        let target = if let Some(secondary) = self
            .relative_centroid_pos(2)
            .map(|x| x.cast::<f32>().normalize())
        {
            let w = (local_pos - primary).norm().sqrt().clamp(0.0, 1.0);
            let merged = w * primary + (1.0 - w) * secondary;
            merged
            //if let Some(tertiary) = self.relative_centroid_pos(3).map(|x| x.cast::<f32>()) {
            //    let skew = tertiary - secondary;
            //    let skew_w = ((local_pos - tertiary) * 0.25).norm().clamp(0.0, 0.5);
            //    (merged + skew_w * skew).normalize()
            //} else {
            //    // Weight the primary and secondary depending on the distance to the primary.
            //    merged
            //}
        } else {
            primary
        };
        Some(target)
    }

    pub fn next_target_thrust(
        &self,
        local_pos: Vector3<f32>,
        current_vel: Vector3<f32>,
        dynamics: &AgentDynamics,
    ) -> Option<Vector3<f32>> {
        let urgency = self.cmd_sequence.first()?.urgency;

        let next_dir = self.next_target_dir(local_pos)?.normalize();
        let next_vel = next_dir * urgency * STABLE_VEL * self.continuity(0.7)?;
        // This is our acceleration direction.agent.rs
        //let mut correction_dir = (next_dir - current_vel.normalize()).normalize();
        //if correction_dir.x.is_nan() || correction_dir.y.is_nan() || correction_dir.z.is_nan() {
        //    correction_dir = Vector3::zeros();
        //}
        //let correction_weight = (0.2 + urgency * 0.2).clamp(0.0, 1.0);
        //let acc_dir = correction_dir * correction_weight + next_dir * (1.0 - correction_weight);
        let acc_dir = next_vel - current_vel;
        //println!(
        //    "next_dir: {}. correction_dir: {}, correction_weight: {}",
        //    next_dir, correction_dir, correction_weight
        //);
        let power = dynamics.power(acc_dir, urgency)?;
        assert!(power <= 1.0);

        dynamics.thrust_force(acc_dir, power)
    }
    /// Calculates the continuity score, which is a measure of how consistent the future path is at
    /// our current period. This can be used to decide how much velocity we are targetting.
    pub fn continuity(&self, discount: f32) -> Option<f32> {
        let mut prev: f32 = 1.0;
        let mut total: f32 = 0.0;
        let target_dir = self.cmd_sequence.first()?.dir;
        for (i, cmd) in self.cmd_sequence.iter().enumerate() {
            let v = if cmd.dir == target_dir { 1.0 } else { 0.5 };
            let x = prev * v * discount.powi(i as i32);
            prev = x;
            total += x;
        }
        Some(total)
    }
    // Gets the relative position of the centroid of the grid at position t (unscaled) from the
    // centroid of the starting cell.
    pub fn relative_centroid_pos(&self, t: usize) -> Option<Vector3<i32>> {
        if self.cmd_sequence.len() >= t {
            Some(
                self.cmd_sequence
                    .iter()
                    .take(t)
                    .filter_map(|x| x.dir.dir_vec())
                    .sum(),
            )
        } else {
            None
        }
    }

    pub fn remove_expired_commands(&mut self, pos: Vector3<f32>) {
        // Check to see if we are closer to the next command than the current. If so, remove
        // current command.

        if let Some(c0) = self.relative_centroid_pos(1) {
            let od = self.origin.cast::<f32>() - pos;
            let c0abs = c0 + self.origin;
            let c0d = c0abs.cast::<f32>() - pos;

            // Check if the current position is closer to the current centroid than the origin
            if c0d.norm() < od.norm() {
                let _ = self.cmd_sequence.remove(0);
                self.origin = c0abs;
            }
        }
    }
}

impl AgentDynamics {
    pub fn max_net_thrust(&self, dir: Vector3<f32>) -> Option<(f32, Vector3<f32>)> {
        // Calculate thrust to cancel g.i
        let mut max_thrust = solve_thrust(dir, self.g, self.thrust, 1e-12);
        max_thrust.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        max_thrust.first().cloned()
    }

    pub fn power(&self, dir: Vector3<f32>, urgency: f32) -> Option<f32> {
        let (net_thrust, _thrust_vec) = self.max_net_thrust(dir)?;
        let u_cap = if net_thrust < self.thrust_urgency_ceiling {
            (self.thrust_urgency_ceiling - self.thrust) / self.thrust_urgency_ceiling
        } else {
            0.0
        };
        let power = (urgency + u_cap).max(1.0);
        Some(power)
    }

    pub fn thrust_force(&self, dir: Vector3<f32>, power: f32) -> Option<Vector3<f32>> {
        let t = self.thrust * power;
        let mut sols = solve_thrust(dir, self.g, t, 1e-12);
        sols.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (_acc, thrust_dir) = sols.last()?.clone();
        Some(thrust_dir * t)
    }
}

/// Solve  x and unit-vector y  from  x·z = t·y + g
///
/// Returns a Vec of (x, y) solutions. 0, 1 or 2 real solutions are possible.
/// Degenerate cases (t == 0 or z == 0) are handled explicitly.
fn solve_thrust(
    z: Vector3<f32>,
    g: Vector3<f32>,
    t: f32,
    tol: f32, // e.g. 1e-12
) -> ArrayVec<[(f32, Vector3<f32>); 2]> {
    const EPS: f32 = 1e-10;
    let g = -g;
    // ---- special cases ----------------------------------------------------
    if t.abs() < EPS {
        // Equation reduces to  x·z = -g
        if z.norm() < EPS {
            // z == 0  ⇒  need g == 0 to be solvable (underdetermined for x)
            if g.norm() < tol {
                // infinite x, y arbitrary unit; return empty to signal ambiguity
                return ArrayVec::new();
            } else {
                return ArrayVec::new(); // inconsistent
            }
        } else {
            // Need g collinear with z
            let proj = g.dot(&z) / z.norm_squared();
            let g_parallel = proj * z;
            if (g - g_parallel).norm() > tol {
                return ArrayVec::new(); // not collinear → no solution
            }
            let x = proj; // since x·z = g ⇒ x = |g|/|z|
            return array_vec!([(f32, Vector3<f32>); 2] => (x, Vector3::new(1.0, 0.0, 0.0)));
            // y arbitrary unit
        }
    }

    if z.norm() < EPS {
        // z == 0  ⇒  t·y = g  and y must be unit
        let y = g / t;
        if (y.norm() - 1.0).abs() < tol {
            // x is irrelevant (underdetermined); return single representative
            return array_vec!([(f32, Vector3<f32>); 2] => (0.0, y));
        } else {
            return ArrayVec::new();
        }
    }

    // ---- generic case -----------------------------------------------------
    let a = z.norm_squared();
    let b = 2.0 * z.dot(&g);
    let c = g.norm_squared() - t * t;

    let disc = b * b - 4.0 * a * c;
    if disc < -tol {
        return ArrayVec::new(); // no real roots
    }

    // Clamp tiny negative discriminants to zero to stabilise sqrt
    let sqrt_disc = (disc.max(0.0)).sqrt();
    let mut sols = ArrayVec::new();

    for &root in &[-b + sqrt_disc, -b - sqrt_disc] {
        let x = root / (2.0 * a);
        let y = (x * z + g) / t;
        if (y.norm() - 1.0).abs() < tol {
            sols.push((x, y));
        }
    }
    sols
}

//fn solve_thrust(
//    z: Vector3<f32>,
//    g: Vector3<f32>,
//    t: f32,
//    tol: f32, // e.g. 1e-12
//) -> ArrayVec<[(f32, Vector3<f32>); 2]> {
//    const EPS: f32 = 1e-10;
//
//    // ---- special cases ----------------------------------------------------
//    if t.abs() < EPS {
//        // Equation reduces to  x·z = -g
//        if z.norm() < EPS {
//            // z == 0  ⇒  need g == 0 to be solvable (underdetermined for x)
//            if g.norm() < tol {
//                // infinite x, y arbitrary unit; return empty to signal ambiguity
//                return ArrayVec::new();
//            } else {
//                return ArrayVec::new(); // inconsistent
//            }
//        } else {
//            // Need g collinear with z
//            let proj = g.dot(&z) / z.norm_squared();
//            let g_parallel = proj * z;
//            if (g - g_parallel).norm() > tol {
//                return ArrayVec::new(); // not collinear → no solution
//            }
//            let x = -proj; // since x·z = -g ⇒ x = -|g|/|z|
//            return array_vec!([(f32, Vector3<f32>); 2] => (x, Vector3::new(1.0, 0.0, 0.0)));
//            // y arbitrary unit
//        }
//    }
//
//    if z.norm() < EPS {
//        // z == 0  ⇒  t·y = -g  and y must be unit
//        let y = -g / t;
//        if (y.norm() - 1.0).abs() < tol {
//            // x is irrelevant (underdetermined); return single representative
//            return array_vec!([(f32, Vector3<f32>); 2] => (0.0, y));
//        } else {
//            return ArrayVec::new();
//        }
//    }
//
//    // ---- generic case -----------------------------------------------------
//    let a = z.norm_squared();
//    let b = 2.0 * z.dot(&g);
//    let c = g.norm_squared() - t * t;
//
//    let disc = b * b - 4.0 * a * c;
//    if disc < -tol {
//        return ArrayVec::new(); // no real roots
//    }
//
//    // Clamp tiny negative discriminants to zero to stabilise sqrt
//    let sqrt_disc = (disc.max(0.0)).sqrt();
//    let mut sols = ArrayVec::new();
//
//    for &root in &[-b + sqrt_disc, -b - sqrt_disc] {
//        let x = root / (2.0 * a);
//        let y = (x * z + g) / t;
//        if (y.norm() - 1.0).abs() < tol {
//            sols.push((x, y));
//        }
//    }
//    sols
//}
