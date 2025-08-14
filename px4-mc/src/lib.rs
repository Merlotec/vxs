//! Thin Rust binding over a C++ static library that wraps PX4's
//! multicopter position/attitude/rate controllers to compute body
//! torque and thrust from target position/velocity.
//!
//! The C++ library is built with CMake from `px4-mc/cpp/px4_mc`.
//! Point `PX4_SRC_DIR` env var to a PX4-Autopilot checkout to use
//! the exact upstream source files.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Px4McInput {
    pos: [f32; 3],
    vel: [f32; 3],
    att_q: [f32; 4],
    rates: [f32; 3],
    target_pos: [f32; 3],
    target_vel: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Px4McOutput {
    torque: [f32; 3],
    thrust: f32,
}

// Opaque handle to the C++ controller aggregate
#[repr(C)]
pub struct Px4McOpaque {
    _private: [u8; 0],
}

extern "C" {
    fn px4_mc_create(dt: f32, mass: f32) -> *mut Px4McOpaque;
    fn px4_mc_destroy(ptr: *mut Px4McOpaque);
    fn px4_mc_update(ptr: *mut Px4McOpaque, input: *const Px4McInput, output: *mut Px4McOutput);

    fn px4_mc_set_att_gains(ptr: *mut Px4McOpaque, p: *const f32, yaw_weight: f32);
    fn px4_mc_set_att_rate_limit(ptr: *mut Px4McOpaque, rate_limit: *const f32);
    fn px4_mc_set_pos_gains(ptr: *mut Px4McOpaque, p: *const f32);
    fn px4_mc_set_vel_gains(ptr: *mut Px4McOpaque, p: *const f32, i: *const f32, d: *const f32);
    fn px4_mc_set_vel_limits(ptr: *mut Px4McOpaque, vel_horizontal: f32, vel_up: f32, vel_down: f32);
    fn px4_mc_set_thrust_limits(ptr: *mut Px4McOpaque, min_thr: f32, max_thr: f32);
    fn px4_mc_set_thr_xy_margin(ptr: *mut Px4McOpaque, margin: f32);
    fn px4_mc_set_tilt_limit(ptr: *mut Px4McOpaque, tilt_rad: f32);
    fn px4_mc_set_hover_thrust(ptr: *mut Px4McOpaque, hover: f32);
    fn px4_mc_set_rate_p(ptr: *mut Px4McOpaque, p: *const f32);
    fn px4_mc_set_dt(ptr: *mut Px4McOpaque, dt: f32);
    fn px4_mc_set_mass(ptr: *mut Px4McOpaque, mass: f32);
}

pub struct Px4McController {
    ptr: *mut Px4McOpaque,
}

unsafe impl Send for Px4McController {}
unsafe impl Sync for Px4McController {}

impl Drop for Px4McController {
    fn drop(&mut self) {
        unsafe { px4_mc_destroy(self.ptr) }
    }
}

impl Px4McController {
    pub fn new(dt: f32, mass: f32) -> Self {
        let ptr = unsafe { px4_mc_create(dt, mass) };
        assert!(!ptr.is_null(), "px4_mc_create returned null");
        Self { ptr }
    }

    pub fn update(
        &mut self,
        pos: &Vector3<f32>,
        vel: &Vector3<f32>,
        att: &UnitQuaternion<f32>,
        rates: &Vector3<f32>,
        target_pos: &Vector3<f32>,
        target_vel: &Vector3<f32>,
    ) -> (Vector3<f32>, f32) {
        let q: Quaternion<f32> = att.clone().into_inner();
        let input = Px4McInput {
            pos: [pos.x, pos.y, pos.z],
            vel: [vel.x, vel.y, vel.z],
            att_q: [q.w, q.i, q.j, q.k],
            rates: [rates.x, rates.y, rates.z],
            target_pos: [target_pos.x, target_pos.y, target_pos.z],
            target_vel: [target_vel.x, target_vel.y, target_vel.z],
        };

        let mut output = Px4McOutput::default();
        unsafe { px4_mc_update(self.ptr, &input, &mut output) };
        (
            Vector3::new(output.torque[0], output.torque[1], output.torque[2]),
            output.thrust,
        )
    }

    pub fn set_att_gains(&mut self, p: &Vector3<f32>, yaw_weight: f32) {
        let arr = [p.x, p.y, p.z];
        unsafe { px4_mc_set_att_gains(self.ptr, arr.as_ptr(), yaw_weight) }
    }

    pub fn set_att_rate_limit(&mut self, rate_limit: &Vector3<f32>) {
        let arr = [rate_limit.x, rate_limit.y, rate_limit.z];
        unsafe { px4_mc_set_att_rate_limit(self.ptr, arr.as_ptr()) }
    }

    pub fn set_pos_gains(&mut self, p: &Vector3<f32>) {
        let arr = [p.x, p.y, p.z];
        unsafe { px4_mc_set_pos_gains(self.ptr, arr.as_ptr()) }
    }

    pub fn set_vel_gains(&mut self, p: &Vector3<f32>, i: &Vector3<f32>, d: &Vector3<f32>) {
        let pa = [p.x, p.y, p.z];
        let ia = [i.x, i.y, i.z];
        let da = [d.x, d.y, d.z];
        unsafe { px4_mc_set_vel_gains(self.ptr, pa.as_ptr(), ia.as_ptr(), da.as_ptr()) }
    }

    pub fn set_vel_limits(&mut self, vel_horizontal: f32, vel_up: f32, vel_down: f32) {
        unsafe { px4_mc_set_vel_limits(self.ptr, vel_horizontal, vel_up, vel_down) }
    }

    pub fn set_thrust_limits(&mut self, min_thr: f32, max_thr: f32) {
        unsafe { px4_mc_set_thrust_limits(self.ptr, min_thr, max_thr) }
    }

    pub fn set_horizontal_thrust_margin(&mut self, margin: f32) {
        unsafe { px4_mc_set_thr_xy_margin(self.ptr, margin) }
    }

    pub fn set_tilt_limit(&mut self, tilt_rad: f32) {
        unsafe { px4_mc_set_tilt_limit(self.ptr, tilt_rad) }
    }

    pub fn set_hover_thrust(&mut self, hover: f32) {
        unsafe { px4_mc_set_hover_thrust(self.ptr, hover) }
    }

    pub fn set_rate_p(&mut self, p: &Vector3<f32>) {
        let arr = [p.x, p.y, p.z];
        unsafe { px4_mc_set_rate_p(self.ptr, arr.as_ptr()) }
    }

    pub fn set_dt(&mut self, dt: f32) { unsafe { px4_mc_set_dt(self.ptr, dt) } }
    pub fn set_mass(&mut self, mass: f32) { unsafe { px4_mc_set_mass(self.ptr, mass) } }
}
