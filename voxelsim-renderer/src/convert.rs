use std::f64::consts::FRAC_PI_2;

use bevy::math::{Quat, Vec3};
use nalgebra::{UnitQuaternion, Vector3};

pub fn client_to_bevy_i32(v: Vector3<i32>) -> Vec3 {
    Vec3::new(v.x as f32, v.z as f32, -v.y as f32)
}

pub fn client_to_bevy_f32(v: Vector3<f32>) -> Vec3 {
    Vec3::new(v.x, v.z, -v.y)
}

pub fn client_to_bevy_quat(q_zup: UnitQuaternion<f64>) -> Quat {
    // R = –90° around X
    let rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -FRAC_PI_2);

    // conjugate your original by R:
    let q_converted = rot * q_zup * rot.inverse();

    // down-cast to f32 and hand off to Bevy:
    let q32 = q_converted.cast::<f32>();
    Quat::from_xyzw(q32.i, q32.j, q32.k, q32.w)
}
