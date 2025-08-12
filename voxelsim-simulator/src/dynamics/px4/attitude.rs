use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

pub struct AttitudeControl {
    proportional_gain: Vector3<f64>,
    yaw_w: f64,
    rate_limit: Vector3<f64>,
}

impl AttitudeControl {
    pub fn update(
        &self,
        q: UnitQuaternion<f64>,
        qd: UnitQuaternion<f64>,
        yawspeed_ff: f64,
    ) -> Vector3<f64> {
        let rm = q.to_rotation_matrix();
        let rm_d = qd.to_rotation_matrix();

        let rm_inv = rm.inverse();

        let e_z: Vector3<f64> = rm.into_inner().column(2).into();
        let e_z_d: Vector3<f64> = rm_d.into_inner().column(2).into();

        // QD is the rotation with the yaw stripped out.
        let mut qd_red = UnitQuaternion::rotation_between(&e_z, &e_z_d).unwrap();

        if qd_red[1].abs() > (1.0 - 1e-5) || qd_red[2].abs() > (1.0 - 1e-5) {
            qd_red = qd;
        } else {
            qd_red *= q;
        }

        let mut q_mix = (qd_red.inverse() * qd).into_inner();
        q_mix *= q_mix[0].signum();

        q_mix[0] = q_mix[0].clamp(-1.0, 1.0);
        q_mix[3] = q_mix[3].clamp(-1.0, 1.0);

        let qd = UnitQuaternion::from_quaternion(
            qd_red.into_inner()
                * Quaternion::new(
                    (self.yaw_w * q_mix[0].acos()).cos(),
                    0.0,
                    0.0,
                    (self.yaw_w * q_mix[3].asin()).sin(),
                ),
        );

        // Rotation from q to qd.
        let qe = q.inverse() * qd;

        let eq = 2.0 * qe[0].signum() * qe.imag();

        let mut rate_setpoint = eq.component_mul(&self.proportional_gain);

        let q_e_inv: Vector3<f64> = rm_inv.into_inner().column(2).into();

        rate_setpoint += q_e_inv * yawspeed_ff;

        // Limit rates.
        for i in 0..3 {
            rate_setpoint[i] = rate_setpoint[i].clamp(-self.rate_limit[i], self.rate_limit[i]);
        }

        rate_setpoint
    }
}

pub struct AttitudeSetpoint {
    pub q_d: UnitQuaternion<f64>,
    pub thrust_body: Vector3<f64>,
}

pub fn thrust_to_attitude(thr_sp: Vector3<f64>, yaw_sp: f64) -> AttitudeSetpoint {
    let yaw_body = yaw_sp;

    let body_z = if thr_sp.norm() > 0.0001 {
        -thr_sp.normalize()
    } else {
        Vector3::z_axis().into_inner()
    };

    // Vector of desired yaw direction.
    let y_c = Vector3::new(-yaw_body.sin(), yaw_body.cos(), 0.0);

    let body_x: Vector3<f64> = if body_z.z.abs() > 0.00001 {
        let mut bx = y_c.cross(&body_z);
        if body_z.z < 0.0 {
            bx = -bx;
        }

        bx.normalize()
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    };

    let body_y = body_z.cross(&body_x);

    let r_sp: Matrix3<f64> = Matrix3::from_columns(&[body_x, body_y, body_z]);

    let q_d = UnitQuaternion::from_matrix(&r_sp);

    AttitudeSetpoint {
        thrust_body: Vector3::new(0.0, 0.0, thr_sp.norm()),
        q_d,
    }
}
