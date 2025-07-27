use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

/// PID parameters for position controller
pub struct PosPIDParams {
    pub kp: Vector3<f64>,
    pub ki: Vector3<f64>,
    pub kd: Vector3<f64>,
}

/// Holds integrals and previous errors for PID
pub struct PosPIDState {
    pub pos_integral: Vector3<f64>,
    pub last_vel_error: Vector3<f64>,
}

/// PID parameters for rate controller
pub struct RatePIDParams {
    pub kp: Vector3<f64>,
    pub ki: Vector3<f64>,
    pub kd: Vector3<f64>,
    pub max_torque: Vector3<f64>,
    pub max_integral: Vector3<f64>,
}

/// Holds integrals and previous errors for PID
pub struct RatePIDState {
    pub integral: Vector3<f64>,
    pub last_error: Vector3<f64>,
}

impl Default for PosPIDParams {
    fn default() -> Self {
        PosPIDParams {
            // ArduPilot default gains for position P: 1.0, I: 0.0, D: 0.5
            kp: Vector3::new(1.0, 1.0, 1.0),
            ki: Vector3::new(0.0, 0.0, 0.0),
            kd: Vector3::new(0.5, 0.5, 0.5),
        }
    }
}

impl Default for PosPIDState {
    fn default() -> Self {
        PosPIDState {
            pos_integral: Vector3::zeros(),
            last_vel_error: Vector3::zeros(),
        }
    }
}

impl Default for RatePIDParams {
    fn default() -> Self {
        RatePIDParams {
            // roughly half of the ArduPilot defaults
            kp: Vector3::new(2.0, 2.0, 4.0),
            // quarter the I‐gain to slow down windup
            ki: Vector3::new(0.05, 0.05, 0.02),
            // bump up D slightly to help damp any residual oscillation
            kd: Vector3::new(0.02, 0.02, 0.01),

            // clamp torques to a realistic small‐quad range
            max_torque: Vector3::new(0.1, 0.1, 0.2),

            // prevent the integral term from growing unbounded
            max_integral: Vector3::new(0.2, 0.2, 0.1),
        }
    }
}

impl Default for RatePIDState {
    fn default() -> Self {
        RatePIDState {
            integral: Vector3::zeros(),
            last_error: Vector3::zeros(),
        }
    }
}

/// Compute desired acceleration from position/velocity targets
/// a_cmd = Kp_pos * pos_error + Ki_pos * ∫pos_error dt + Kd_pos * vel_error + a_tgt (feedforward)
pub fn compute_accel_cmd(
    p_act: Vector3<f64>,
    v_act: Vector3<f64>,
    p_tgt: Vector3<f64>,
    v_tgt: Vector3<f64>,
    a_tgt: Vector3<f64>,
    pid_params: &PosPIDParams,
    pid_state: &mut PosPIDState,
    dt: f64,
) -> Vector3<f64> {
    // Position error
    let pos_error = p_tgt - p_act;
    // Integrate position error
    pid_state.pos_integral += pos_error * dt;
    // Velocity error (desired minus actual)
    let vel_error = v_tgt - v_act;
    // PID terms
    let p_term = pid_params.kp.component_mul(&pos_error);
    let i_term = pid_params.ki.component_mul(&pid_state.pos_integral);
    let d_term = pid_params.kd.component_mul(&vel_error);
    // Save last velocity error
    pid_state.last_vel_error = vel_error;
    // Compute a_cmd
    p_term + i_term + d_term + a_tgt
}

/// Convert desired body orientation R_cmd and current orientation R_act
/// into body rate setpoint using small-angle approximation
pub fn attitude_to_bodyrate(
    r_cmd: &Rotation3<f64>,
    r_act: &Rotation3<f64>,
    yaw_rate_ff: f64,
) -> Vector3<f64> {
    // 1. Convert each rotation to a quaternion
    let q_cmd = UnitQuaternion::from_rotation_matrix(r_cmd);
    let q_act = UnitQuaternion::from_rotation_matrix(r_act);

    // 2. Compute the error quaternion
    //    (this encodes the “delta” rotation from actual → command)
    let q_err = q_cmd * q_act.inverse();

    // 3. scaled_axis() returns (axis * angle) as a Vector3
    let rot_vec = q_err.scaled_axis();

    // 4. Add yaw-rate feed-forward on the Z axis
    rot_vec + Vector3::new(0.0, 0.0, yaw_rate_ff)
}

/// Compute desired torque from rate error using PID
pub fn control_torque(
    rate_error: Vector3<f64>,
    pid_state: &mut RatePIDState,
    params: &RatePIDParams,
    dt: f64,
) -> Vector3<f64> {
    let p_term = params.kp.component_mul(&rate_error);
    pid_state.integral += rate_error * dt;
    let i_term = params.ki.component_mul(&pid_state.integral);
    let d_error = (rate_error - pid_state.last_error) / dt;
    let d_term = params.kd.component_mul(&d_error);
    pid_state.last_error = rate_error;
    p_term + i_term + d_term
}

pub fn build_body_rotation(z_b_cmd: &Vector3<f64>, yaw: f64) -> Matrix3<f64> {
    // 1) Ensure Z is unit length
    let z_b = z_b_cmd.normalize();

    // 2) Desired “course” X projected onto the horizontal plane
    let x_c = Vector3::new(yaw.cos(), yaw.sin(), 0.0);

    // 3) Compute body-Y = Z × Xc, then normalize
    let y_b = z_b.cross(&x_c).normalize();

    // 4) Body-X = Y × Z
    let x_b = y_b.cross(&z_b);

    // 5) Build rotation matrix: columns are Xb, Yb, Zb
    Matrix3::from_columns(&[x_b, y_b, z_b])
}
