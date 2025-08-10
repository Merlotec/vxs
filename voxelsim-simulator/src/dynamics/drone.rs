use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

/// PID parameters for position controller
#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Debug, Copy, Clone, PartialEq)]
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
            // Safer defaults to reduce aggressive reversal acceleration
            kp: Vector3::new(0.6, 0.6, 0.6),
            ki: Vector3::new(0.0, 0.0, 0.0),
            kd: Vector3::new(0.2, 0.2, 0.2),
        }
    }
}

impl PosPIDParams {
    pub fn default_moving() -> Self {
        PosPIDParams {
            // More conservative moving gains to avoid "shoot away" on direction flips
            kp: Vector3::new(1.2, 1.2, 1.2),
            ki: Vector3::new(0.01, 0.01, 0.01),
            kd: Vector3::new(0.6, 0.6, 0.6),
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
            // Slightly softer attitude rate gains for smoother response
            kp: Vector3::new(1.5, 1.5, 3.0),
            ki: Vector3::new(0.04, 0.04, 0.02),
            kd: Vector3::new(0.02, 0.02, 0.01),

            // Clamp torques to a realistic, conservative range
            max_torque: Vector3::new(0.1, 0.1, 0.2),

            // Prevent integral windup
            max_integral: Vector3::new(0.2, 0.2, 0.1),
        }
    }
}

impl RatePIDParams {
    pub fn default_moving() -> Self {
        RatePIDParams {
            // Softer moving gains for stability during aggressive maneuvers
            kp: Vector3::new(2.0, 2.0, 4.0),
            ki: Vector3::new(0.06, 0.06, 0.03),
            kd: Vector3::new(0.025, 0.025, 0.012),

            // Slightly higher torque limits for moving, still conservative
            max_torque: Vector3::new(0.12, 0.12, 0.24),

            // Integral clamp for moving case
            max_integral: Vector3::new(0.25, 0.25, 0.12),
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
