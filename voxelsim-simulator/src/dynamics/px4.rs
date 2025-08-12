use nalgebra::{Vector2, Vector3};

/// PX4 position controller, emulating the necessary parts of the PX4's `PositionControl` class.
pub struct PositionControl {
    /// Corresponds to _gain_pos_p.
    gain_pos_p: Vector3<f64>,
    /// Corresponds to _lim_vel_horizontal.
    lim_vel_horizontal: f64,
    // Corresponsds to _constraints.
    constraints: CtrlConstraints,
}

pub struct CtrlConstraints {
    speed_up: f64,
    speed_down: f64,
}

pub struct PositionTarget {
    pos_sp: Vector3<f64>,
    vel_sp: Vector3<f64>,
}

impl PositionControl {
    pub fn position_controller(&self, pos: Vector3<f64>, target: PositionTarget) -> Vector3<f64> {
        // const Vector3f vel_sp_position = (_pos_sp - _pos).emult(_gain_pos_p);
        let vel_sp_position = (target.pos_sp - pos).component_mul(&self.gain_pos_p);

        // _vel_sp = vel_sp_position + _vel_sp;
        let mut vel_sp = vel_sp_position + target.vel_sp;

        // const Vector2f vel_sp_xy = ControlMath::constrainXY(Vector2f(vel_sp_position), Vector2f(_vel_sp - vel_sp_position), _lim_vel_horizontal);
        let vel_sp_xy = constrain_xy(
            vel_sp.xy(),
            (vel_sp - vel_sp_position).xy(),
            self.lim_vel_horizontal,
        );

        // _vel_sp(0) = vel_sp_xy(0);
        // _vel_sp(1) = vel_sp_xy(1);
        vel_sp.x = vel_sp_xy.x;
        vel_sp.y = vel_sp_xy.y;

        vel_sp.z = vel_sp
            .z
            .clamp(-self.constraints.speed_up, self.constraints.speed_down);

        vel_sp
    }
}

fn constrain_xy(vel_xy: Vector2<f64>, ff_xy: Vector2<f64>, max_xy: f64) -> Vector2<f64> {
    // Sum feedback + feedforward
    let combined = vel_xy + ff_xy;

    // Limit magnitude to max_xy
    let norm = combined.norm();
    if norm > max_xy {
        combined * (max_xy / norm)
    } else {
        combined
    }
}
