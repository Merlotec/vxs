use nalgebra::Vector3;

pub struct AlphaFilter {
    alpha: f64,
    state: f64,
}

pub struct RateState {
    pub lp_filters_d: [AlphaFilter; 3],
    pub torque: Vector3<f64>,
    rate_prev_filtered: Vector3<f64>,
    rate_int: Vector3<f64>,
}

pub struct SaturationFlags {
    pos: Vector3<bool>, // Roll, pitch yaw.
    neg: Vector3<bool>,
}

impl AlphaFilter {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, state: 0.0 }
    }

    fn apply(&mut self, input: f64) -> f64 {
        self.state = self.alpha * input + (1.0 - self.alpha) * self.state;
        self.state
    }
}

impl RateState {
    pub fn apply_filters(input: Vector3<f64>, filters: &mut [AlphaFilter; 3]) -> Vector3<f64> {
        input.map_with_location(|i, _, v| filters[i].apply(v))
    }
}

pub struct RateControl {
    gain_p: Vector3<f64>,
    gain_i: Vector3<f64>,
    gain_d: Vector3<f64>,
    gain_ff: Vector3<f64>,
    lim_int: Vector3<f64>,
}

impl RateControl {
    pub fn update(
        &self,
        rate: Vector3<f64>,
        rate_sp: Vector3<f64>,
        mut state: RateState,
        saturation_flags: SaturationFlags,
        dt: f64,
    ) -> (Vector3<f64>, RateState) {
        let rate_error = rate_sp - rate;

        let rate_filtered = RateState::apply_filters(rate, &mut state.lp_filters_d);

        let rate_d = if dt > std::f64::EPSILON {
            (rate_filtered - state.rate_prev_filtered) / dt
        } else {
            Vector3::zeros()
        };

        // PID control with feed foward
        let torque = self.gain_p.component_mul(&rate_error) + state.rate_int
            - self.gain_d.component_mul(&rate_d)
            + self.gain_ff.component_mul(&rate_sp);

        state.rate_prev_filtered = rate_filtered;

        // Real PX4 only executes this if we have not landed.
        state = self.update_integral(state, rate_error, saturation_flags, dt);

        (torque, state)
    }

    pub fn update_integral(
        &self,
        mut state: RateState,
        mut rate_error: Vector3<f64>,
        saturation_flags: SaturationFlags,
        dt: f64,
    ) -> RateState {
        for i in 0..3 {
            if saturation_flags.pos[i] {
                rate_error[i] = rate_error[i].min(0.0)
            }

            if saturation_flags.neg[i] {
                rate_error[i] = rate_error[i].max(0.0);
            }

            // = 400 radians.
            let mut i_factor = rate_error[i] / 6.981317007977318;

            i_factor = f64::max(0.0, 1.0 - i_factor * i_factor);

            let rate_i = state.rate_int[i] + i_factor * self.gain_i[i] * rate_error[i] * dt;

            if rate_i.is_finite() {
                state.rate_int[i] = rate_i.clamp(-self.lim_int[i], self.lim_int[i]);
            }
        }

        state
    }
}
