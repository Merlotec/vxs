use std::time::Duration;

use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::quasinewton::LBFGS;
use bspline::BSpline;
use finitediff::FiniteDiff;
use nalgebra::Vector3;

use crate::Coord;

//// Signed distance to the unit cube centered at `coord`
/// Uses SDF for axis-aligned box of side length 1.
#[inline(always)]
pub fn sdf(coord: &Coord, p: &Vector3<f64>) -> f64 {
    // distance from p to centered box half-size = 0.5
    let c = coord.cast::<f64>();
    let d = (p - c).map(|x| x.abs() - 0.5);
    let outside_dist = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
    outside_dist.norm()
}
/// Build a clamped uniform knot vector for degree `p`
// fn uniform_knots(num_ctrl: usize, degree: usize) -> Vec<f64> {
//     let n = num_ctrl + degree + 1;
//     (0..n)
//         .map(|i| {
//             if i < degree {
//                 0.0
//             } else if i > num_ctrl {
//                 1.0
//             } else {
//                 (i - degree) as f64 / (num_ctrl - degree) as f64
//             }
//         })
//         .collect()
// }

fn uniform_knots(num_ctrl: usize, degree: usize) -> Vec<f64> {
    // open-clamped: (p+1) zeros, (n_ctrl-p-1) interior, (p+1) ones
    let mut knots = Vec::with_capacity(num_ctrl + degree + 1);
    // start clamp
    for _ in 0..=degree {
        knots.push(0.0);
    }
    // interior
    if num_ctrl > degree + 1 {
        let interior = num_ctrl - degree - 1;
        for i in 1..=interior {
            knots.push((i as f64) / ((interior + 1) as f64));
        }
    }
    // end clamp
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

struct SplineCost {
    degree: usize,
    knots: Vec<f64>,
    start: Vector3<f64>,
    end: Vector3<f64>,
    cells: Vec<Coord>,
    lambda_inside: f64,
    lambda_out: f64,
    smooth: f64,
    samples_per_seg: usize,
}

impl CostFunction for SplineCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        // Rebuild full control-point list: [start] + interiors + [end]
        let mut cps = Vec::with_capacity(x.len() / 3 + 2);
        cps.push(self.start);
        for v in x.chunks(3) {
            cps.push(Vector3::new(v[0], v[1], v[2]));
        }
        cps.push(self.end);

        let spline = BSpline::new(self.degree, cps.clone(), self.knots.clone());
        let (t0, t1) = spline.knot_domain();

        let mut cost = 0.0;
        // 1) Attraction & penalty (interior sampling)
        let m = self.cells.len() as f64;
        for (j, cell) in self.cells.iter().enumerate() {
            let t = t0 + ((j as f64 + 1.0) / (m + 1.0)) * (t1 - t0);
            let p = spline.point(t);
            let err = (p - cell.cast::<f64>()).norm_squared();
            let sd = sdf(&cell, &p);
            cost += self.lambda_inside * err + self.lambda_out * sd.powi(2);
        }

        // 2) Smoothness via second finite difference
        let h = 1e-3;
        let total_samples = self.samples_per_seg * (cps.len().saturating_sub(self.degree));
        let dt = (t1 - t0) / (total_samples as f64);
        for k in 0..=total_samples {
            let t = t0 + k as f64 * dt;
            let t_lo = (t - h).max(t0);
            let t_hi = (t + h).min(t1);
            let p_lo = spline.point(t_lo);
            let p = spline.point(t);
            let p_hi = spline.point(t_hi);
            let d2 = (p_hi - 2.0 * p + p_lo) * (1.0 / (h * h));
            cost += self.smooth * d2.norm_squared();
        }

        Ok(cost)
    }
}

impl Gradient for SplineCost {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Error> {
        let f = |v: &Vec<f64>| self.cost(v).unwrap();
        Ok(x.central_diff(&f))
    }
}

pub fn generate_centroid_fit_spline(
    start: Vector3<f64>,
    cells: &[Coord],
) -> BSpline<Vector3<f64>, f64> {
    // build the list of control points: the start + each cell’s centroid
    let control_points: Vec<_> = std::iter::once(start)
        .chain(cells.iter().map(|c| c.cast::<f64>()))
        .collect();

    // pick degree = 3, but no higher than n-1, and at least 1
    let degree = std::cmp::min(3, control_points.len().saturating_sub(1)).max(1);

    // uniform, clamped knot vector
    let knots = uniform_knots(control_points.len(), degree);

    // build and return the spline
    BSpline::new(degree, control_points, knots)
}

/// Generate an optimised B-spline through the integer grid cells
pub fn generate_spline(start: Vector3<f64>, cells: &[Coord]) -> BSpline<Vector3<f64>, f64> {
    // Build full initial control points: [start] + cell centroids
    let full_pts: Vec<_> = std::iter::once(start)
        .chain(cells.iter().map(|c| c.cast::<f64>()))
        .collect();

    // Degree = min(3, n-1) but >= 1
    let degree = (3).min(full_pts.len().saturating_sub(1)).max(1);
    let knots = uniform_knots(full_pts.len(), degree);

    // Special-case 2-point = straight line
    if full_pts.len() == 2 {
        let knots = uniform_knots(full_pts.len(), 0);
        return BSpline::new(0, full_pts, knots);
    }

    // Extract fixed endpoints and interior initial guesses
    let start_pt = full_pts.first().cloned().unwrap();
    let end_pt = full_pts.last().cloned().unwrap();
    let interior = &full_pts[1..full_pts.len() - 1];
    let init_param: Vec<f64> = interior.iter().flat_map(|p| vec![p.x, p.y, p.z]).collect();

    // Set up cost and solver
    let cost = SplineCost {
        degree,
        knots: knots.clone(),
        start: start_pt,
        end: end_pt,
        cells: cells.to_vec(),
        lambda_inside: 1.0,
        lambda_out: 10.0,
        smooth: 0.01,
        samples_per_seg: 4,
    };
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(1e-8).unwrap())
        .rho(0.5)
        .unwrap();
    let solver = LBFGS::new(linesearch, 100);
    let result = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_param.clone())
                .max_iters(300)
                .target_cost(1e-8)
        })
        .timeout(Duration::from_millis(10))
        .run()
        .unwrap();

    // Reconstruct final control-points
    let opt_vec = result.state.param.unwrap();
    let mut final_cps = Vec::with_capacity(opt_vec.len() / 3 + 2);
    final_cps.push(start_pt);
    for v in opt_vec.chunks(3) {
        final_cps.push(Vector3::new(v[0], v[1], v[2]));
    }
    final_cps.push(end_pt);

    BSpline::new(degree, final_cps, knots)
}

/// Lightweight wrapper for runtime evaluation
#[derive(Debug, Clone)]
pub struct Trajectory {
    spline: BSpline<Vector3<f64>, f64>,
    urgencies: Vec<f64>,
    yaw_seq: Vec<f64>,
    pub progress: f64,
}

impl Default for Trajectory {
    fn default() -> Self {
        Self {
            spline: BSpline::new(0, Vec::new(), Vec::new()),
            urgencies: Vec::new(),
            yaw_seq: Vec::new(),
            progress: 0.0,
        }
    }
}

impl Trajectory {
    /// Generates a spline through the grid path for target times specified in the tuple (coord, time).
    pub fn generate(start: Vector3<f64>, cells: &[(Coord, f64)]) -> Self {
        let (cells, times): (Vec<Coord>, Vec<f64>) = cells.to_vec().into_iter().unzip();
        Self {
            spline: generate_spline(start, &cells),
            urgencies: times,
            yaw_seq: Vec::new(),
            progress: 0.0,
        }
    }

    pub fn inner(&self) -> &BSpline<Vector3<f64>, f64> {
        &self.spline
    }

    pub fn inner_mut(&mut self) -> &mut BSpline<Vector3<f64>, f64> {
        &mut self.spline
    }

    #[inline(always)]
    pub fn position(&self, t: f64) -> Option<Vector3<f64>> {
        Some(self.inner().point(self.domain_t(t)?))
    }

    #[inline(always)]
    pub fn velocity(&self, t: f64) -> Option<Vector3<f64>> {
        let h = 1e-3;
        let t_lo = (t - h).max(0.0);
        let t_hi = (t + h).min(1.0);
        Some((self.position(t_hi)? - self.position(t_lo)?) * (1.0 / (t_hi - t_lo)))
    }

    #[inline(always)]
    pub fn acceleration(&self, t: f64) -> Option<Vector3<f64>> {
        let h = 1e-3;
        let t_lo = (t - h).max(0.0);
        let t_hi = (t + h).min(0.0);
        Some(
            (self.position(t_hi)? - 2.0 * self.position(t)? + self.position(t_lo)?)
                * (1.0 / (h * h)),
        )
    }

    /// Returns (position, t) of the waypoint.
    #[inline(always)]
    pub fn waypoints(&self, splits: usize) -> Vec<(f64, Vector3<f64>)> {
        let (min, max) = self.inner().knot_domain();
        let dsplit = (max - min) / splits as f64;

        (0..splits)
            .map(|i| {
                let t = min + dsplit * i as f64;
                (t, self.inner().point(min + dsplit * i as f64))
            })
            .collect()
    }

    pub fn urgencies(&self) -> &[f64] {
        &self.urgencies
    }

    pub fn yaw_seq(&self) -> &[f64] {
        &self.yaw_seq
    }

    pub fn len(&self) -> f64 {
        self.urgencies.len() as f64
    }

    pub fn domain_t(&self, t: f64) -> Option<f64> {
        let (min, max) = self.spline.knot_domain();
        let dom_t = min + (max - min) * t;

        if dom_t <= max { Some(dom_t) } else { None }
    }

    pub fn sample_urgencies(&self, t: f64) -> Option<f64> {
        if let Some(floor) = self.urgencies.get(t.floor() as usize) {
            if let Some(ceil) = self.urgencies.get(t.ceil() as usize) {
                let w = t - floor;
                let weighted_urgency = floor * w + ceil * (1.0 - w);
                Some(weighted_urgency)
            } else {
                Some(*floor)
            }
        } else {
            None
        }
    }

    pub fn sample_yaw(&self, t: f64) -> Option<f64> {
        if let Some(floor) = self.yaw_seq.get(t.floor() as usize) {
            if let Some(ceil) = self.yaw_seq.get(t.ceil() as usize) {
                Some(*ceil)
            } else {
                Some(*floor)
            }
        } else {
            None
        }
    }

    pub fn sample_spline(&self, t: f64) -> Option<Vector3<f64>> {
        let dom_t = self.domain_t(t)?;
        Some(self.spline.point(dom_t))
    }
    pub fn sample(&self, t: f64) -> Option<(Vector3<f64>, f64)> {
        let p = self.inner().point(t);
        let u = self.sample_urgencies(t)?;
        Some((p, u))
    }

    pub fn find_nearest_param_in_range(
        &self,
        start: f64,
        end: f64,
        point: &Vector3<f64>,
    ) -> Option<f64> {
        // Coarse uniform sampling to find initial guess
        let samples = 100;
        let mut best_t = start;
        let mut best_dist2 = f64::MAX;
        for i in 0..=samples {
            let t = start + (end - start) * (i as f64) / (samples as f64);
            let p = self.position(t)?;
            let d2 = (p - point).norm_squared();
            if d2 < best_dist2 {
                best_dist2 = d2;
                best_t = t;
            }
        }

        // Refine with a ternary search around best_t
        let delta = (end - start) / (samples as f64);
        let mut left = (best_t - delta).max(start);
        let mut right = (best_t + delta).min(end);
        for _ in 0..10 {
            let t1 = left + (right - left) / 3.0;
            let t2 = right - (right - left) / 3.0;
            let d1 = (self.position(t1)? - point).norm_squared();
            let d2 = (self.position(t2)? - point).norm_squared();
            if d1 < d2 {
                right = t2;
            } else {
                left = t1;
            }
        }

        // Return midpoint of final interval
        Some((left + right) / 2.0)
    }
}

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Internal plain‐old‐data for Serde round‐trips
#[derive(Serialize, Deserialize)]
struct SplineData {
    degree: usize,
    knots: Vec<f64>,
    ctrl_pts: Vec<[f64; 3]>,
    // In knot domain.
    urgencies: Vec<f64>,
    yaw_seq: Vec<f64>,
    // In knot domain.
    progress: f64,
}

impl Serialize for Trajectory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let spline = &self.inner();

        // Extract all data
        let ctrl_pts: Vec<[f64; 3]> = spline.control_points().map(|p| [p.x, p.y, p.z]).collect();
        let knots: Vec<f64> = spline.knots().map(|x| *x).collect();

        // Recover degree = knots.len() - ctrl_pts.len() - 1
        let n_knots = knots.len();
        let n_ctrl = ctrl_pts.len();
        let degree = n_knots.checked_sub(n_ctrl + 1).ok_or_else(|| {
            serde::ser::Error::custom("Invalid BSpline: knot/control-point count mismatch")
        })?;

        let data = SplineData {
            degree,
            knots,
            ctrl_pts,
            urgencies: self.urgencies.clone(),
            yaw_seq: self.yaw_seq.clone(),
            progress: self.progress,
        };
        data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Trajectory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = SplineData::deserialize(deserializer)?;
        // Rebuild control points
        let cps: Vec<_> = data
            .ctrl_pts
            .into_iter()
            .map(|[x, y, z]| Vector3::new(x, y, z))
            .collect();
        // Validate length: should be cps.len() + degree + 1 == knots.len()
        if data.knots.len() != cps.len() + data.degree + 1 {
            return Err(serde::de::Error::custom(
                "Invalid knot/control‐point lengths",
            ));
        }
        Ok(Trajectory {
            spline: BSpline::new(data.degree, cps, data.knots),
            urgencies: data.urgencies,
            yaw_seq: data.yaw_seq,
            progress: data.progress,
        })
    }
}
