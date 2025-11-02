use dashmap::DashMap;
use nalgebra::Vector3;
use nalgebra::{Matrix3, SymmetricEigen};
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use tinyvec::{ArrayVec, array_vec};

use crate::{Coord, VoxelSet};

#[derive(Default)]
pub struct PrincipalComponent {
    pub l: f64,
    pub u: Vector3<f64>,
    pub expl_var: f64,
}

pub struct PhaseFlow {
    sources: DashMap<Coord, ArrayVec<[PrincipalComponent; 3]>>,
}

pub struct PhaseGrid {
    pub cells: DashMap<Coord, f64>,
}

impl PhaseFlow {
    pub fn from_frames(mut frames: Vec<(VoxelSet, f64)>) -> Self {
        let (present, t_end) = frames.pop().unwrap();

        let sources: DashMap<Coord, ArrayVec<[PrincipalComponent; 3]>> = DashMap::new();

        let threshold = 3.0;
        let lambda_threshold = 2.0;

        let (points, base_weights): (Vec<Vector3<f64>>, Vec<f64>) = frames
            .iter()
            .filter_map(|f| {
                let v = threshold - (t_end - f.1);
                if v > 0.0 {
                    let w = v / threshold;
                    Some(
                        f.0.cells()
                            .iter()
                            .map(|x| (x.cast::<f64>(), w))
                            .collect::<Vec<(Vector3<f64>, f64)>>(),
                    )
                } else {
                    None
                }
            })
            .flatten()
            .unzip();

        present.cells().par_iter().for_each(|src| {
            let s: Vector3<f64> = src.cast();
            let weights: Vec<f64> = base_weights
                .iter()
                .enumerate()
                .map(|(i, w)| w / points[i].metric_distance(&s))
                .collect();
            if let Some((m, lambda)) = weighted_pca3_with_centroid(&points, &weights, src.cast()) {
                let lsum: f64 = lambda.iter().sum();
                for (i, l) in lambda.iter().enumerate() {
                    if l > &lambda_threshold {
                        let expl_var = l / lsum;
                        let pc = PrincipalComponent {
                            l: *l,
                            u: m.column(i).into_owned(),
                            expl_var,
                        };
                        if let Some(mut pcs) = sources.get_mut(&src) {
                            pcs.push(pc);
                        } else {
                            sources.insert(*src, array_vec!([PrincipalComponent; 3] => pc));
                        }
                    }
                }
            }
        });

        Self { sources }
    }

    pub fn build_grid(&self, t: f64) -> PhaseGrid {
        // Accumulate contributions from each source along its principal components
        // into a scalar field over voxel coordinates.
        let cells: DashMap<Coord, f64> = DashMap::new();

        if t <= 0.0 {
            return PhaseGrid { cells };
        }

        // Tuning constants for cone marching
        let step: f64 = 0.5; // march step along the ray (in voxel units)
        let base_radius: f64 = 0.5; // initial radius near the source
        let radius_slope: f64 = 0.45; // how fast the cone opens with distance

        self.sources.par_iter().for_each(|arg| {
            let (coord, pcs) = (arg.key(), arg.value());
            let src = coord.cast::<f64>();

            for pc in pcs.iter() {
                if pc.l <= 0.0 {
                    continue;
                }

                // Length scales linearly with t and sqrt(l)
                let length = t * pc.l.sqrt();
                if length <= 0.0 {
                    continue;
                }

                let u = pc.u;
                let norm = u.norm();
                if norm == 0.0 {
                    continue;
                }
                let dir = u / norm; // unit direction

                // Strength scales with explained variance; clamp to [0,1]
                let base_strength = pc.expl_var.clamp(0.0, 1.0);

                // March forward along the ray, widening the cone
                let mut s = 0.0;
                while s <= length {
                    let center = src + dir * s;
                    let radius = base_radius + radius_slope * s;

                    // Compute bounds for integer voxels to consider around this slice
                    let min = (center - Vector3::new(radius, radius, radius))
                        .map(|v| v.floor() as i32 - 1);
                    let max = (center + Vector3::new(radius, radius, radius))
                        .map(|v| v.ceil() as i32 + 1);

                    for x in min.x..=max.x {
                        for y in min.y..=max.y {
                            for z in min.z..=max.z {
                                let c = Vector3::new(x, y, z);
                                let p = c.cast::<f64>();
                                // Radial distance from the cone axis at this slice
                                let dist = (p - center).norm();
                                if dist <= radius {
                                    // Radial falloff inside the cone (1 at axis → 0 at boundary)
                                    let radial = 1.0 - (dist / (radius + 1e-9));
                                    // Slightly sharpen towards axis
                                    let radial = radial.clamp(0.0, 1.0).powf(2.0);

                                    // Contribution at this voxel slice
                                    let delta = base_strength * radial;

                                    // Speed-of-light style accumulation: approach but never reach 1.0
                                    let mut entry = cells.entry(c).or_insert(0.0);
                                    *entry = *entry + (1.0 - *entry) * delta;
                                }
                            }
                        }
                    }

                    s += step;
                }
            }
        });

        PhaseGrid { cells }
    }
}

/// Weighted PCA in 3D *around a given centroid*.
///
/// Inputs:
/// - `points`: data points
/// - `weights`: point weights (>=0)
/// - `centroid`: the centroid to subtract (user provided)
///
/// Outputs:
/// - `components`: 3×3 matrix whose columns are PC1, PC2, PC3 (descending variance)
/// - `variances`: eigenvalues for those components
pub fn weighted_pca3_with_centroid(
    points: &[Vector3<f64>],
    weights: &[f64],
    centroid: Vector3<f64>,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if points.len() != weights.len() || points.is_empty() {
        return None;
    }

    // 1) Weighted scatter matrix around the given centroid
    let mut sum_w = 0.0;
    let mut c = Matrix3::<f64>::zeros();

    for (p, &w) in points.iter().zip(weights) {
        if w <= 0.0 {
            continue;
        }
        let r = p - centroid;
        c += (r * r.transpose()) * w;
        sum_w += w;
    }

    if sum_w == 0.0 {
        return None;
    }

    c /= sum_w; // weighted covariance matrix (up to scale)

    // 2) Eigen decomposition (symmetric ⇒ real)
    let eig = SymmetricEigen::new(c);

    // nalgebra returns eigenvalues ascending; we want descending
    let idx = [2, 1, 0];

    let mut components = Matrix3::<f64>::zeros();
    let mut variances = Vector3::<f64>::zeros();

    for (j, &k) in idx.iter().enumerate() {
        let v = eig.eigenvectors.column(k);
        let vn = v.norm();
        if vn == 0.0 {
            return None;
        }
        components.set_column(j, &(v / vn)); // PC j
        variances[j] = eig.eigenvalues[k]; // λ j
    }

    Some((components, variances))
}
