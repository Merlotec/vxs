use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, Vector3};
use nalgebra::{Matrix3, SymmetricEigen};
use tinyvec::{ArrayVec, array_vec};

use crate::Coord;

// Dense grid with phase vectors.
pub struct DynamicFrame {
    time: f64,
    cells: HashSet<Coord>,
}

#[derive(Default)]
pub struct PrincipalComponent {
    pub l: f64,
    pub u: Vector3<f64>,
    pub expl_var: f64,
}

pub struct PhaseFlow {
    sources: HashMap<Coord, ArrayVec<[PrincipalComponent; 3]>>,
}

pub struct PhaseGrid {
    cells: HashSet<Coord, f64>,
}

impl PhaseFlow {
    pub fn from_frames(mut frames: Vec<DynamicFrame>) -> Self {
        let present = frames.pop().unwrap();

        let mut sources: HashMap<Coord, ArrayVec<[PrincipalComponent; 3]>> = HashMap::new();

        let t_end = present.time;

        let threshold = 3.0;
        let lambda_threshold = 2.0;

        let (points, base_weights): (Vec<Vector3<f64>>, Vec<f64>) = frames
            .iter()
            .filter_map(|f| {
                let v = threshold - (t_end - f.time);
                if v > 0.0 {
                    let w = v / threshold;
                    Some(
                        f.cells
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

        for src in present.cells.iter() {
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
                        if let Some(pcs) = sources.get_mut(src) {
                            pcs.push(pc);
                        } else {
                            sources.insert(*src, array_vec!([PrincipalComponent; 3] => pc));
                        }
                    }
                }
            }
        }

        Self { sources }
    }

    // pub fn build_grid(&self, t: f64) -> PhaseGrid {
    //     for (coord, pcs) in self.sources.iter() {

    //     }
    // }
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
