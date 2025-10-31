use std::collections::HashSet;

use nalgebra::Vector3;

use crate::Coord;

// Dense grid with phase vectors.
pub struct PhaseGrid {
    sources: HashSet<Coord>,
    sinks: HashSet<Coord>,

    cells: ndarray::Array3<Vector3<f64>>,
}
