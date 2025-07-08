use nalgebra::Matrix3;

use crate::{Cell, Coord, VoxelGrid};
use std::collections::HashMap;
use std::time::SystemTime;

use super::*;

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ViewCell {
    pub time: SystemTime,
    pub cell: Cell,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ViewGrid {
    pub cells: HashMap<Coord, Cell>,
}

impl ViewGrid {
    pub fn view_world(&mut self, world: &VoxelGrid, camera_vp: Matrix3<f32>) {}
}
