use bitflags::bitflags;
use dashmap::DashMap;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use tinyvec::ArrayVec;

bitflags! {
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
    pub struct Cell: u32 {
        const DRONE_OCCUPIED    = 0b0000_0001;
        const DRONE_ADJACENT    = 0b0000_0010;
        const DRONE_HISTORIC    = 0b0000_0100;
        const DRONE_TRAJECTORY  = 0b0000_1000;
        const FILLED            = 0b0001_0000;
        const SPARSE            = 0b0010_0000;
        const GROUND            = 0b0100_0000;
        const TARGET            = 0b1000_0000;
    }

}

pub type Coord = Vector3<i32>;

pub type GridShell = [Vector3<i32>; 26];
pub type CollisionShell = ArrayVec<[(Coord, Cell); 26]>;

pub(crate) fn adjacent_coords(coord: Vector3<i32>) -> GridShell {
    let mut coords = [Vector3::zeros(); 26];
    let mut t = 0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                if i != 1 || j != 1 || k != 1 {
                    coords[t] = coord + Vector3::new(i - 1, j - 1, k - 1);
                    t += 1;
                }
            }
        }
    }
    coords
}

/// Returns `true` when the axis-aligned box centred at `pos` with full
/// extents `dims` touches or overlaps the unit cube whose centre is the
/// integer coordinate `coord`.
///
/// Touching faces/edges counts as an intersection; change the comparisons
/// from `>=` to `>` if you need strictly positive volume overlap.
pub fn intersects(coord: Vector3<i32>, pos: Vector3<f64>, dims: Vector3<f64>) -> bool {
    // --- Unit-cube bounds (½-extent is 0.5 on every axis) ------------------
    let cube_min = Vector3::new(
        coord.x as f64 - 0.5,
        coord.y as f64 - 0.5,
        coord.z as f64 - 0.5,
    );
    let cube_max = Vector3::new(
        coord.x as f64 + 0.5,
        coord.y as f64 + 0.5,
        coord.z as f64 + 0.5,
    );

    // --- Object bounds -----------------------------------------------------
    let half = dims * 0.5; // component-wise scalar multiply
    let obj_min = pos - half;
    let obj_max = pos + half;

    // --- AABB overlap test -------------------------------------------------
    obj_max.x >= cube_min.x
        && cube_max.x >= obj_min.x
        && obj_max.y >= cube_min.y
        && cube_max.y >= obj_min.y
        && obj_max.z >= cube_min.z
        && cube_max.z >= obj_min.z
}

pub(crate) fn within_bounds<N: PartialOrd>(v: Vector3<N>, b: Vector3<N>) -> bool {
    v.iter().zip(b.iter()).all(|(vi, bi)| vi <= bi)
}

// Dynamic height voxel grid.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VoxelGrid {
    cells: DashMap<Coord, Cell>,
}

impl VoxelGrid {
    pub fn new() -> Self {
        Self {
            cells: DashMap::new(),
        }
    }

    pub fn from_cells(cells: DashMap<Coord, Cell>) -> Self {
        Self { cells }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cells: DashMap::with_capacity(capacity),
        }
    }

    pub fn cull(&mut self, centre: Vector3<i32>, bounds: Vector3<i32>) {
        self.cells
            .retain(|k, _| within_bounds(Vector3::from(*k) - centre, bounds));
    }

    pub fn cells(&self) -> &DashMap<Coord, Cell> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut DashMap<Coord, Cell> {
        &mut self.cells
    }

    pub fn set(&mut self, coord: Coord, cell: Cell) {
        self.cells.insert(coord, cell);
    }

    pub fn remove(&mut self, coord: &Coord) -> Option<Cell> {
        self.cells.remove(coord).map(|x| x.1)
    }

    /// Returns a the list of cells if an object with the given centre coordinate and dimensions collides with any
    /// cells.
    pub fn collisions(&self, centre: Vector3<f64>, dims: Vector3<f64>) -> CollisionShell {
        let mut collisions = ArrayVec::new();
        // We only need to check the cubes around.
        assert!(dims.x < 1.0 && dims.y < 1.0 && dims.z < 1.0);
        for cell_coord in adjacent_coords(centre.try_cast::<i32>().unwrap()) {
            if let Some(cell) = self.cells().get(&cell_coord) {
                if intersects(cell_coord, centre, dims) {
                    collisions.push((cell_coord, *cell));
                }
            }
        }
        collisions
    }
}
