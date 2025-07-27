use dashmap::DashMap;
use nalgebra::{Matrix4, Perspective3, Point3, Vector2, Vector3};

use crate::{Cell, Coord};

use super::*;

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all, set_all))]
pub struct CameraProjection {
    pub aspect: f64,
    pub fov_vertical: f64,
    pub max_distance: f64,
    pub near_distance: f64,
}

impl CameraProjection {
    pub fn projection_matrix(&self) -> Matrix4<f64> {
        Perspective3::new(
            self.aspect,        // width/height
            self.fov_vertical,  // vertical FOV (radians)
            self.near_distance, // near plane
            self.max_distance,  // far plane
        )
        .to_homogeneous()
    }
}

impl Default for CameraProjection {
    fn default() -> Self {
        Self {
            aspect: 1.4,
            fov_vertical: 0.8,
            max_distance: 1_000.0,
            near_distance: 0.5,
        }
    }
}

pub type VirtualCollisionShell = ArrayVec<[(Coord, VirtualCell); 26]>;

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VirtualCell {
    pub cell: Cell,
    pub uncertainty: f64,
}

impl Default for VirtualCell {
    fn default() -> Self {
        Self {
            cell: Cell::default(),
            uncertainty: Default::default(),
        }
    }
}

impl VirtualCell {
    /// Calculate priority for merging decisions
    /// Higher priority = better/more reliable data
    pub fn priority(&self) -> f64 {
        // Priority is inverse of uncertainty, plus bonus for recent data
        let uncertainty_priority = if self.uncertainty > 0.0 {
            1.0 / self.uncertainty
        } else {
            1000.0 // Very high priority for zero uncertainty
        };

        uncertainty_priority.min(1000.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VirtualGrid {
    pub cells: DashMap<Coord, VirtualCell>,
}

impl VirtualGrid {
    pub fn new() -> Self {
        Self {
            cells: DashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cells: DashMap::with_capacity(capacity),
        }
    }

    pub fn create_block(
        scale: usize,
        pos: Coord,
        centre_cell: VirtualCell,
    ) -> DashMap<Coord, VirtualCell> {
        let block_cells = DashMap::new();
        // Calculate the half-extent of the cube
        let half_scale = (scale as i32) / 2;

        // Create a scale × scale × scale cube centered around pos
        for dx in -(half_scale)..(half_scale + (scale as i32) % 2) {
            for dy in -(half_scale)..(half_scale + (scale as i32) % 2) {
                for dz in -(half_scale)..(half_scale + (scale as i32) % 2) {
                    let cube_coord = pos + Vector3::new(dx, dy, dz);

                    block_cells.insert(cube_coord, centre_cell.clone());
                }
            }
        }

        block_cells
    }

    pub fn cull(&mut self, centre: Vector3<i32>, bounds: Vector3<i32>) {
        self.cells
            .retain(|k, _| crate::env::within_bounds(Vector3::from(*k) - centre, bounds));
    }

    pub fn cells(&self) -> &DashMap<Coord, VirtualCell> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut DashMap<Coord, VirtualCell> {
        &mut self.cells
    }

    pub fn set(&mut self, coord: Coord, cell: VirtualCell) {
        self.cells.insert(coord, cell);
    }

    pub fn remove(&mut self, coord: &Coord) -> Option<VirtualCell> {
        self.cells.remove(coord).map(|x| x.1)
    }

    /// Returns a the list of cells if an object with the given centre coordinate and dimensions collides with any
    /// cells.
    pub fn collisions(&self, centre: Vector3<f64>, dims: Vector3<f64>) -> VirtualCollisionShell {
        let mut collisions = ArrayVec::new();
        // We only need to check the cubes around.
        assert!(dims.x < 1.0 && dims.y < 1.0 && dims.z < 1.0);
        for cell_coord in crate::env::adjacent_coords(centre.map(|e| e.round() as i32)) {
            if let Some(cell) = self.cells().get(&cell_coord) {
                if crate::env::intersects(cell_coord, centre, dims) {
                    collisions.push((cell_coord, *cell));
                }
            }
        }
        collisions
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct CameraView {
    /// World‐space camera position, as (X, Y, Z) with Z = up
    pub camera_pos: Vector3<f64>,
    /// Forward direction, in the world X–Y plane (i.e. camera looks “along” +Y by default)
    pub camera_forward: Vector3<f64>,
    /// Up direction → should now be +Z
    pub camera_up: Vector3<f64>,
    /// Right direction → +X
    pub camera_right: Vector3<f64>,
}

impl CameraView {
    pub fn new(
        camera_pos: Vector3<f64>,
        camera_forward: Vector3<f64>,
        camera_up: Vector3<f64>,
        camera_right: Vector3<f64>,
    ) -> Self {
        Self {
            camera_pos,
            camera_forward,
            camera_up,
            camera_right,
        }
    }

    pub fn view_matrix(&self) -> Matrix4<f64> {
        Matrix4::look_at_rh(
            &Point3::from(self.camera_pos),                       // eye
            &Point3::from(self.camera_pos + self.camera_forward), // target
            &self.camera_up,                                      // up = +Z
        )
    }

    /// Calculate distance from camera to a coordinate.
    /// We assume Coord is (x, y, z) but now treat z as the vertical axis.
    pub fn distance_to(&self, coord: Coord) -> f64 {
        // Swap Y↔Z so that coord.z is our “height”
        let pos = coord.cast::<f64>();
        (pos - self.camera_pos).norm()
    }
}

impl Default for CameraView {
    fn default() -> Self {
        // looking along +Y, with +Z up, +X right
        Self {
            camera_pos: Vector3::new(0.0, 0.0, 0.0),
            camera_forward: Vector3::new(0.0, 1.0, 0.0),
            camera_up: Vector3::new(0.0, 0.0, 1.0),
            camera_right: Vector3::new(1.0, 0.0, 0.0),
        }
    }
}
