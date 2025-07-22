use dashmap::DashMap;
use nalgebra::{Matrix4, Perspective3, Point3, Vector2, Vector3};
use rayon::prelude::*;

use crate::{Cell, Coord, VoxelGrid};
use std::collections::HashMap;

use super::*;

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all, set_all))]
pub struct CameraProjection {
    pub aspect: f32,
    pub fov_vertical: f32,
    pub max_distance: f32,
    pub near_distance: f32,
}

impl CameraProjection {
    pub fn projection_matrix(&self) -> Matrix4<f32> {
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
    pub uncertainty: f32,
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
    pub fn priority(&self) -> f32 {
        // Priority is inverse of uncertainty, plus bonus for recent data
        let uncertainty_priority = if self.uncertainty > 0.0 {
            1.0 / self.uncertainty
        } else {
            1000.0 // Very high priority for zero uncertainty
        };

        uncertainty_priority.min(1000.0)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct Intersection {
    coord: Coord,
    spread: Vector2<f32>,
    ty: Cell,
}

/// Intersection map storing the first voxel coordinate hit by each ray
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct IntersectionMap {
    pub width: usize,
    pub height: usize,
    pub camera_pos: Vector3<f32>,
    pub intersections: Vec<Option<Intersection>>,
}

impl IntersectionMap {
    pub fn new(width: usize, height: usize, camera_pos: Vector3<f32>) -> Self {
        Self {
            width,
            height,
            intersections: vec![None; width * height],
            camera_pos,
        }
    }

    /// Create an uninitialized intersection map for direct chunk writing
    pub fn new_uninitialized(width: usize, height: usize, camera_pos: Vector3<f32>) -> Self {
        Self {
            width,
            height,
            intersections: Vec::with_capacity(width * height),
            camera_pos,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Option<Intersection> {
        if x < self.width && y < self.height {
            self.intersections[y * self.width + x]
        } else {
            None
        }
    }

    pub fn set(&mut self, x: usize, y: usize, coord: Option<Intersection>) {
        if x < self.width && y < self.height {
            self.intersections[y * self.width + x] = coord;
        }
    }

    pub fn write_chunk(&mut self, start: usize, chunk: &[Option<Intersection>]) {
        let dst = &mut self.intersections[start..start + chunk.len()];
        dst.copy_from_slice(chunk);
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

    fn create_block(
        scale: usize,
        pos: Coord,
        centre_cell: VirtualCell,
    ) -> HashMap<Coord, VirtualCell> {
        let mut block_cells = HashMap::new();
        // Calculate the half-extent of the cube
        let half_scale = (scale as i32) / 2;

        // Create a scale × scale × scale cube centered around pos
        for dx in -(half_scale)..(half_scale + (scale as i32) % 2) {
            for dy in -(half_scale)..(half_scale + (scale as i32) % 2) {
                for dz in -(half_scale)..(half_scale + (scale as i32) % 2) {
                    let cube_coord = [pos[0] + dx, pos[1] + dy, pos[2] + dz];

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
    pub fn collisions(&self, centre: Vector3<f32>, dims: Vector3<f32>) -> VirtualCollisionShell {
        let mut collisions = ArrayVec::new();
        // We only need to check the cubes around.
        assert!(dims.x < 1.0 && dims.y < 1.0 && dims.z < 1.0);
        for cell_coord in crate::env::adjacent_coords(centre.map(|e| e.round() as i32)) {
            let coord: [i32; 3] = cell_coord.into();
            if let Some(cell) = self.cells().get(&coord) {
                if crate::env::intersects(cell_coord, centre, dims) {
                    collisions.push((coord, *cell));
                }
            }
        }
        collisions
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct CameraView {
    pub camera_pos: Vector3<f32>,
    pub camera_forward: Vector3<f32>,
    pub camera_up: Vector3<f32>,
    pub camera_right: Vector3<f32>,
}

impl CameraView {
    pub fn new(
        camera_pos: Vector3<f32>,
        camera_forward: Vector3<f32>,
        camera_up: Vector3<f32>,
        camera_right: Vector3<f32>,
    ) -> Self {
        Self {
            camera_pos,
            camera_forward,
            camera_up,
            camera_right,
        }
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(
            &Point3::from(self.camera_pos),                       // eye
            &Point3::from(self.camera_pos + self.camera_forward), // target
            &self.camera_up,                                      // up
        )
    }

    /// Calculate distance from camera to a coordinate
    pub fn distance_to(&self, coord: Coord) -> f32 {
        let pos = Vector3::from(coord).cast::<f32>();
        (pos - self.camera_pos).norm()
    }
}
