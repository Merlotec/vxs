use nalgebra::{Matrix3, Vector2, Vector3};
use rayon::prelude::*;

use crate::{Cell, Coord, VoxelGrid};
use std::collections::HashMap;
use std::time::SystemTime;

use super::*;

pub type VirtualCollisionShell = ArrayVec<[(Coord, VirtualCell); 26]>;

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VirtualCell {
    pub time: SystemTime,
    pub cell: Cell,
    pub uncertainty: f32,
}

impl Default for VirtualCell {
    fn default() -> Self {
        Self {
            time: SystemTime::UNIX_EPOCH,
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
        let age_bonus = match self.time.elapsed() {
            Ok(duration) => {
                let age_seconds = duration.as_secs_f32();
                // Newer data gets higher priority (exponential decay)
                (-age_seconds / 60.0).exp() // 1-minute half-life
            }
            Err(_) => 0.0, // Future timestamps get no bonus
        };

        let uncertainty_priority = if self.uncertainty > 0.0 {
            1.0 / self.uncertainty
        } else {
            1000.0 // Very high priority for zero uncertainty
        };

        uncertainty_priority + age_bonus
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Intersection {
    coord: Coord,
    spread: Vector2<f32>,
    ty: Cell,
}

/// Intersection map storing the first voxel coordinate hit by each ray
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

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct VirtualGrid {
    pub cells: HashMap<Coord, VirtualCell>,
}

impl VirtualGrid {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }

    /// Merge two virtual grids based on camera frustum and priority
    /// Checks ALL cells within the frustum, including empty space
    pub fn merge(mut self, other: VirtualGrid, camera: &CameraView) -> Self {
        // Get all coordinates within the camera frustum
        let frustum_coords = camera.get_frustum_coords();

        // Process each coordinate in the frustum
        for coord in frustum_coords {
            let distance_to_camera = camera.distance_to(coord);

            // Get cells from both grids (None if cell doesn't exist)
            let self_cell = self.cells.get(&coord);
            let other_cell = other.cells.get(&coord);

            match (self_cell, other_cell) {
                (Some(self_cell), Some(other_cell)) => {
                    // Both grids have a cell at this coordinate - keep higher priority
                    if other_cell.priority() > self_cell.priority() {
                        self.cells.insert(coord, other_cell.clone());
                    }
                    // If self_cell has higher priority, keep it (do nothing)
                }
                (Some(self_cell), None) => {
                    // Only self has a cell - keep it only if uncertainty < distance
                    if self_cell.uncertainty >= distance_to_camera {
                        // Remove the cell because empty space (zero uncertainty) has higher priority
                        self.cells.remove(&coord);
                    }
                }
                (None, Some(other_cell)) => {
                    // Only other has a cell - add it
                    self.cells.insert(coord, other_cell.clone());
                }
                (None, None) => {
                    // Neither grid has a cell (empty space) - check if we should remove any existing cell
                    // Empty space has uncertainty = distance (perfect knowledge at that distance)
                    // This case is already handled above in (Some, None)
                }
            }
        }

        // Final cleanup: remove any remaining self cells outside frustum or with high uncertainty
        self.cells.retain(|&coord, cell| {
            let distance_to_camera = camera.distance_to(coord);
            cell.uncertainty < distance_to_camera
        });

        self
    }

    pub fn world_from_intersection_map(map: IntersectionMap) -> VirtualGrid {
        let mut virtual_grid = VirtualGrid::new();
        let base_scale = Vector2::<f32>::new(map.width as f32, map.height as f32);

        for grid_pos in map.intersections.iter().flatten() {
            let scale = base_scale.dot(&grid_pos.spread);
            let block_scale = scale.max(1.0) as usize;

            let dist = (map.camera_pos - Vector3::from(grid_pos.coord).cast::<f32>()).norm();
            // Create a block at the intersection point
            let block_cells = Self::create_block(
                block_scale,
                grid_pos.coord,
                VirtualCell {
                    time: SystemTime::now(),
                    cell: grid_pos.ty,
                    uncertainty: dist,
                },
            );

            // Add all block cells to the virtual grid
            for (coord, virtual_cell) in block_cells {
                virtual_grid.cells.insert(coord, virtual_cell);
            }
        }

        virtual_grid
    }

    fn create_block(
        scale: usize,
        pos: Coord,
        centre_cell: VirtualCell,
    ) -> HashMap<Coord, VirtualCell> {
        let mut block_cells = HashMap::new();
        let current_time = SystemTime::now();
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

    pub fn cells(&self) -> &HashMap<Coord, VirtualCell> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut HashMap<Coord, VirtualCell> {
        &mut self.cells
    }

    pub fn set(&mut self, coord: Coord, cell: VirtualCell) {
        self.cells.insert(coord, cell);
    }

    pub fn remove(&mut self, coord: &Coord) -> Option<VirtualCell> {
        self.cells.remove(coord)
    }

    /// Returns a the list of cells if an object with the given centre coordinate and dimensions collides with any
    /// cells.
    pub fn collisions(&self, centre: Vector3<f32>, dims: Vector3<f32>) -> VirtualCollisionShell {
        let mut collisions = ArrayVec::new();
        // We only need to check the cubes around.
        assert!(dims.x < 1.0 && dims.y < 1.0 && dims.z < 1.0);
        for cell_coord in crate::env::adjacent_coords(centre.try_cast::<i32>().unwrap()) {
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

pub struct CameraView {
    pub camera_pos: Vector3<f32>,
    pub camera_forward: Vector3<f32>,
    pub camera_up: Vector3<f32>,
    pub camera_right: Vector3<f32>,

    pub fov_horizontal: f32,
    pub fov_vertical: f32,

    pub max_distance: f32,
}

impl CameraView {
    pub fn new(
        camera_pos: Vector3<f32>,
        camera_forward: Vector3<f32>,
        camera_up: Vector3<f32>,
        camera_right: Vector3<f32>,
        fov_horizontal: f32,
        fov_vertical: f32,
        max_distance: f32,
    ) -> Self {
        Self {
            camera_pos,
            camera_forward,
            camera_up,
            camera_right,
            fov_horizontal,
            fov_vertical,
            max_distance,
        }
    }

    /// Check if a coordinate is within the camera frustum
    pub fn is_in_frustum(&self, coord: Coord) -> bool {
        let pos = Vector3::from(coord).cast::<f32>();
        let to_point = pos - self.camera_pos;
        let distance = to_point.norm();

        // Check distance
        if distance > self.max_distance {
            return false;
        }

        // Check if point is in front of camera
        let forward_dot = to_point.dot(&self.camera_forward);
        if forward_dot <= 0.0 {
            return false;
        }

        // Check horizontal FOV
        let right_dot = to_point.dot(&self.camera_right);
        let horizontal_angle = (right_dot / forward_dot).abs();
        if horizontal_angle > (self.fov_horizontal * 0.5).tan() {
            return false;
        }

        // Check vertical FOV
        let up_dot = to_point.dot(&self.camera_up);
        let vertical_angle = (up_dot / forward_dot).abs();
        if vertical_angle > (self.fov_vertical * 0.5).tan() {
            return false;
        }

        true
    }

    /// Calculate distance from camera to a coordinate
    pub fn distance_to(&self, coord: Coord) -> f32 {
        let pos = Vector3::from(coord).cast::<f32>();
        (pos - self.camera_pos).norm()
    }

    /// Get all coordinates within the camera frustum
    /// This generates a bounding box and tests each coordinate for frustum inclusion
    pub fn get_frustum_coords(&self) -> Vec<Coord> {
        let mut coords = Vec::new();

        // Calculate tighter bounding box based on frustum geometry
        let max_dist = self.max_distance;
        let half_fov_h = self.fov_horizontal * 0.5;
        let half_fov_v = self.fov_vertical * 0.5;

        // Calculate the frustum extents at max distance
        let far_extent_h = max_dist * half_fov_h.tan();
        let far_extent_v = max_dist * half_fov_v.tan();

        // Project camera coordinate system to integer coordinates
        let camera_coord = [
            self.camera_pos.x.round() as i32,
            self.camera_pos.y.round() as i32,
            self.camera_pos.z.round() as i32,
        ];

        // Create a tighter bounding box
        let max_extent = (far_extent_h.max(far_extent_v) + 1.0).ceil() as i32;
        let forward_extent = (max_dist + 1.0).ceil() as i32;

        // Pre-allocate vector with estimated size
        let estimated_volume = (max_extent * 2 + 1).pow(2) * forward_extent;
        coords.reserve(estimated_volume as usize / 4); // Conservative estimate

        // Iterate through a more efficient bounding box
        for dx in -max_extent..=max_extent {
            for dy in -max_extent..=max_extent {
                for dz in 0..=forward_extent {
                    let coord = [
                        camera_coord[0] + dx,
                        camera_coord[1] + dy,
                        camera_coord[2] + dz,
                    ];

                    if self.is_in_frustum(coord) {
                        coords.push(coord);
                    }
                }
            }
        }

        coords.shrink_to_fit(); // Free unused memory
        coords
    }
}

/// Cast rays in an n×m grid and return intersection map (direct memory writing)
pub fn raycast_slice(
    world: &VoxelGrid,
    camera: CameraView,
    width: usize,
    height: usize,
) -> IntersectionMap {
    // Calculate ray direction parameters once
    let half_fov_h = camera.fov_horizontal * 0.5;
    let half_fov_v = camera.fov_vertical * 0.5;
    let tan_half_fov_h = half_fov_h.tan();
    let tan_half_fov_v = half_fov_v.tan();
    let step_x = (2.0 * tan_half_fov_h) / (width as f32 - 1.0);
    let step_y = (2.0 * tan_half_fov_v) / (height as f32 - 1.0);

    let fov_spread = Vector2::new(tan_half_fov_v, tan_half_fov_h);

    let total_pixels = width * height;
    let chunk_size = calculate_chunk_size(total_pixels);

    // Create chunk ranges that align with memory layout
    let num_chunks = (total_pixels + chunk_size - 1) / chunk_size;
    let chunk_ranges: Vec<(usize, usize)> = (0..num_chunks)
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_pixels);
            (start, end)
        })
        .collect();

    // Process chunks in parallel, returning results in order
    let chunk_results: Vec<Vec<Option<Intersection>>> = chunk_ranges
        .par_iter()
        .map(|&(start, end)| {
            let chunk_len = end - start;
            let mut chunk_data = Vec::with_capacity(chunk_len);

            // Process pixels in sequential order within chunk
            for index in start..end {
                let x = index % width;
                let y = index / width;

                // Calculate normalized device coordinates [-1, 1]
                let ndc_x = (x as f32 * step_x) - tan_half_fov_h;
                let ndc_y = tan_half_fov_v - (y as f32 * step_y);

                // Calculate ray direction in world space
                let ray_dir = (camera.camera_forward
                    + camera.camera_right * ndc_x
                    + camera.camera_up * ndc_y)
                    .normalize();

                // Perform DDA ray casting and store result directly in order
                let hit =
                    dda_raycast_static(world, camera.camera_pos, ray_dir, camera.max_distance);

                chunk_data.push(hit.map(|(h, ty)| Intersection {
                    ty,
                    coord: h,
                    spread: (Vector3::<i32>::from(h).cast::<f32>() - camera.camera_pos).norm()
                        * fov_spread, // Calculates how 'spread out' the rays are at a given point.
                }));
            }

            chunk_data
        })
        .collect();

    // Create intersection map with pre-allocated capacity
    let mut intersection_map = IntersectionMap::new(width, height, camera.camera_pos);

    // Write chunks directly to memory in order
    for (chunk_idx, chunk_data) in chunk_results.into_iter().enumerate() {
        let start_offset = chunk_idx * chunk_size;
        intersection_map.write_chunk(start_offset, &chunk_data);
    }
    // Finalize the vector with correct length

    intersection_map
}

/// Fast DDA (Digital Differential Analyzer) ray casting algorithm (static version for threading)
/// Returns the first solid voxel coordinate hit by the ray
fn dda_raycast_static(
    world: &VoxelGrid,
    ray_start: Vector3<f32>,
    ray_dir: Vector3<f32>,
    max_distance: f32,
) -> Option<(Coord, Cell)> {
    // Current voxel coordinates
    let mut map_x = ray_start.x.floor() as i32;
    let mut map_y = ray_start.y.floor() as i32;
    let mut map_z = ray_start.z.floor() as i32;

    // Calculate delta distances (distance to next voxel boundary)
    let delta_dist_x = if ray_dir.x == 0.0 {
        f32::INFINITY
    } else {
        (1.0 / ray_dir.x).abs()
    };
    let delta_dist_y = if ray_dir.y == 0.0 {
        f32::INFINITY
    } else {
        (1.0 / ray_dir.y).abs()
    };
    let delta_dist_z = if ray_dir.z == 0.0 {
        f32::INFINITY
    } else {
        (1.0 / ray_dir.z).abs()
    };

    // Calculate step direction and initial side distances
    let (step_x, mut side_dist_x) = if ray_dir.x < 0.0 {
        (-1, (ray_start.x - map_x as f32) * delta_dist_x)
    } else {
        (1, (map_x as f32 + 1.0 - ray_start.x) * delta_dist_x)
    };

    let (step_y, mut side_dist_y) = if ray_dir.y < 0.0 {
        (-1, (ray_start.y - map_y as f32) * delta_dist_y)
    } else {
        (1, (map_y as f32 + 1.0 - ray_start.y) * delta_dist_y)
    };

    let (step_z, mut side_dist_z) = if ray_dir.z < 0.0 {
        (-1, (ray_start.z - map_z as f32) * delta_dist_z)
    } else {
        (1, (map_z as f32 + 1.0 - ray_start.z) * delta_dist_z)
    };

    // Perform DDA traversal
    let mut current_distance = 0.0;

    loop {
        // Check if we've hit a solid voxel
        let coord = [map_x, map_y, map_z];
        if let Some(cell) = world.cells().get(&coord) {
            if cell.intersects(Cell::FILLED | Cell::GROUND) {
                return Some((coord, *cell));
            }
        }

        // Check if we've exceeded max distance
        if current_distance > max_distance {
            break;
        }

        // Jump to next voxel boundary
        if side_dist_x < side_dist_y && side_dist_x < side_dist_z {
            side_dist_x += delta_dist_x;
            map_x += step_x;
            current_distance = side_dist_x;
        } else if side_dist_y < side_dist_z {
            side_dist_y += delta_dist_y;
            map_y += step_y;
            current_distance = side_dist_y;
        } else {
            side_dist_z += delta_dist_z;
            map_z += step_z;
            current_distance = side_dist_z;
        }
    }

    None
}

/// Fast DDA (Digital Differential Analyzer) ray casting algorithm (instance method)
/// Returns the first solid voxel coordinate hit by the ray
fn dda_raycast(
    world: &VoxelGrid,
    ray_start: Vector3<f32>,
    ray_dir: Vector3<f32>,
    max_distance: f32,
) -> Option<(Coord, Cell)> {
    dda_raycast_static(world, ray_start, ray_dir, max_distance)
}

/// Calculate optimal chunk size for parallel processing
/// Balances thread overhead with work distribution
fn calculate_chunk_size(total_pixels: usize) -> usize {
    let num_threads = rayon::current_num_threads();
    let base_chunk_size = total_pixels / (num_threads * 4); // 4x oversubscription for better load balancing

    // Minimum chunk size to avoid excessive overhead
    let min_chunk_size = 64;

    // Maximum chunk size to ensure good work distribution
    let max_chunk_size = 4096;

    base_chunk_size.clamp(min_chunk_size, max_chunk_size)
}
