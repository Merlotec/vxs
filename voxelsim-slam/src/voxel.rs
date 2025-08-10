use anyhow::Result;
use nalgebra::{Isometry3, Point3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Bring in the shared VoxelGrid and Cell flags from the voxelsim crate
use voxelsim::{Cell, VoxelGrid};

// Simple occupancy voxel integrator that supports de-integration and re-integration
// for loop-closure pose corrections.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VoxelParams {
    pub voxel_size_m: f32,
    pub truncation_m: f32, // not used in simple occupancy, but kept for TSDF compat
    pub max_range_m: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey(i32, i32, i32);

fn voxel_key(p: &Point3<f32>, size: f32) -> VoxelKey {
    VoxelKey(
        (p.x / size).floor() as i32,
        (p.y / size).floor() as i32,
        (p.z / size).floor() as i32,
    )
}

#[derive(Debug, Clone, Default)]
pub struct VoxelCount {
    pub occ: i32,
    pub free: i32,
}

#[derive(Debug, Clone)]
pub struct FrameIntegrationRecord {
    pub frame_id: u64,
    pub voxel_deltas: HashMap<VoxelKey, VoxelCount>, // signed contributions
}

#[derive(Debug)]
pub struct VoxelIntegrator {
    // Current grid view compatible with voxelsim APIs
    pub grid: VoxelGrid,
    // Internal counts to enable de-integration and re-integration
    counts: HashMap<VoxelKey, VoxelCount>,
    params: VoxelParams,
    pub history: Vec<FrameIntegrationRecord>,
    next_frame_id: u64,
}

impl VoxelIntegrator {
    pub fn new(params: VoxelParams) -> Self {
        Self {
            grid: VoxelGrid::new(),
            counts: HashMap::new(),
            params,
            history: Vec::new(),
            next_frame_id: 0,
        }
    }

    pub fn integrate_points(&mut self, world_T_cam: &Isometry3<f32>, points_cam: &[Point3<f32>]) -> u64 {
        let id = self.next_frame_id;
        self.next_frame_id += 1;
        let mut deltas: HashMap<VoxelKey, VoxelCount> = HashMap::new();
        let vs = self.params.voxel_size_m;
        let max_r2 = self.params.max_range_m * self.params.max_range_m;

        for p_cam in points_cam.iter() {
            // Transform to world
            let p_w = world_T_cam.transform_point(p_cam);
            let r2 = p_w.coords.norm_squared();
            if r2.is_nan() || r2 > max_r2 { continue; }
            // End voxel (occupied increment)
            let end_key = voxel_key(&p_w, vs);
            let e = deltas.entry(end_key).or_default();
            e.occ += 1;

            // Raycast free space: from camera origin to p_w
            let cam_o = world_T_cam.translation.vector;
            let dir = p_w.coords - cam_o;
            let dist = dir.norm();
            if dist <= 1e-4 { continue; }
            let steps = (dist / vs).ceil() as i32;
            let step = dir / (steps as f32);
            let mut cur = cam_o;
            for _ in 0..steps.max(1) {
                cur += step;
                let cur_p = Point3::from(cur);
                let k = voxel_key(&cur_p, vs);
                if k == end_key { break; }
                let c = deltas.entry(k).or_default();
                c.free += 1;
            }
        }

        // Apply deltas to internal counts and update voxelsim::VoxelGrid view
        for (k, d) in deltas.iter() {
            let cnt = self.counts.entry(*k).or_default();
            cnt.occ += d.occ;
            cnt.free += d.free;
            self.sync_key_to_grid(*k, *cnt);
        }

        self.history.push(FrameIntegrationRecord { frame_id: id, voxel_deltas: deltas });
        id
    }

    pub fn deintegrate_frame(&mut self, frame_id: u64) -> Result<()> {
        if let Some(idx) = self.history.iter().position(|r| r.frame_id == frame_id) {
            let record = self.history.remove(idx);
            for (k, d) in record.voxel_deltas.iter() {
                if let Some(cnt) = self.counts.get_mut(k) {
                    cnt.occ -= d.occ;
                    cnt.free -= d.free;
                    // Remove zeroed entries to keep counts sparse
                    if cnt.occ == 0 && cnt.free == 0 {
                        self.counts.remove(k);
                    }
                    self.sync_key_to_grid(*k, *cnt);
                }
            }
        }
        Ok(())
    }

    fn sync_key_to_grid(&mut self, key: VoxelKey, cnt: VoxelCount) {
        let coord = Vector3::new(key.0, key.1, key.2);
        // Heuristic mapping: any positive occ -> FILLED, else if free -> SPARSE, else remove
        if cnt.occ > 0 {
            self.grid.set(coord, Cell::FILLED);
        } else if cnt.free > 0 {
            self.grid.set(coord, Cell::SPARSE);
        } else {
            let _ = self.grid.remove(&coord);
        }
    }
}

// Adapter traits for voxelsim-daemon (unknown exact API). We keep an abstract sink
// so you can route the current grid to your daemon process.
pub trait VoxelSimSink: Send + Sync {
    fn publish(&mut self, grid: &VoxelGrid) -> Result<()>;
}

pub struct LogSink;
impl VoxelSimSink for LogSink {
    fn publish(&mut self, grid: &VoxelGrid) -> Result<()> {
        // DashMap exposes len() for approximate count; clone keys isn't needed
        println!("voxels: {}", grid.cells().len());
        Ok(())
    }
}
