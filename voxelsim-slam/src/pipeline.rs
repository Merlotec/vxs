use crate::camera::{CameraConfig, RealSenseCamera};
use crate::slam::{NoopSlam, SlamProvider, SlamUpdate};
use crate::voxel::{LogSink, VoxelIntegrator, VoxelParams, VoxelSimSink};
use anyhow::Result;
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub camera: CameraConfig,
    pub voxel: VoxelParams,
    // TODO: slam config
}

pub struct BridgeRunner<S: SlamProvider, X: VoxelSimSink> {
    cam: RealSenseCamera,
    slam: S,
    integrator: VoxelIntegrator,
    sink: X,
    // Keep track of keyframe to frame_id mapping so we can de/re-integrate on loop closure
    keyframe_frame_ids: Vec<(u64 /*kf*/, u64 /*frame*/)> ,
    // Store per-keyframe points for re-integration (memory-bound; tune/limit in production)
    stored_points: Vec<(u64 /*kf*/, Vec<nalgebra::Point3<f32>>)>,
}

impl BridgeConfig {
    pub fn default_nano() -> Self {
        Self {
            camera: crate::camera::CameraConfig { width: 848, height: 480, fps: 30, max_range_m: 30.0 },
            voxel: crate::voxel::VoxelParams { voxel_size_m: 0.1, truncation_m: 0.2, max_range_m: 30.0 },
        }
    }
}

impl BridgeRunner<NoopSlam, LogSink> {
    pub fn new_default(cfg: BridgeConfig) -> Result<Self> {
        let cam = RealSenseCamera::new(cfg.camera)?;
        let slam = NoopSlam::new();
        let integrator = VoxelIntegrator::new(cfg.voxel);
        let sink = LogSink;
        Ok(Self { cam, slam, integrator, sink, keyframe_frame_ids: Vec::new(), stored_points: Vec::new() })
    }
}

#[cfg(feature = "cuvslam-ffi")]
impl crate::slam::cuvslam_ffi::CuVslamProvider {
    pub fn make_runner(cfg: BridgeConfig) -> Result<super::BridgeRunner<Self, LogSink>> {
        let cam = RealSenseCamera::new(cfg.camera.clone())?;
        let slam = Self::new(cfg.clone())?;
        let integrator = VoxelIntegrator::new(cfg.voxel);
        let sink = LogSink;
        Ok(super::BridgeRunner { cam, slam, integrator, sink, keyframe_frame_ids: Vec::new(), stored_points: Vec::new() })
    }
}

// Convenience constructor when using ROS2 pose bridge
#[cfg(feature = "ros2")]
impl crate::slam::ros2_bridge::Ros2Slam {
    pub fn make_runner(cfg: BridgeConfig, odom_topic: &str) -> Result<super::BridgeRunner<Self, LogSink>> {
        let cam = RealSenseCamera::new(cfg.camera.clone())?;
        let slam = Self::new(odom_topic)?;
        let integrator = VoxelIntegrator::new(cfg.voxel);
        let sink = LogSink;
        Ok(super::BridgeRunner { cam, slam, integrator, sink, keyframe_frame_ids: Vec::new(), stored_points: Vec::new() })
    }
}

impl<S: SlamProvider, X: VoxelSimSink> BridgeRunner<S, X> {
    pub fn spin_once(&mut self) -> Result<()> {
        let frame = self.cam.next_frame()?;
        let ts = frame.timestamp_ns;
        let upd: SlamUpdate = self
            .slam
            .process_frame(crate::slam::SlamFrameInput { timestamp_ns: ts, points_cam: &frame.points })?;
        let world_T_cam: Isometry3<f32> = upd.world_T_cam.into();

        let frame_id = self.integrator.integrate_points(&world_T_cam, &frame.points);

        if let Some(kf) = upd.keyframe_id {
            self.keyframe_frame_ids.push((kf, frame_id));
            // Save points for re-integration
            self.stored_points.push((kf, frame.points.clone()));
        }

        // On loop closure, a real SLAM would provide adjusted keyframe poses.
        // Here we demonstrate how to de/re-integrate affected frames.
        if upd.loop_closure {
            if let Some(corrected) = self.slam.corrected_keyframes()? {
                for (kf_id, pose) in corrected.iter() {
                    if let Some((_, frame_id)) = self.keyframe_frame_ids.iter().find(|(k, _)| k == kf_id) {
                        // Deintegration
                        let _ = self.integrator.deintegrate_frame(*frame_id);
                        // Re-integration with corrected pose
                        if let Some((_, pts)) = self.stored_points.iter().find(|(k, _)| k == kf_id) {
                            let iso: Isometry3<f32> = (*pose).into();
                            self.integrator.integrate_points(&iso, pts);
                        }
                    }
                }
            }
        }

        self.sink.publish(&self.integrator.grid)?;
        Ok(())
    }
}
