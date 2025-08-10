use anyhow::Result;
use nalgebra::{Isometry3, Quaternion, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pose {
    pub t: [f32; 3],
    pub q: [f32; 4], // w,x,y,z
}

impl Default for Pose {
    fn default() -> Self {
        Self { t: [0.0; 3], q: [1.0, 0.0, 0.0, 0.0] }
    }
}

impl From<Pose> for Isometry3<f32> {
    fn from(p: Pose) -> Self {
        let rot = UnitQuaternion::from_quaternion(Quaternion::new(p.q[0], p.q[1], p.q[2], p.q[3]));
        Isometry3::from_parts(Vector3::from(p.t).into(), rot)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamConfig {
    pub use_cuvslam: bool,
}

#[derive(Debug, Clone)]
pub struct SlamFrameInput<'a> {
    pub timestamp_ns: i64,
    pub points_cam: &'a [nalgebra::Point3<f32>],
}

#[derive(Debug, Clone)]
pub struct SlamUpdate {
    pub world_T_cam: Pose,
    pub keyframe_id: Option<u64>,
    pub loop_closure: bool,
}

pub trait SlamProvider: Send + Sync {
    fn process_frame(&mut self, input: SlamFrameInput) -> Result<SlamUpdate>;
    // If a loop closure has happened, SLAM can provide corrected poses for prior keyframes.
    // Return Some(list) when new corrections are available.
    fn corrected_keyframes(&mut self) -> Result<Option<Vec<(u64, Pose)>>> {
        Ok(None)
    }
}

// Stub provider that does no SLAM (identity pose). Useful to test the pipeline.
pub struct NoopSlam;
impl NoopSlam {
    pub fn new() -> Self { Self }
}
impl SlamProvider for NoopSlam {
    fn process_frame(&mut self, input: SlamFrameInput) -> Result<SlamUpdate> {
        let _ = input;
        Ok(SlamUpdate { world_T_cam: Pose::default(), keyframe_id: None, loop_closure: false })
    }
}

// ROS2 pose bridge: subscribe to Isaac ROS Visual SLAM odometry and expose poses to the integrator.
#[cfg(feature = "ros2")]
pub mod ros2_bridge {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    // Import ROS2 crates
    use nav_msgs::msg::Odometry;

    pub struct Ros2Slam {
        latest: Arc<Mutex<Pose>>,
    }

    impl Ros2Slam {
        pub fn new(odom_topic: &str) -> anyhow::Result<Self> {
            let latest = Arc::new(Mutex::new(Pose::default()));
            let latest_cb = latest.clone();
            let topic = odom_topic.to_string();

            thread::spawn(move || {
                if let Err(e) = Self::spin_ros2(&topic, latest_cb) {
                    eprintln!("ROS2 bridge thread exited: {}", e);
                }
            });

            Ok(Self { latest })
        }

        fn odom_to_pose(msg: &Odometry) -> Pose {
            let p = &msg.pose.pose;
            Pose {
                t: [p.position.x as f32, p.position.y as f32, p.position.z as f32],
                // ROS quaternions are (x,y,z,w); our Pose expects (w,x,y,z)
                q: [p.orientation.w as f32, p.orientation.x as f32, p.orientation.y as f32, p.orientation.z as f32],
            }
        }

        fn spin_ros2(topic: &str, latest: Arc<Mutex<Pose>>) -> anyhow::Result<()> {
            let context = rclrs::Context::new([])?;
            let node = rclrs::create_node(&context, "voxelsim_slam_pose_bridge")?;
            let _sub = node.create_subscription::<Odometry>(topic, rclrs::QosProfile::default(), move |msg: Odometry| {
                let pose = Self::odom_to_pose(&msg);
                if let Ok(mut guard) = latest.lock() {
                    *guard = pose;
                }
            })?;
            rclrs::spin(&node)?;
            Ok(())
        }
    }

    impl SlamProvider for Ros2Slam {
        fn process_frame(&mut self, _input: SlamFrameInput) -> anyhow::Result<SlamUpdate> {
            let pose = { self.latest.lock().ok().cloned().unwrap_or_default() };
            Ok(SlamUpdate { world_T_cam: pose, keyframe_id: None, loop_closure: false })
        }
    }
}

// Placeholder cuVSLAM FFI adapter. On Nano and without ROS, this would require
// NVIDIA's closed-source library. We provide the interface only.
#[cfg(feature = "cuvslam-ffi")]
pub mod cuvslam_ffi {
    use super::*;
    use libloading::{Library, Symbol};

    #[allow(non_camel_case_types)]
    type elb_handle_t = *mut std::os::raw::c_void;

    // FFI symbol types (names are placeholders; adapt to your lib’s API)
    type elb_create_t = unsafe extern "C" fn() -> elb_handle_t;
    type elb_destroy_t = unsafe extern "C" fn(elb_handle_t);
    type elb_set_camera_t = unsafe extern "C" fn(
        elb_handle_t,
        width: i32,
        height: i32,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> i32;
    type elb_push_image_t = unsafe extern "C" fn(
        elb_handle_t,
        timestamp_ns: i64,
        data: *const u8,
        len: usize,
        stride: i32,
    ) -> i32;
    type elb_get_pose_t = unsafe extern "C" fn(elb_handle_t, out: *mut f32 /* len 7: wxyz + txyz */) -> i32;
    type elb_poll_loopclosure_t = unsafe extern "C" fn(elb_handle_t) -> i32;
    type elb_get_corrected_poses_t = unsafe extern "C" fn(
        elb_handle_t,
        out_buf: *mut f32,
        max_elems: usize,
    ) -> i32; // returns count of (kf_id, pose[7]) tuples serialised

    pub struct CuVslamProvider {
        _lib: Library,
        h: elb_handle_t,
        create: Symbol<'static, elb_create_t>,
        destroy: Symbol<'static, elb_destroy_t>,
        set_camera: Symbol<'static, elb_set_camera_t>,
        push_image: Symbol<'static, elb_push_image_t>,
        get_pose: Symbol<'static, elb_get_pose_t>,
        poll_loopclosure: Symbol<'static, elb_poll_loopclosure_t>,
        get_corrected_poses: Symbol<'static, elb_get_corrected_poses_t>,
    }

    unsafe impl Send for CuVslamProvider {}
    unsafe impl Sync for CuVslamProvider {}

    impl CuVslamProvider {
        pub fn new(cfg: crate::pipeline::BridgeConfig) -> anyhow::Result<Self> {
            let lib_path = std::env::var("CUVSLAM_LIB").unwrap_or_else(|_| "libcuvslam.so".to_string());
            let lib = unsafe { Library::new(lib_path) }?;

            unsafe {
                let create: Symbol<elb_create_t> = lib.get(b"elb_create")?;
                let destroy: Symbol<elb_destroy_t> = lib.get(b"elb_destroy")?;
                let set_camera: Symbol<elb_set_camera_t> = lib.get(b"elb_set_camera")?;
                let push_image: Symbol<elb_push_image_t> = lib.get(b"elb_push_image_bgr8")?;
                let get_pose: Symbol<elb_get_pose_t> = lib.get(b"elb_get_pose")?;
                let poll_loopclosure: Symbol<elb_poll_loopclosure_t> = lib.get(b"elb_poll_loopclosure")?;
                let get_corrected_poses: Symbol<elb_get_corrected_poses_t> = lib.get(b"elb_get_corrected_poses")?;

                let h = create();
                if h.is_null() { anyhow::bail!("cuVSLAM create returned null handle"); }

                let cam = cfg.camera;
                let r = set_camera(h, cam.width as i32, cam.height as i32, cfg.camera.width as f32, cfg.camera.height as f32, 0.5 * cam.width as f32, 0.5 * cam.height as f32);
                if r != 0 { anyhow::bail!("cuVSLAM set_camera failed: {}", r); }

                Ok(Self { _lib: lib, h, create, destroy, set_camera, push_image, get_pose, poll_loopclosure, get_corrected_poses })
            }
        }
    }

    impl Drop for CuVslamProvider {
        fn drop(&mut self) {
            unsafe { (self.destroy)(self.h) }
        }
    }

    impl SlamProvider for CuVslamProvider {
        fn process_frame(&mut self, input: SlamFrameInput) -> anyhow::Result<SlamUpdate> {
            // For now we assume caller provides a color frame via global state; this adapter only fetches pose
            // A production adapter would take an image buffer reference and push it here.
            // Push image is optional in this skeleton to allow compile.
            let mut buf = [0f32; 7];
            let r = unsafe { (self.get_pose)(self.h, buf.as_mut_ptr()) };
            if r != 0 { anyhow::bail!("cuVSLAM get_pose failed: {}", r); }
            let pose = Pose { q: [buf[0], buf[1], buf[2], buf[3]], t: [buf[4], buf[5], buf[6]] };
            let lc = unsafe { (self.poll_loopclosure)(self.h) } != 0;
            let _ = input; // unused in this simplified skeleton
            Ok(SlamUpdate { world_T_cam: pose, keyframe_id: None, loop_closure: lc })
        }

        fn corrected_keyframes(&mut self) -> anyhow::Result<Option<Vec<(u64, Pose)>>> {
            // Pull corrected keyframe poses in a simple packed float buffer
            let max = 1024usize;
            let mut floats = vec![0f32; max * 8];
            let n = unsafe { (self.get_corrected_poses)(self.h, floats.as_mut_ptr(), max) };
            if n <= 0 { return Ok(None); }
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..(n as usize) {
                let base = i * 8;
                let kf_id = floats[base] as u64;
                let pose = Pose { q: [floats[base + 1], floats[base + 2], floats[base + 3], floats[base + 4]], t: [floats[base + 5], floats[base + 6], floats[base + 7]] };
                out.push((kf_id, pose));
            }
            Ok(Some(out))
        }
    }
}
