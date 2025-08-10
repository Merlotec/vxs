use anyhow::{Context, Result};
use nalgebra::{Point3, Vector3};

// RealSense
use realsense_rust as rs2;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CameraConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub max_range_m: f32, // e.g., 30.0
}

pub struct RealSenseCamera {
    pub ctx: rs2::Context,
    pub pipeline: rs2::Pipeline,
    pub align: rs2::Align,
    pub intr: CameraIntrinsics,
    pub cfg: CameraConfig,
}

pub struct FramePoints {
    pub points: Vec<Point3<f32>>, // in camera frame
}

pub struct FrameData {
    pub points: Vec<Point3<f32>>,     // in camera frame (meters)
    pub color_bgr8: Option<Vec<u8>>,  // width*height*3
    pub width: u32,
    pub height: u32,
    pub timestamp_ns: i64,
}

impl RealSenseCamera {
    pub fn new(cfg: CameraConfig) -> Result<Self> {
        let ctx = rs2::Context::new().context("create realsense context")?;
        let mut pipeline = rs2::Pipeline::new(&ctx)?;
        let mut config = rs2::Config::new();
        config
            .enable_stream(
                rs2::StreamKind::Depth,
                None,
                cfg.width as i32,
                cfg.height as i32,
                rs2::Format::Z16,
                cfg.fps as i32,
            )
            .context("enable depth stream")?;
        // Enable color stream for SLAM (BGR/RGB)
        config.enable_stream(
            rs2::StreamKind::Color,
            None,
            cfg.width as i32,
            cfg.height as i32,
            rs2::Format::Bgr8,
            cfg.fps as i32,
        )?;

        let profile = pipeline.start(&config)?;
        let depth_stream = profile
            .get_stream(rs2::StreamKind::Depth)
            .context("get depth stream profile")?;
        let video = depth_stream.as_video_stream_profile().unwrap();
        let intr = video.get_intrinsics()?;
        let intr = CameraIntrinsics {
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.ppx,
            cy: intr.ppy,
        };
        let align = rs2::Align::new(rs2::StreamKind::Depth);
        Ok(Self {
            ctx,
            pipeline,
            align,
            intr,
            cfg,
        })
    }

    pub fn next_points(&mut self) -> Result<FramePoints> {
        let frames = self.pipeline.wait(None)?; // blocking wait
        let aligned = self.align.process(&frames)?;
        let depth = aligned
            .frames()
            .find_map(|f| f.as_depth_frame())
            .context("no depth frame in aligned set")?;

        let w = depth.width() as usize;
        let h = depth.height() as usize;
        let mut out = Vec::with_capacity(w * h / 2);
        let max_r2 = self.cfg.max_range_m * self.cfg.max_range_m;

        for y in 0..h {
            for x in 0..w {
                if let Some(z_m) = depth.get_distance(x as i32, y as i32) {
                    // Skip invalid zeros and out of range
                    if z_m <= 0.0 || z_m.is_nan() || z_m * z_m > max_r2 {
                        continue;
                    }
                    // Back-project to camera frame (meters)
                    let x_m = (x as f32 - self.intr.cx) * z_m / self.intr.fx;
                    let y_m = (y as f32 - self.intr.cy) * z_m / self.intr.fy;
                    out.push(Point3::new(x_m, y_m, z_m));
                }
            }
        }
        Ok(FramePoints { points: out })
    }

    pub fn next_frame(&mut self) -> Result<FrameData> {
        let frames = self.pipeline.wait(None)?;
        let ts_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let aligned = self.align.process(&frames)?;
        let depth = aligned
            .frames()
            .find_map(|f| f.as_depth_frame())
            .context("no depth frame in aligned set")?;
        let color = aligned
            .frames()
            .find_map(|f| f.as_video_frame())
            .and_then(|vf| {
                let prof = vf.get_profile();
                if prof.stream_kind() == rs2::StreamKind::Color && prof.format() == rs2::Format::Bgr8 {
                    Some(vf)
                } else {
                    None
                }
            });

        let w = depth.width() as usize;
        let h = depth.height() as usize;
        let mut points = Vec::with_capacity(w * h / 2);
        let max_r2 = self.cfg.max_range_m * self.cfg.max_range_m;
        for y in 0..h {
            for x in 0..w {
                if let Some(z_m) = depth.get_distance(x as i32, y as i32) {
                    if z_m <= 0.0 || z_m.is_nan() || z_m * z_m > max_r2 {
                        continue;
                    }
                    let x_m = (x as f32 - self.intr.cx) * z_m / self.intr.fx;
                    let y_m = (y as f32 - self.intr.cy) * z_m / self.intr.fy;
                    points.push(Point3::new(x_m, y_m, z_m));
                }
            }
        }

        let (cw, ch) = if let Some(ref vf) = color { (vf.width() as usize, vf.height() as usize) } else { (0, 0) };
        let color_bgr8 = color.map(|vf| vf.as_bytes().to_vec());
        Ok(FrameData {
            points,
            color_bgr8,
            width: cw as u32,
            height: ch as u32,
            timestamp_ns: ts_ns,
        })
    }
}
