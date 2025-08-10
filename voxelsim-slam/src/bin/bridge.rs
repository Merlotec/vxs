use anyhow::Result;
use clap::Parser;
use voxelsim_slam::{BridgeConfig, BridgeRunner};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Optional config JSON file
    #[arg(short, long)]
    config: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = if let Some(p) = args.config {
        let s = std::fs::read_to_string(p)?;
        serde_json::from_str(&s)?
    } else {
        BridgeConfig::default_nano()
    };

    // Prefer ROS2 if enabled
    #[cfg(feature = "ros2")]
    let mut runner = {
        use voxelsim_slam::slam::ros2_bridge::Ros2Slam;
        let topic = std::env::var("ROS_ODOM_TOPIC").unwrap_or_else(|_| "/visual_slam/tracking/odometry".to_string());
        Ros2Slam::make_runner(cfg, &topic)?
    };
    // Else use cuVSLAM FFI if enabled
    #[cfg(all(not(feature = "ros2"), feature = "cuvslam-ffi"))]
    let mut runner = {
        use voxelsim_slam::slam::cuvslam_ffi::CuVslamProvider;
        CuVslamProvider::make_runner(cfg)?
    };
    // Fallback to Noop SLAM
    #[cfg(all(not(feature = "ros2"), not(feature = "cuvslam-ffi")))]
    let mut runner = BridgeRunner::new_default(cfg)?;
    loop {
        runner.spin_once()?;
    }
}
