vslam_voxelsim_bridge
=====================

RealSense → pointcloud → voxel grid bridge with a pluggable SLAM provider and loop-closure-safe integration for use with a voxelsim daemon.

Status: skeleton implementation focusing on the data plumbing and reintegration logic. It does not ship a cuVSLAM binding; you can add it via FFI where available.

Components
----------

- RealSense capture: uses `realsense-rust` (librealsense2) to stream depth and back-project points up to a configurable range (e.g., 30 m).
- SLAM provider trait: define your pose source (e.g., cuVSLAM). A `NoopSlam` stub is provided.
- Voxel integrator: simple occupancy map with per-frame deltas allowing de-integration and re-integration on loop closure (pose corrections).
- Sink abstraction: publish the current voxel grid to your `voxelsim-daemon` (adapter stub provided).

Jetson Nano notes
-----------------

- Isaac ROS Visual SLAM (ROS2 + cuVSLAM) typically targets JetPack 5 on Xavier/Orin. Jetson Nano (JetPack 4.x) is generally unsupported by Isaac ROS. If you need NVIDIA cuVSLAM without ROS, you will need access to the underlying SDK and libraries compatible with Nano. Otherwise, consider an alternative VSLAM (e.g., ORB-SLAM3) and wrap it behind the `SlamProvider` trait.
- Install `librealsense2` on Nano and enable the correct UVC/UDEV permissions. Use reduced resolutions (e.g., 848x480@30) for throughput.

Build
-----

Default (Noop SLAM):

    cargo build --release -p voxelsim-slam

With cuVSLAM FFI (requires compatible libcuvslam.so):

    cargo build --release -p voxelsim-slam --features cuvslam-ffi

Run
---

Noop SLAM:

    cargo run -p voxelsim-slam --bin bridge

cuVSLAM FFI:

    CUVSLAM_LIB=/path/to/libcuvslam.so cargo run -p voxelsim-slam --bin bridge --features cuvslam-ffi

ROS2 Pose Bridge (Isaac ROS Visual SLAM):

    # Ensure ROS2 and Isaac ROS Visual SLAM are running and publishing odometry
    # Enable the ROS2 feature and optionally set the odom topic
    ROS_ODOM_TOPIC=/visual_slam/tracking/odometry \
      cargo run -p voxelsim-slam --bin bridge --features ros2

Optional config file (JSON):

    {
      "camera": { "width": 848, "height": 480, "fps": 30, "max_range_m": 30.0 },
      "voxel": { "voxel_size_m": 0.1, "truncation_m": 0.2, "max_range_m": 30.0 }
    }

Wiring cuVSLAM
--------------

- A `CuVslamProvider` skeleton is included behind `cuvslam-ffi`, using `libloading` to bind symbols at runtime. Set `CUVSLAM_LIB` to point at your `libcuvslam.so`.
- The symbol names and signatures in `slam::cuvslam_ffi` are placeholders; adjust to your SDK. Wire image push (BGR8) and intrinsics as required by your version.
- On loop closure, `corrected_keyframes()` is polled for corrected poses; the pipeline de-integrates and re-integrates stored keyframes accordingly.

Integrating with voxelsim-daemon
--------------------------------

- Replace `LogSink` with an adapter that publishes the `voxelsim::VoxelGrid` directly to your daemon (via IPC/GRPC or direct crate usage).
- With the `voxelsim` crate included as a dependency, implement a `VoxelSimSink` that publishes the grid incrementally (delta updates per frame) to avoid large transfers.

ROS2 Details
------------

- Feature `ros2` adds a `Ros2Slam` provider that subscribes to a ROS2 odometry topic (default `/visual_slam/tracking/odometry`) and feeds poses into the voxel integrator.
- Configure the topic using `ROS_ODOM_TOPIC` environment variable when running the bridge.

Caveats
-------

- Current SLAM is a stub; mapping will accumulate in camera frame without a real pose estimate.
- Pointcloud generation is naive and may be CPU-heavy; consider GPU pointcloud or decimation on Nano.
- For 30 m range, ensure your RealSense model supports the requested range and lighting.
- cuVSLAM typically expects stereo/RGB + optionally IMU. D456 provides depth + RGB; verify your cuVSLAM build supports mono RGB. If not, substitute a supported camera or another SLAM.
