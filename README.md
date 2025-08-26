# VoxelSim (VXS)

Modular voxel-based 3D simulation stack for robotics and autonomy. Rust first, with optional Python bindings; includes a real-time renderer, simulator, agent-vision compute, SLAM bridge, and UAV control integration.

## Highlights
- Core `voxelsim` crate with terrain, agents, networking, and Python bindings (feature `python`).
- Bevy-based desktop renderer (`voxelsim-renderer`) for real-time voxel worlds and agents.
- Simulator (`voxelsim-simulator`) with physics, terrain generation, and binary network streaming.
- Agent vision/compute pipeline (`voxelsim-compute`) for POV filtering and dense snapshots.
- SLAM bridge (`voxelsim-slam`) for RealSense → voxel grid with optional ROS2/cuVSLAM adapters.
- UAV integration: `voxelsim-daemon` (MAVLink, WIP) and `px4-mc` (PX4 multicopter controller bindings).
- Python package (`voxelsim-py`) exposing the Rust crates as a single module (`import voxelsim`).

## Repository Structure
- `voxelsim/`: Core types and API (Rust, optional PyO3 bindings).
- `voxelsim-renderer/`: Bevy 3D renderer binary. See `voxelsim-renderer/README.md`.
- `voxelsim-simulator/`: Simulation engine + optional Python bindings. See `voxelsim-simulator/README.md`.
- `voxelsim-compute/`: Agent POV rendering/filtering with optional Python bindings.
- `voxelsim-py/`: Python extension module stitching the crates into `voxelsim`.
- `voxelsim-daemon/`: Daemon for hardware/IO (MAVLink). Early WIP.
- `voxelsim-slam/`: RealSense capture, SLAM provider trait, voxel reintegration. See `voxelsim-slam/README.md`.
- `px4-mc/`: Thin Rust bindings over PX4 multicopter controllers (CMake-built static lib).
- `python/`: Experiments, training, and example scripts (heavy ML deps in `requirements.txt`).
- `logs/`, `target/`: Runtime/build artifacts.

## Prerequisites
- Rust toolchain with edition 2024 support (latest stable or nightly).
- Python 3.10+ and `maturin` if using Python bindings.
- Renderer: GPU + graphics backend (Vulkan/Metal/DirectX depending on OS).
- SLAM (optional): `librealsense2`; ROS2 for the ROS bridge; cuVSLAM libraries if enabling that feature.
- PX4 (optional): CMake; set `PX4_SRC_DIR` if building against PX4 sources for `px4-mc`.

## Quick Start

### 1) Build and run the renderer
```
cd voxelsim-renderer
cargo run --release
```
The renderer listens locally for simulation data (world/agents) on TCP ports 8080/8081 by default.

The renderer can also be configured to render the world that the agent sees (called the filter world).
This can be done by passing the `--virtual <channel_idx>` flat to `voxelsim-renderer`.

### 2) Send a world + agents from the simulator (Rust)
```
# New terminal
echo "Running simulator"
cd voxelsim-simulator
cargo run --release
```
This will generate terrain and stream world/agent updates to the renderer.

### 2b) Or from Python
First, build the Python extension that bundles the Rust crates:
```
cd voxelsim-py
pip install maturin
maturin develop
```
Then run an example script from `python/` (e.g., create a world and send to the renderer):
```
python python/povtest.py
# or
python python/sim_from_world.py
```

## Building Individual Crates
This repository does not use a top-level Cargo workspace; build crates from their folders, e.g.:
- `cd voxelsim && cargo build --release`
- `cd voxelsim-compute && cargo build --release`
- `cd voxelsim-slam && cargo build --release` (add features as needed)

## Features and Options
- `voxelsim`/`voxelsim-simulator`: `python` to enable PyO3 bindings.
- `voxelsim-slam`: `ros2` to enable ROS2 pose bridge; `cuvslam-ffi` to dynamically load cuVSLAM.
- `voxelsim-py`: `px4` to include PX4-related features from dependent crates.

## Development Notes
- Rust edition 2024 is used in several crates; ensure your toolchain is up to date.
- Some components are early-stage (e.g., `voxelsim-daemon`). Expect API changes.
- Heavy ML dependencies listed in `requirements.txt` are only required for the experimental Python workflows in `python/`.

## Documentation
- Renderer usage and protocol: `voxelsim-renderer/README.md`
- Simulator API and Python examples: `voxelsim-simulator/README.md`
- SLAM bridge details and Jetson notes: `voxelsim-slam/README.md`

## License
BSD 3-Clause. See `LICENSE`.

## Acknowledgements
- Bevy engine for rendering.
- PX4 and ArduPilot communities for flight control stacks.
- librealsense2 and ROS2 ecosystems for perception support.
