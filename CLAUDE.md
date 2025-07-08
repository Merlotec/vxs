# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoxelSim is a 3D voxel-based simulation engine with real-time rendering capabilities. It consists of two main components:

1. **voxelsim/** - Core simulation library with optional Python bindings
2. **voxelsim-renderer/** - Real-time 3D renderer using Bevy engine

The architecture uses a client-server model where the simulation sends data over TCP sockets to the renderer using binary serialization (bincode).

## Build Commands

### Core Library (voxelsim)
```bash
# Build Rust library only
cd voxelsim && cargo build --release

# Build with Python bindings
cd voxelsim && cargo build --release --features python

# Python development build (requires maturin)
cd voxelsim && maturin develop --features python
```

### Renderer (voxelsim-renderer)
```bash
# Build renderer
cd voxelsim-renderer && cargo build --release

# Run renderer (listens on ports 8080/8081)
cd voxelsim-renderer && cargo run --release
```

### Testing
```bash
# Run Rust tests
cd voxelsim && cargo test
cd voxelsim-renderer && cargo test

# Test Python bindings
cd voxelsim/python && python test.py
```

## Architecture

### Core Components

**voxelsim/src/lib.rs**: Main library entry point with module re-exports
- `terrain.rs`: Procedural terrain generation with Perlin noise
- `agent/mod.rs`: Agent physics simulation with thrust-based movement
- `env.rs`: VoxelGrid world representation and GlobalEnv simulation container
- `network.rs`: RendererClient for TCP communication with renderer
- `py.rs`: Python bindings (PyO3, enabled with `python` feature)

**voxelsim-renderer/src/main.rs**: Renderer entry point
- `render.rs`: Bevy-based 3D rendering pipeline
- Async network listeners for world (port 8080) and agent (port 8081) data
- Crossbeam channels for thread-safe communication

### Key Data Structures

- **VoxelGrid**: 3D HashMap-based voxel world with terrain generation
- **Agent**: Physics-enabled entities with position, velocity, and thrust
- **GlobalEnv**: Simulation container holding world and agents
- **RendererClient**: Network client for sending binary data to renderer

### Network Protocol

Binary protocol with 4-byte length prefix + bincode serialization:
- Port 8080: VoxelGrid world data
- Port 8081: Vec<Agent> agent data
- High-performance for real-time simulation

## Development Workflow

1. **Renderer First**: Start renderer with `cargo run --release` in voxelsim-renderer/
2. **Simulation**: Run simulation code (Rust or Python) that connects to renderer
3. **Python Testing**: Use `voxelsim/python/test.py` as reference for Python API

## Feature Configuration

The `python` feature enables PyO3 bindings:
- **Default**: Rust library only
- **With Python**: `cargo build --features python`

## Dependencies

- **nalgebra**: 3D math and linear algebra
- **noise**: Perlin noise for terrain generation
- **bevy**: 3D rendering engine (renderer only)
- **tokio**: Async networking
- **bincode**: Binary serialization
- **pyo3**: Python bindings (optional)
- **serde**: Serialization framework

## Performance Considerations

- Binary protocol optimized for real-time data transfer
- Voxel mesh optimization with face culling
- Efficient HashMap-based world representation
- Thrust-based physics simulation suitable for drone/agent dynamics