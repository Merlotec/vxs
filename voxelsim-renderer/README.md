# VoxelSim Renderer

3D renderer for VoxelSim using Bevy engine. This project listens for world and agent data over network sockets and renders the simulation in real-time.

## Dependencies

- **voxelsim-simulator**: Core simulation logic and data structures
- **Bevy**: 3D rendering engine
- **Tokio**: Async networking

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

The renderer will start and listen on:
- **Port 8080**: World data (terrain configs and voxel grids)
- **Port 8081**: Agent data (agent states and commands)

## Architecture

```
Python Client
     ↓ (uses simulator bindings)
Simulator (Rust + PyO3)
     ↓ (JSON over TCP)
Renderer (Rust + Bevy)
     ↓ (3D graphics)
Screen/Window
```

## Network Protocol

The renderer expects JSON messages on separate ports:

### World Data (Port 8080)
```json
{
    "terrain_config": {
        "size": [200, 64, 200],
        "height_scale": 80.0,
        "seed": 42
    },
    "voxel_grid": null
}
```

Or direct voxel data:
```json
{
    "terrain_config": null,
    "voxel_grid": {
        "cells": {
            "[10, 5, 10]": 16
        }
    }
}
```

### Agent Data (Port 8081)
```json
{
    "agents": [
        {
            "id": 0,
            "position": [50.0, 30.0, 50.0],
            "velocity": [0.0, 0.0, 0.0],
            "thrust": [0.0, 0.0, 0.0],
            "commands": [
                {
                    "direction": "Forward", 
                    "urgency": 1.0
                }
            ]
        }
    ]
}
```

## Controls

- **Mouse**: Orbit camera around scene
- **Scroll**: Zoom in/out
- **Z**: Cycle between agents
- **A/B/C/D**: Trigger agent movement commands
- **R**: Regenerate world

## Features

- Real-time terrain rendering with multiple tree types
- Agent visualization with movement trajectories  
- Camera controls for scene navigation
- Network-based data streaming
- Automatic world generation from terrain configs