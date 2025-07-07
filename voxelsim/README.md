# VoxelSim Simulator

Core simulation engine for VoxelSim with Python bindings. This project contains the terrain generation, agent simulation, and physics logic. It provides Python bindings via PyO3/maturin for easy integration.

## Building

### Rust Library
```bash
cargo build --release
```

### Python Bindings
```bash
# Install maturin if not already installed
pip install maturin

# Development build (creates .so file in target/wheels/)
maturin develop

# Production build
maturin build --release
```

## Python Usage

After building with `maturin develop`, you can use the simulator in Python:

```python
import voxelsim_simulator as sim

# Create terrain configuration
config = sim.PyTerrainConfig(
    size=(200, 64, 200),
    height_scale=80.0,
    seed=42
)

# Create simulation
simulation = sim.PySimulation()
simulation.generate_terrain(config)

# Create agents
agent = sim.PyAgent(0, x=50.0, y=30.0, z=50.0)
agent.add_move_command("forward", 1.0)
simulation.add_agent(agent)

# Get data for sending to renderer
world_data = simulation.get_world_data()  # JSON string
agent_data = simulation.get_agent_data()  # JSON string
```

## Network Protocol

The simulator generates JSON data that can be sent to the renderer:

### World Data
```json
{
    "terrain_config": null,
    "voxel_grid": {
        "cells": {
            "[10, 5, 10]": 16,
            "[11, 5, 10]": 32
        }
    }
}
```

### Agent Data
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

## API Reference

### PyTerrainConfig
- `PyTerrainConfig(size, height_scale, flat_band, max_terrain_height, forest_scale, max_forest_height, seed, base_thickness)`
- Properties: `size`, `seed`
- Methods: `to_json()`

### PyVoxelGrid  
- `PyVoxelGrid()`
- Methods: `generate_terrain(config)`, `to_json()`, `cell_count()`

### PyAgent
- `PyAgent(id, x, y, z)`
- Properties: `id`, `position`
- Methods: `add_move_command(direction, urgency)`, `to_json()`

### PySimulation
- `PySimulation()`
- Methods: `generate_terrain(config)`, `add_agent(agent)`, `create_agent(id, x, y, z)`, `get_world_data()`, `get_agent_data()`