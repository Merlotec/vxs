#!/usr/bin/env python3
"""
Wind Turbine Inspection Demo
Generates a map with wind turbines and defects for investor demo
"""

import voxelsim as vxs
import math


def create_wind_farm_map():
    """
    Create a demonstration wind farm with 3 turbines
    - Turbine 1: No defect (clean)
    - Turbine 2: Has defect at mid-height
    - Turbine 3: No defect (clean)
    """
    print("Generating wind farm terrain...")

    # Generate flat terrain
    terrain_gen = vxs.TerrainGenerator()
    config = vxs.TerrainConfig.default_py()
    config.set_world_dimensions_py(220, 60, 220)  # x, y, z (y is height)
    terrain_gen.generate_terrain_py(config)
    world = terrain_gen.generate_world_py()

    print("Extracting terrain cells...")
    # Extract existing terrain cells
    coords, values = world.as_numpy()
    cells_dict = {}
    for i in range(len(coords)):
        x, y, z = int(coords[i][0]), int(coords[i][1]), int(coords[i][2])
        # values[i]: 1.0=FILLED, 0.5=SPARSE
        if values[i] >= 1.0:
            cells_dict[(x, y, z)] = vxs.Cell.filled()
        elif values[i] >= 0.5:
            cells_dict[(x, y, z)] = vxs.Cell.sparse()

    print("Adding wind turbines...")

    # Turbine parameters - diagonal placement
    turbine_positions = [
        (30, 30, False),   # (x, z, has_defect)
        (120, 120, True),    # Middle turbine has defect
        (210, 210, False),
    ]
    base_height = 6
    tower_height = 120
    radius = 2

    # Add turbines
    for tx, tz, has_defect in turbine_positions:
        print(f"  Creating turbine at ({tx}, {tz}) defect={has_defect}")

        # Build tower (cylinder) - vertical along Z axis (NED coords)
        # In NED: X=East, Y=North, Z=Down (so negative Z is UP)
        for height_offset in range(tower_height):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        # Place tower vertically in -Z direction (upward)
                        world_z = -(base_height + height_offset)
                        cells_dict[(tx + dx, tz + dy, world_z)] = vxs.Cell.filled()

        # Add defect markers
        if has_defect:
            defect_height_offset = tower_height // 2
            defect_world_z = -(base_height + defect_height_offset)
            defect_x = tx + radius
            print(f"    Adding defect at ({defect_x}, {tz}, {defect_world_z})")

            # Make defect visible with multiple cells
            # Use INTEREST flag only (it will render red)
            cells_dict[(defect_x, tz, defect_world_z)] = vxs.Cell.interest()
            cells_dict[(defect_x, tz, defect_world_z - 1)] = vxs.Cell.interest()
            cells_dict[(defect_x, tz, defect_world_z + 1)] = vxs.Cell.interest()

    print(f"Wind farm created! Total turbines: {len(turbine_positions)}")
    print(f"Turbines with defects: {sum(1 for _, _, d in turbine_positions if d)}")

    # Create new world from combined cells
    print("Building final world...")
    world = vxs.VoxelGrid.from_dict_py(cells_dict)

    return world


def main():
    """Main demo entry point"""
    print("=== Wind Turbine Inspection Demo ===\n")

    # Create the world
    world = create_wind_farm_map()

    # Create renderer client and connect
    print("\nConnecting to renderer...")
    client = vxs.RendererClient.default_localhost_py(pov_count=1)

    # Send world to renderer
    print("Sending world to renderer...")
    client.send_world_py(world)

    # Create an agent (drone)
    print("Creating inspection drone...")
    agent = vxs.Agent(0)
    agent.set_hold_py([50, 50, -15], 0.0)  # Start position above center
    # agent.set_dims_py([2.0, 2.0, 2.0])

    # Send agent to renderer
    client.send_agents_py({0: agent})

    print("\n" + "="*50)
    print("DEMO READY!")
    print("="*50)
    print("\nInspection Task:")
    print("  Find and inspect all wind turbines in the area")
    print("  Defects are marked in RED (INTEREST cells)")
    print("\nTurbine locations:")
    print("  - Turbine 1: (30, 30) - CLEAN")
    print("  - Turbine 2: (60, 30) - DEFECT DETECTED")
    print("  - Turbine 3: (90, 30) - CLEAN")
    print("\nPress Ctrl+C to exit")
    print("="*50 + "\n")

    # Keep running
    try:
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
