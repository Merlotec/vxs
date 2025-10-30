"""
Wind Farm Scenario
Generates a map with wind turbines and defects for inspection demos
"""

import voxelsim as vxs


def create_wind_farm_map():
    """
    Create a demonstration wind farm with 3 turbines
    - Turbine 1: (30, 30) - No defect (clean)
    - Turbine 2: (120, 120) - Has defect at mid-height
    - Turbine 3: (210, 210) - No defect (clean)

    Returns:
        VoxelGrid: The generated wind farm world
    """
    # Generate flat terrain
    terrain_gen = vxs.TerrainGenerator()
    config = vxs.TerrainConfig.default_py()
    config.set_world_dimensions_py(220, 60, 220)
    terrain_gen.generate_terrain_py(config)
    world = terrain_gen.generate_world_py()

    # Extract existing terrain cells
    coords, values = world.as_numpy()
    cells_dict = {}
    for i in range(len(coords)):
        x, y, z = int(coords[i][0]), int(coords[i][1]), int(coords[i][2])
        if values[i] >= 1.0:
            cells_dict[(x, y, z)] = vxs.Cell.filled()
        elif values[i] >= 0.5:
            cells_dict[(x, y, z)] = vxs.Cell.sparse()

    # Turbine parameters - diagonal placement
    turbine_positions = [
        (30, 30, False),
        (120, 120, True),    # Middle turbine has defect
        (210, 210, False),
    ]
    base_height = 6
    tower_height = 120
    radius = 2

    # Add turbines
    for tx, tz, has_defect in turbine_positions:
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

            # Make defect visible with multiple cells
            # Use INTEREST flag only (it will render red)
            cells_dict[(defect_x, tz, defect_world_z)] = vxs.Cell.interest()
            cells_dict[(defect_x, tz, defect_world_z - 1)] = vxs.Cell.interest()
            cells_dict[(defect_x, tz, defect_world_z + 1)] = vxs.Cell.interest()

    # Create new world from combined cells
    world = vxs.VoxelGrid.from_dict_py(cells_dict)

    return world
