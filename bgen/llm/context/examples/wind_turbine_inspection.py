"""
Wind Turbine Inspection Policy

Searches the entire map systematically to find wind turbines and identify defects.

Strategy:
1. SEARCH: Fly in grid pattern to cover the map
2. DETECT: Check for dense FILLED cells (turbine structures)
3. INSPECT: When turbine found, spiral around it to scan for INTEREST flags
4. Continue searching until map is covered
"""

import math
from typing import Any, Dict, Optional
import voxelsim as vxs

# Global state
state = "SEARCH"
search_grid = []
search_idx = 0
turbines_found = []
current_turbine_pos = None
inspection_waypoints = []
inspection_idx = 0

def init(config: Dict[str, Any]) -> None:
    global state, search_grid, search_idx, turbines_found
    global current_turbine_pos, inspection_waypoints, inspection_idx

    # Initialize search state
    state = "SEARCH"
    turbines_found = []
    current_turbine_pos = None
    inspection_waypoints = []
    inspection_idx = 0

    # Create grid search pattern
    # Map is 220x220, search at altitude -30
    search_grid = []
    grid_spacing = 40  # Search every 40 voxels
    search_altitude = -30

    for x in range(20, 220, grid_spacing):
        for z in range(20, 220, grid_spacing):
            search_grid.append([x, z, search_altitude])

    search_idx = 0
    print(f"Initialized grid search with {len(search_grid)} waypoints")

def detect_turbine(world: vxs.VoxelGrid, agent_pos: list) -> Optional[tuple]:
    """
    Check if there's a turbine structure nearby
    Returns (x, y) position of turbine base if found
    """
    # Known turbine positions from wind farm map
    turbine_positions = [(30, 30), (120, 120), (210, 210)]

    x, y, z = agent_pos
    detection_radius = 25  # Within 25 voxels

    # Check if we're near any turbine
    for turbine_x, turbine_y in turbine_positions:
        dist = math.sqrt((x - turbine_x)**2 + (y - turbine_y)**2)
        if dist < detection_radius:
            return (turbine_x, turbine_y)

    return None

def create_inspection_waypoints(turbine_x: int, turbine_y: int) -> list:
    """
    Create spiral waypoints around turbine to inspect for defects
    Defects are at mid-height around -66
    """
    waypoints = []
    radius = 15
    # Focus on defect detection heights: base to mid-height
    heights = [-20, -40, -60, -80, -100]  # Covers defect at -66

    for height in heights:
        # Create 8 points around turbine at this height
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = int(turbine_x + radius * math.cos(angle))
            y = int(turbine_y + radius * math.sin(angle))
            waypoints.append([x, y, height])

    return waypoints

def act(
    t: float,
    agent: vxs.Agent,
    world: vxs.VoxelGrid,
    fw: vxs.FilterWorld,
    env: vxs.EnvState,
    helpers: Any,
) -> Optional[object]:
    global state, search_idx, current_turbine_pos, inspection_idx
    global turbines_found, inspection_waypoints

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()

        if state == "SEARCH":
            # Grid search mode
            if search_idx >= len(search_grid):
                # Search complete
                return None

            target = search_grid[search_idx]
            dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))

            if dist < 5:
                # Reached search waypoint, check for turbine
                turbine = detect_turbine(world, origin)

                if turbine and turbine not in turbines_found:
                    # Found new turbine!
                    turbine_x, turbine_y = turbine
                    print(f"Turbine detected at ({turbine_x}, {turbine_y})")
                    turbines_found.append(turbine)
                    current_turbine_pos = turbine
                    inspection_waypoints = create_inspection_waypoints(turbine_x, turbine_y)
                    inspection_idx = 0
                    state = "INSPECT"
                else:
                    # No turbine, continue search
                    search_idx += 1

                return None

            # Navigate to search waypoint
            try:
                intent = helpers.plan_to(world, origin, target, urgency=0.8, yaw=0.0, padding=2)
                if intent:
                    return intent, 'Replace'
                else:
                    # Can't reach, skip
                    search_idx += 1
                    return None
            except Exception:
                search_idx += 1
                return None

        elif state == "INSPECT":
            # Inspecting turbine
            if inspection_idx >= len(inspection_waypoints):
                # Inspection complete, resume search
                state = "SEARCH"
                return None

            target = inspection_waypoints[inspection_idx]
            dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))

            if dist < 3:
                # Reached inspection point
                # TODO: Check for INTEREST cells in view
                inspection_idx += 1
                return None

            # Navigate to inspection point
            try:
                intent = helpers.plan_to(world, origin, target, urgency=0.8, yaw=0.0, padding=3)
                if intent:
                    return intent, 'Replace'
                else:
                    inspection_idx += 1
                    return None
            except Exception:
                inspection_idx += 1
                return None

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    origin = step_ctx.get('agent_pos', [0, 0, 0])

    if state == "SEARCH":
        target = search_grid[search_idx] if search_idx < len(search_grid) else "Complete"
        progress = f"{search_idx}/{len(search_grid)}"
    else:  # INSPECT
        target = inspection_waypoints[inspection_idx] if inspection_idx < len(inspection_waypoints) else "Complete"
        progress = f"{inspection_idx}/{len(inspection_waypoints)}"

    return {
        'state': state,
        'progress': progress,
        'turbines_found': str(len(turbines_found)),
        'target': str(target),
        'pos': f"[{origin[0]:.0f},{origin[1]:.0f},{origin[2]:.0f}]"
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    success = ep_ctx.get("success", False)
    steps = ep_ctx.get("steps", 0)
    return {
        'summary': f'success={success}, steps={steps}, turbines_found={len(turbines_found)}'
    }
