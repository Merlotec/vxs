"""
Iteration 4: Full ground scan first, then all turbines vertically
"""
import math
from typing import Any, Dict, Optional
import voxelsim as vxs

# Ground scan: full coverage but less dense (faster)
GROUND_GRID = []
GROUND_HEIGHT = -15
# Scan entire 220x220 map in grid pattern
for x in range(20, 220, 35):  # Every 35 voxels - covers whole map faster
    for z in range(20, 220, 35):
        GROUND_GRID.append([x, z, GROUND_HEIGHT])

# Turbine vertical scans (less waypoints per turbine)
TURBINE_BASES = [(30, 30), (120, 120), (200, 200)]  # Moved turbine 3 away from edge
SCAN_HEIGHTS = [-60, -90]  # Just 2 heights instead of 4
SCAN_RADIUS = 8

TURBINE_WAYPOINTS = []
for tx, tz in TURBINE_BASES:
    for height in SCAN_HEIGHTS:
        # Just 2 positions per height instead of 4
        TURBINE_WAYPOINTS.append([tx - SCAN_RADIUS, tz, height])
        TURBINE_WAYPOINTS.append([tx + SCAN_RADIUS, tz, height])

# Combine: ground first, then turbines, then home
WAYPOINTS = GROUND_GRID + TURBINE_WAYPOINTS + [[110, 110, -80]]

current_idx = 0
ground_waypoints = len(GROUND_GRID)
turbine_start_idx = ground_waypoints

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0
    print(f"Ground-first scan: {len(GROUND_GRID)} ground + {len(TURBINE_WAYPOINTS)} turbine = {len(WAYPOINTS)} total")

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(WAYPOINTS):
        return None

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = WAYPOINTS[current_idx]

        # Check if reached
        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 6:
            current_idx += 1
            return None

        # Plan with padding=0
        try:
            intent = helpers.plan_to(world, origin, target, urgency=0.8, yaw=0.0, padding=0)
            if intent:
                return intent, "Replace"
        except Exception:
            pass

        # Skip if planning fails
        current_idx += 1

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx, ground_waypoints, turbine_start_idx

    if current_idx >= len(WAYPOINTS) - 1:
        phase = "Return_Home"
        detail = "Done"
    elif current_idx < ground_waypoints:
        phase = "Ground_Scan"
        detail = f"{current_idx}/{ground_waypoints}"
    else:
        phase = "Turbine_Scan"
        turbine_idx = current_idx - turbine_start_idx
        turbine_num = (turbine_idx // 4) + 1  # Each turbine now has 4 waypoints (2 heights x 2 positions)
        height_idx = (turbine_idx % 4) // 2
        detail = f"T{turbine_num}_H{SCAN_HEIGHTS[height_idx]}"

    return {
        "waypoint": f"{current_idx}/{len(WAYPOINTS)}",
        "phase": phase,
        "detail": detail
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx, ground_waypoints
    complete = current_idx >= len(WAYPOINTS)

    if current_idx < ground_waypoints:
        status = f"Ground scan incomplete: {current_idx}/{ground_waypoints}"
    else:
        turbines_done = min((current_idx - ground_waypoints) // 4, 3)  # Each turbine has 4 waypoints
        status = f"Ground complete, turbines: {turbines_done}/3"

    return {"summary": f"{status}. Total: {current_idx}/{len(WAYPOINTS)}. Steps: {ep_ctx.get('steps', 0)}"}
