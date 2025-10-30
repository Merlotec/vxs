"""
Iteration 3: Full 3D scan of all turbines at multiple heights
"""
import math
from typing import Any, Dict, Optional
import voxelsim as vxs

# Generate full scan waypoints for all 3 turbines
TURBINE_BASES = [(30, 30), (120, 120), (210, 210)]
SCAN_HEIGHTS = [-50, -66, -80, -100, -120]  # 5 heights including defect level (-66)
SCAN_RADIUS = 8
GROUND_HEIGHT = -15  # Low altitude for ground scan

WAYPOINTS = []

for tx, tz in TURBINE_BASES:
    # First, scan ground around turbine
    WAYPOINTS.append([tx - 15, tz - 15, GROUND_HEIGHT])
    WAYPOINTS.append([tx + 15, tz - 15, GROUND_HEIGHT])
    WAYPOINTS.append([tx + 15, tz + 15, GROUND_HEIGHT])
    WAYPOINTS.append([tx - 15, tz + 15, GROUND_HEIGHT])

    # Then scan turbine vertically at multiple heights
    for height in SCAN_HEIGHTS:
        # 8 points around turbine for better coverage (including diagonals)
        import math as m
        for i in range(8):
            angle = 2 * m.pi * i / 8
            x = int(tx + SCAN_RADIUS * m.cos(angle))
            z = int(tz + SCAN_RADIUS * m.sin(angle))
            WAYPOINTS.append([x, z, height])

# Return home
WAYPOINTS.append([110, 110, -80])

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0
    print(f"Full scan initialized: {len(WAYPOINTS)} waypoints")

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
    global current_idx

    if current_idx >= len(WAYPOINTS) - 1:
        turbine_num = "Home"
        scan_type = "Return"
    else:
        # Each turbine has 44 waypoints (4 ground + 40 vertical: 5 heights x 8 positions)
        turbine_num = (current_idx // 44) + 1
        waypoint_in_turbine = current_idx % 44

        if waypoint_in_turbine < 4:
            # Ground scan first
            scan_type = "Ground"
        else:
            # Then vertical scan
            height_idx = (waypoint_in_turbine - 4) // 8
            scan_type = f"Height_{SCAN_HEIGHTS[height_idx]}"

        turbine_num = f"T{turbine_num}"

    return {
        "waypoint": f"{current_idx}/{len(WAYPOINTS)}",
        "turbine": str(turbine_num),
        "scan": scan_type
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    complete = current_idx >= len(WAYPOINTS)
    turbines_done = min(current_idx // 44, 3)
    return {"summary": f"Scanned {turbines_done}/3 turbines (detailed scan). Waypoints: {current_idx}/{len(WAYPOINTS)}. Steps: {ep_ctx.get('steps', 0)}"}
