"""
Demo Policy 3: PERFECT MISSION WITH FULL SCANNING
Flies around each turbine in a circle pattern to scan, then returns home
"""
from typing import Any, Dict, Optional
import voxelsim as vxs
import math

# Turbine base positions
TURBINE_BASES = [
    (30, 30),
    (120, 120),
    (210, 210),
]

# Scanning parameters
SCAN_RADIUS = 10  # Distance from turbine
SCAN_POINTS = 8   # Number of points in circle
SCAN_HEIGHT = -66  # Mid-height of turbine

# Generate all waypoints upfront
WAYPOINTS = []
for tx, tz in TURBINE_BASES:
    # Create circle of waypoints around this turbine
    for i in range(SCAN_POINTS):
        angle = 2 * math.pi * i / SCAN_POINTS
        x = int(tx + SCAN_RADIUS * math.cos(angle))
        z = int(tz + SCAN_RADIUS * math.sin(angle))
        WAYPOINTS.append([x, z, SCAN_HEIGHT])

# Add home position at the end
WAYPOINTS.append([110, 110, -80])

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(WAYPOINTS):
        return None  # Mission complete

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = WAYPOINTS[current_idx]

        # Check if reached current waypoint
        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 5:
            current_idx += 1
            return None

        # Plan to current waypoint
        intent = helpers.plan_to(world, origin, target, urgency=0.7, yaw=0.0, padding=2)
        if intent:
            return intent, "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    turbine_num = current_idx // SCAN_POINTS + 1 if current_idx < len(WAYPOINTS) - 1 else "Home"
    scan_point = (current_idx % SCAN_POINTS) + 1 if current_idx < len(WAYPOINTS) - 1 else "N/A"

    return {
        "status": f"Turbine_{turbine_num}_Point_{scan_point}" if turbine_num != "Home" else "Returning_Home",
        "progress": f"{current_idx}/{len(WAYPOINTS)}",
        "turbines_scanned": str(min(current_idx // SCAN_POINTS, 3))
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    success = current_idx >= len(WAYPOINTS)
    turbines_done = min(current_idx // SCAN_POINTS, 3)
    return {"summary": f"Mission {'SUCCESS' if success else 'INCOMPLETE'}: Scanned {turbines_done}/3 turbines. Steps: {ep_ctx.get('steps', 0)}"}
