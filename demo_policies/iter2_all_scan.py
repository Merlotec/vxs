"""
Demo Policy 2: SCANS ALL TURBINES BUT DOESN'T RETURN HOME
Flies around each turbine in a circle pattern, then stops at last turbine
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
SCAN_RADIUS = 10
SCAN_POINTS = 8
SCAN_HEIGHT = -66

# Generate all waypoints (NO home at the end)
WAYPOINTS = []
for tx, tz in TURBINE_BASES:
    for i in range(SCAN_POINTS):
        angle = 2 * math.pi * i / SCAN_POINTS
        x = int(tx + SCAN_RADIUS * math.cos(angle))
        z = int(tz + SCAN_RADIUS * math.sin(angle))
        WAYPOINTS.append([x, z, SCAN_HEIGHT])

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(WAYPOINTS):
        return None  # All turbines scanned, stop here (no return home)

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = WAYPOINTS[current_idx]

        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 5:
            current_idx += 1
            return None

        intent = helpers.plan_to(world, origin, target, urgency=0.7, yaw=0.0, padding=2)
        if intent:
            return intent, "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    turbine_num = current_idx // SCAN_POINTS + 1
    scan_point = (current_idx % SCAN_POINTS) + 1

    return {
        "status": f"Turbine_{turbine_num}_Point_{scan_point}",
        "progress": f"{current_idx}/{len(WAYPOINTS)}",
        "turbines_scanned": str(min(current_idx // SCAN_POINTS, 3))
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    turbines_done = min(current_idx // SCAN_POINTS, 3)
    return {"summary": f"Scanned {turbines_done}/3 turbines, didn't return home. Steps: {ep_ctx.get('steps', 0)}"}
