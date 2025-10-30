"""
Demo Policy 1: SCANS FIRST TURBINE ONLY
Flies around first turbine in a circle, then stops
"""
from typing import Any, Dict, Optional
import voxelsim as vxs
import math

# Only first turbine
TURBINE_BASE = (30, 30)

# Scanning parameters
SCAN_RADIUS = 10
SCAN_POINTS = 8
SCAN_HEIGHT = -66

# Generate waypoints around first turbine only
WAYPOINTS = []
for i in range(SCAN_POINTS):
    angle = 2 * math.pi * i / SCAN_POINTS
    x = int(TURBINE_BASE[0] + SCAN_RADIUS * math.cos(angle))
    z = int(TURBINE_BASE[1] + SCAN_RADIUS * math.sin(angle))
    WAYPOINTS.append([x, z, SCAN_HEIGHT])

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(WAYPOINTS):
        return None  # First turbine scanned, stop

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
    scan_point = current_idx + 1 if current_idx < len(WAYPOINTS) else "Complete"

    return {
        "status": f"Turbine_1_Point_{scan_point}",
        "progress": f"{current_idx}/{len(WAYPOINTS)}",
        "turbines_scanned": "1" if current_idx >= len(WAYPOINTS) else "0"
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    complete = current_idx >= len(WAYPOINTS)
    return {"summary": f"Scanned turbine 1 {'completely' if complete else 'partially'}. Steps: {ep_ctx.get('steps', 0)}"}
