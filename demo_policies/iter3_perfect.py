"""
Demo Policy 3: PERFECT MISSION
Visits all 3 turbines and returns home
"""
from typing import Any, Dict, Optional
import voxelsim as vxs
import math

# Hardcoded waypoints: 3 turbines + home position
# Fly at mid-height (-66) near defect location for better visual inspection
WAYPOINTS = [
    [35, 35, -66],      # Turbine 1 - fly close at mid-height
    [125, 125, -66],    # Turbine 2 - fly close at mid-height (has defect here)
    [215, 215, -66],    # Turbine 3 - fly close at mid-height
    [110, 110, -80],    # Home (return to start altitude)
]

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
        if dist < 8:
            current_idx += 1
            return None

        # Plan to current waypoint using helpers
        intent = helpers.plan_to(world, origin, target, urgency=0.6, yaw=0.0, padding=2)
        if intent:
            return intent, "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    waypoint_names = ["Turbine_1", "Turbine_2", "Turbine_3", "Home"]
    current_name = waypoint_names[current_idx] if current_idx < len(waypoint_names) else "Complete"

    return {
        "status": f"navigating_to_{current_name}",
        "turbines_visited": str(min(current_idx, 3)),
        "progress": f"{current_idx}/{len(WAYPOINTS)}"
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    success = current_idx >= len(WAYPOINTS)
    return {"summary": f"Mission {'SUCCESS' if success else 'INCOMPLETE'}: Visited all turbines and returned home. Steps: {ep_ctx.get('steps', 0)}"}
