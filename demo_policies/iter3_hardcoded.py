"""
Iteration 3: Visit all 3 turbines with hardcoded positions
"""
import math
from typing import Any, Dict, Optional
import voxelsim as vxs

# Hardcoded turbine positions and inspection points
TURBINE_INSPECTION_WAYPOINTS = [
    # Turbine 1 inspection points (circle around 30,30)
    [25, 30, -66], [30, 25, -66], [35, 30, -66], [30, 35, -66],
    # Turbine 2 inspection points (circle around 120,120)
    [115, 120, -66], [120, 115, -66], [125, 120, -66], [120, 125, -66],
    # Turbine 3 inspection points (circle around 210,210)
    [205, 210, -66], [210, 205, -66], [215, 210, -66], [210, 215, -66],
    # Return home
    [110, 110, -80],
]

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(TURBINE_INSPECTION_WAYPOINTS):
        return None

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = TURBINE_INSPECTION_WAYPOINTS[current_idx]

        # Check if reached
        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 6:
            current_idx += 1
            return None

        # Use helpers.plan_to with padding=0 (no obstacle avoidance)
        try:
            intent = helpers.plan_to(world, origin, target, urgency=0.7, yaw=0.0, padding=0)
            if intent:
                return intent, "Replace"
        except Exception:
            pass

        # If planning fails, skip this waypoint
        current_idx += 1

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    turbine = "Home" if current_idx >= len(TURBINE_INSPECTION_WAYPOINTS) - 1 else f"T{(current_idx // 4) + 1}"
    return {
        "waypoint": f"{current_idx}/{len(TURBINE_INSPECTION_WAYPOINTS)}",
        "turbine": turbine
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    complete = current_idx >= len(TURBINE_INSPECTION_WAYPOINTS)
    return {"summary": f"{'Complete' if complete else 'Incomplete'}: {current_idx}/{len(TURBINE_INSPECTION_WAYPOINTS)} waypoints. Steps: {ep_ctx.get('steps', 0)}"}
