"""
Demo Policy 1: REACHES FIRST TURBINE ONLY
Uses A* to reach the first turbine, then stops
"""
from typing import Any, Dict, Optional
import voxelsim as vxs
import math

# Hardcoded first turbine position (from wind farm)
# Turbine is at (30, 30), fly nearby at mid-height for inspection
TURBINE_1 = [35, 35, -66]
reached = False

def init(config: Dict[str, Any]) -> None:
    global reached
    reached = False

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global reached

    if reached:
        return None  # Already reached, do nothing

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = TURBINE_1

        # Check if we reached it
        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 8:
            reached = True
            return None

        # Plan to turbine using helpers
        intent = helpers.plan_to(world, origin, target, urgency=0.6, yaw=0.0, padding=2)
        if intent:
            return intent, "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    status = "reached_turbine_1" if reached else "navigating_to_turbine_1"
    return {"status": status, "turbines_visited": "1" if reached else "0"}

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"summary": f"Reached turbine 1, stopped. Steps: {ep_ctx.get('steps', 0)}"}
