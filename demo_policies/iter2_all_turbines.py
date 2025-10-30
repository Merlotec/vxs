"""
Demo Policy 2: VISITS ALL TURBINES BUT DOESN'T RETURN HOME
Visits all 3 turbines in sequence, then stops at the last one
"""
from typing import Any, Dict, Optional
import voxelsim as vxs
import math

# Hardcoded turbine positions (from wind farm at 30,30 / 120,120 / 210,210)
# Fly at mid-height (-66) for better visual inspection
TURBINES = [
    [35, 35, -66],
    [125, 125, -66],
    [215, 215, -66],
]

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(TURBINES):
        return None  # All turbines visited, stop here

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = TURBINES[current_idx]

        # Check if reached current turbine
        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 8:
            current_idx += 1
            return None

        # Plan to current turbine using helpers
        intent = helpers.plan_to(world, origin, target, urgency=0.6, yaw=0.0, padding=2)
        if intent:
            return intent, "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    return {
        "status": f"navigating_to_turbine_{current_idx + 1}",
        "turbines_visited": str(current_idx),
        "progress": f"{current_idx}/{len(TURBINES)}"
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    return {"summary": f"Visited {current_idx}/{len(TURBINES)} turbines, didn't return home. Steps: {ep_ctx.get('steps', 0)}"}
