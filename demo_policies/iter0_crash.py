"""
Demo Policy 0: CRASH IMMEDIATELY
Just flies forward - will crash into obstacles
"""
from typing import Any, Dict, Optional
import voxelsim as vxs

def init(config: Dict[str, Any]) -> None:
    pass

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    # Always fly forward - no obstacle avoidance
    if agent.get_action_py() is None:
        return helpers.intent(0.5, 0.0, [vxs.MoveDir.Forward]), "Replace"
    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"status": "flying_forward"}

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"summary": f"Crashed after {ep_ctx.get('steps', 0)} steps"}
