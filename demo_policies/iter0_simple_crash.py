"""
Iteration 0: Just flies forward, crashes immediately
"""
from typing import Any, Dict, Optional
import voxelsim as vxs

def init(config: Dict[str, Any]) -> None:
    pass

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    if agent.get_action_py() is None:
        return helpers.intent(0.5, 0.0, [vxs.MoveDir.Forward, vxs.MoveDir.Down]), "Replace"
    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"status": "crashing"}

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"summary": "Crashed"}
