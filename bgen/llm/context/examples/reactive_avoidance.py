from typing import Any, Dict, Optional
import voxelsim as vxs


def init(config: Dict[str, Any]) -> None:
    pass


def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[vxs.ActionIntent]:
    if agent.get_action_py() is None:
        # Check for immediate collisions around current pos and sidestep
        cols = world.collisions_py(agent.get_pos(), [0.6, 0.6, 0.6])
        if cols:
            seq = [vxs.MoveDir.Left]
        else:
            seq = [vxs.MoveDir.Forward]
        return helpers.intent(0.7, 0.0, seq)
    return None


def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"collisions": str(step_ctx.get("collisions_count"))}


def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"summary": f"collisions_total={ep_ctx.get('collisions_total')}"}

