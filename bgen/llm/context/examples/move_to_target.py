from typing import Any, Dict, Optional
import voxelsim as vxs

TARGET = [60, 60, -20]


def init(config: Dict[str, Any]) -> None:
    pass


def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[vxs.ActionIntent]:
    # If not moving, plan to target with A*
    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        intent = helpers.plan_to(world, origin, TARGET, yaw=0.0, urgency=0.9, padding=1)
        return intent
    return None


def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    pos = step_ctx.get("agent_pos")
    dist = step_ctx.get("distance_to_target")
    return {"pos": f"{pos}", "dist": f"{dist:.2f}" if dist is not None else "n/a"}


def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"summary": f"steps={ep_ctx.get('steps')}, success={ep_ctx.get('success')}"}

