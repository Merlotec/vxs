"""
Policy Template Contract
------------------------

Functions to implement:
  - init(config) -> None (optional): one-time setup per episode.
  - act(t, agent, world, fw, env, helpers) -> vxs.ActionIntent | tuple[vxs.ActionIntent, str] | None:
      Return either an ActionIntent or (ActionIntent, cmd) where cmd is
      "Replace" (overwrite current action) or "Push" (queue behind current).
  - collect(step_ctx) -> dict[str, str]: return small, human-readable logs for the step.
  - finalize(ep_ctx) -> dict[str, str]: episode summary text fields.

Constraints:
  - Allowed imports: voxelsim (as vxs), math, typing (optionally numpy if allowed by runner flag).
  - No network/file I/O (runner handles logging).
  - Keep logs concise (strings, <= 1–2KB per step).
"""

from typing import Any, Dict, Optional
import math
import voxelsim as vxs

_TARGET = None  # optional global target provided via init(config)


def init(config: Dict[str, Any]) -> None:
    """Optional setup. `config` may include target coords or thresholds."""
    global _TARGET
    if isinstance(config, dict) and "target" in config:
        _TARGET = config["target"]


def act(
    t: float,
    agent: vxs.Agent,
    world: vxs.VoxelGrid,
    fw: vxs.FilterWorld,
    env: vxs.EnvState,
    helpers: Any,
) -> Optional[object]:  # vxs.ActionIntent or (vxs.ActionIntent, Literal["Replace","Push"]) or None
    """Called every step. Return a new ActionIntent or None.

    Recommendation: use A* to generate a path when idle; keep sequences small and replan as needed.
    """
    if agent.get_action_py() is None:
        # Use target from init(config) if available; otherwise a default cell
        target = _TARGET or [55, 55, -20]
        origin = agent.get_coord_py()
        try:
            # Example: queue a planned path without disrupting any current action
            return helpers.plan_to(world, origin, target, yaw=0.0, urgency=0.8, padding=1), "Replace"
        except Exception:
            return helpers.intent(urgency=0.6, yaw=0.0, moves=[vxs.MoveDir.Forward]), "Replace"
    return None


def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    """Return per-step textual insights for LLM feedback."""
    pos = step_ctx.get("agent_pos")
    collisions = step_ctx.get("collisions_count")
    return {
        "pos": f"{pos}",
        "collisions": str(collisions),
        "note": "moving forward if idle",
    }


def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    """Episode summary. Return small text fields (no large blobs)."""
    success = ep_ctx.get("success", False)
    steps = ep_ctx.get("steps", 0)
    collisions = ep_ctx.get("collisions_total", 0)
    return {
        "summary": f"success={success}, steps={steps}, collisions={collisions}",
    }
