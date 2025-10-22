from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import voxelsim as vxs


@dataclass
class Helpers:
    def move(self) -> vxs.MoveDir:
        return vxs.MoveDir

    def intent(self, urgency: float, yaw: float, moves: List[vxs.MoveDir]) -> vxs.ActionIntent:
        return vxs.ActionIntent(urgency, yaw, moves, None)

    def plan_to(
        self,
        world: vxs.VoxelGrid,
        origin: List[int],
        dst: List[int],
        yaw: float,
        urgency: float,
        padding: int = 0,
    ) -> vxs.ActionIntent:
        planner = vxs.AStarActionPlanner(padding)
        return planner.plan_action_py(world, origin, dst, yaw, urgency)

