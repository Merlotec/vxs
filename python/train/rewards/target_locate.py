from __future__ import annotations

import random
import math
from typing import Optional, Tuple

import numpy as np
import voxelsim as vxs

# Import the base reward class used by the GridWorldAStarEnv
from envs.grid_world_astar import RewardBase


class RewardTargetLocate(RewardBase):
    """
    Reward for locating a target block placed in the world.

    Behavior:
    - On reset, spawns a target block via `world.add_target([x, y, z])` at a random location.
    - Each step, when the agent reaches the target voxel, grants `reach_reward` and (optionally) respawns a new target.

    Notes:
    - Placement uses simple heuristics to choose a random location. If you want to control
      target altitude, set `z_level` explicitly; otherwise it defaults to -40 (matches common start).
    - The environment will call `update(world, agent, agent_bounds)` each step if present, which
      allows this reward to detect reaches without changing the `compute(...)` signature.
    """

    def __init__(
        self,
        *,
        living_cost: float = -0.02,
        collision_penalty: float = -100.0,
        override_penalty: float = -5.0,
        plan_fail_penalty: float = -2.0,
        reach_reward: float = 50.0,
        respawn_on_reach: bool = True,
        # Random placement controls
        # Preferred: place within max_distance of the drone (center).
        max_distance: Optional[int] = None,
        min_distance: int = 2,
        # Back-compat: if max_distance is None, fall back to annulus [rmin, rmax].
        xy_radius: Tuple[int, int] = (40, 120),  # min,max radius for XY displacement from origin
        z_level: Optional[int] = -40,
        seed: Optional[int] = None,
    ) -> None:
        self.living_cost = living_cost
        self.collision_penalty = collision_penalty
        self.override_penalty = override_penalty
        self.plan_fail_penalty = plan_fail_penalty
        self.reach_reward = reach_reward
        self.respawn_on_reach = bool(respawn_on_reach)
        self.max_distance = int(max_distance) if max_distance is not None else None
        self.min_distance = int(min_distance)
        self.xy_radius = tuple(int(x) for x in xy_radius)
        self.z_level = z_level
        self._rng = random.Random(seed)

        # Runtime state
        self._world: Optional[vxs.World] = None
        self._target_voxel: Optional[Tuple[int, int, int]] = None
        self._just_reached: bool = False

    # ----- RewardBase API -----
    def post_reset(self, world: vxs.VoxelGrid, agent: vxs.Agent) -> None:
        # Store world and place target immediately relative to the agent spawn.
        self._world = world  # type: ignore[assignment]
        self._target_voxel = None
        self._just_reached = False
        try:
            center = agent.get_coord_py()
            tx, ty, tz = self._sample_target_voxel(center=center)
            self._place_target((tx, ty, tz))
        except Exception:
            # Defer placement if something goes wrong; update() will retry.
            self._target_voxel = None

    def compute(
        self,
        *,
        collided: bool,
        overridden: bool,
        failed: bool,
        plan_len: int = 0,
    ) -> float:
        r = self.living_cost
        if collided:
            r += self.collision_penalty
        if overridden:
            r += self.override_penalty
        if failed:
            r += self.plan_fail_penalty
        # Add reach reward once when we detect it
        if self._just_reached:
            r += self.reach_reward
            self._just_reached = False
        return r

    # ----- Env integration hook (optional) -----
    # def update(self, *, world: vxs.World, agent: vxs.Agent, agent_bounds) -> None:
    #     """Called by the environment each step to update internal state.

    #     Spawns a target if none exists. Checks whether the agent has reached the target voxel.
    #     """
    #     if self._world is None:
    #         self._world = world

    #     if self._target_voxel is None:
    #         # Spawn relative to the agent's current voxel for better reachability
    #         ac = agent.get_coord_py()
    #         tx, ty, tz = self._sample_target_voxel(center=ac)
    #         self._place_target((tx, ty, tz))

    #     # Check reach: compare voxel coordinates
    #     if self._target_voxel is not None:
    #         ac = agent.get_coord_py()
    #         if (ac[0], ac[1], ac[2]) == self._target_voxel:
    #             self._just_reached = True
    #             if self.respawn_on_reach:
    #                 # Immediately respawn a new target near the current position
    #                 tx, ty, tz = self._sample_target_voxel(center=ac)
    #                 self._place_target((tx, ty, tz))

    # ----- Internals -----
    def _sample_target_voxel(self, center: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
        # Sample relative to center (agent voxel) when available
        if center is None:
            cx, cy, cz = (0, 0, self.z_level if self.z_level is not None else -40)
        else:
            cx, cy, cz = center

        # If max_distance is provided, prefer sampling within a disk of radius <= max_distance
        if self.max_distance is not None and self.max_distance > 0:
            rmin = max(1, self.min_distance)
            rmax = max(rmin, int(self.max_distance))
            # Rejection-free polar sampling; round to integer offsets
            for _ in range(32):  # try a few times to avoid collapsing to zero after rounding
                r = self._rng.uniform(rmin, rmax)
                theta = self._rng.uniform(0.0, 2.0 * math.pi)
                dx = int(round(r * math.cos(theta)))
                dy = int(round(r * math.sin(theta)))
                if abs(dx) + abs(dy) >= rmin:
                    break
            else:
                # Fallback to cardinal direction at min distance
                dirx, diry = self._rng.choice([(1,0), (-1,0), (0,1), (0,-1)])
                dx, dy = dirx * rmin, diry * rmin
        else:
            # Back-compat annulus sampling [rmin, rmax]
            rmin, rmax = self.xy_radius
            if rmax < rmin:
                rmin, rmax = rmax, rmin
            dx = self._rng.randint(-rmax, rmax)
            dy = self._rng.randint(-rmax, rmax)
            if abs(dx) + abs(dy) < max(2, rmin):
                if abs(dx) < abs(dy):
                    dx = rmin if dx >= 0 else -rmin
                else:
                    dy = rmin if dy >= 0 else -rmin

        z = self.z_level if self.z_level is not None else cz
        return int(cx + dx), int(cy + dy), int(z)

    def _place_target(self, voxel: Tuple[int, int, int]) -> None:
        if self._world is None:
            return
        # Best effort: place target and record position
        try:
            target_cell = [int(voxel[0]), int(voxel[1]), int(voxel[2])]
            self._world.add_target_cell(target_cell)
            self._target_voxel = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            print(f"Target cell added to {target_cell}")
        except Exception:
            # If placement fails, drop the target; we'll retry later
            self._target_voxel = None
            print("Failed to set voxel target cell!")
