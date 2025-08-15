from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import voxelsim as vxs


class SimpleReward:
    def __init__(
        self,
        living_cost: float = -0.02,
        collision_penalty: float = -100.0,
        override_penalty: float = -5.0,
        plan_fail_penalty: float = -2.0,
        # Optional positive shaping to encourage longer planned paths
        plan_success_bonus: float = 0.0,
        distance_bonus_per_step: float = 0.0,
    ):
        self.living_cost = living_cost
        self.collision_penalty = collision_penalty
        self.override_penalty = override_penalty
        self.plan_fail_penalty = plan_fail_penalty
        self.plan_success_bonus = plan_success_bonus
        self.distance_bonus_per_step = distance_bonus_per_step

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
        else:
            if plan_len > 0:
                r += self.plan_success_bonus
                r += self.distance_bonus_per_step * float(plan_len)
        return r


class GridWorldAStarEnv(gym.Env):
    """
    Minimal RL env that takes a continuous action and plans a route via A*.

    Action (Box 6): [dx, dy, dz, urgency, yaw, priority]
      - dx/dy/dz: relative voxel offsets (floats, rounded to ints, clamped to ±max_offset)
      - urgency: [0, 1]
      - yaw: [-pi, pi]
      - priority: [0, 1] if >= override_threshold and an action is running, override it

    Each step, we compute dst = round(agent_voxel + [dx,dy,dz]) and call
    AStarPlanner.plan_py(origin, dst, urgency, yaw) to produce an ActionIntent,
    then perform the action with perform_py (optionally overriding current action).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        planner_padding: int = 1,
        override_threshold: float = 0.8,
        max_offset: int = 12,
        action_gain: float = 1.0,
        attempt_scales: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
        allow_override: bool = False,
        start_pos: Tuple[int, int, int] = (0, 0, 0),
        delta_time: float = 0.01,
        filter_update_lag: float = 0.2,
        filter_delta: float = 0.25,
        reward: Optional[SimpleReward] = None,
        render_client: Optional[vxs.RendererClient] = None,
        dense_half_dims: Tuple[int, int, int] = (40, 40, 30),
        noise: Optional[vxs.NoiseParams] = None,
    ):
        super().__init__()

        self.override_threshold = float(override_threshold)
        self.max_offset = int(max_offset)
        self.astar = vxs.AStarActionPlanner(int(planner_padding))
        self.delta_time = float(delta_time)
        self.filter_update_lag = float(filter_update_lag)
        self.filter_delta = float(filter_delta)
        self.reward_fn = reward or SimpleReward()
        self.start_pos = tuple(start_pos)
        self.dense_half_dims = tuple(int(x) for x in dense_half_dims)
        self.client = render_client
        self.noise = noise or vxs.NoiseParams.default_with_seed_py([0, 0, 0])
        self.action_gain = float(action_gain)
        self.attempt_scales = tuple(float(s) for s in attempt_scales)
        self.allow_override = bool(allow_override)

        # Continuous action space: [dx,dy,dz,urgency,yaw,priority]
        low = np.array(
            [-self.max_offset, -self.max_offset, -self.max_offset, 0.5, -math.pi, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [self.max_offset, self.max_offset, self.max_offset, 1.0, math.pi, 1.0],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Observation: grid + last_action
        hx, hy, hz = self.dense_half_dims
        grid_shape = (2 * hx + 1, 2 * hy + 1, 2 * hz + 1)
        grid_space = gym.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            shape=grid_shape,
            dtype=np.int32,
        )
        # Record relative offsets for last_action (not absolute voxels)
        la_low = np.array(
            [-self.max_offset, -self.max_offset, -self.max_offset, 0.0, -math.pi, 0.0],
            dtype=np.float32,
        )
        la_high = np.array(
            [self.max_offset, self.max_offset, self.max_offset, 1.0, math.pi, 1.0],
            dtype=np.float32,
        )
        last_action_space = gym.spaces.Box(low=la_low, high=la_high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "grid": grid_space,
                "last_action": last_action_space,
            }
        )

        # World/agent state
        self.world: Optional[vxs.World] = None
        self.filter_world: Optional[vxs.FilterWorld] = None
        self.agent: Optional[vxs.Agent] = None
        self.vision: Optional[vxs.AgentVisionRenderer] = None
        self.camera_proj = vxs.CameraProjection.default_py()
        self.camera_orientation = vxs.CameraOrientation.vertical_tilt_py(-0.5)
        self.agent_bounds = (0.5, 0.5, 0.4)

        self.chaser = vxs.FixedLookaheadChaser.default_py()
        self.dynamics = vxs.px4.Px4Dynamics.default_py()

        self.world_time: float = 0.0
        self.next_changeset: Optional[vxs.WorldChangeset] = None
        self.filter_world_upd_ts: Optional[float] = None

        self._last_action = np.zeros(6, dtype=np.float32)

        self._init_world()

    # --------------- World setup / rendering ---------------------------------
    def _init_world(self):
        self._gen_world(seed=random.randint(1, 10**6))
        self.agent = vxs.Agent(0)
        self.agent.set_hold_py(self.start_pos, 0.0)
        self.filter_world = vxs.FilterWorld()
        self.vision = vxs.AgentVisionRenderer(self.world, (160, 100), self.noise)
        if self.client:
            self.client.send_world_py(self.world)

    def _gen_world(self, seed: int):
        terrain = vxs.TerrainGenerator()
        terrain.generate_terrain_py(vxs.TerrainConfig(seed))
        self.world = terrain.generate_world_py()

    def _render_agent_vision(self):
        self.filter_world_upd_ts = self.world_time
        self.vision.render_changeset_py(
            self.agent.camera_view_py(self.camera_orientation),
            self.camera_proj,
            self.filter_world,
            self.filter_world_upd_ts,
            lambda ch: self._update_callback(ch),
        )

    def _update_callback(self, changeset: vxs.WorldChangeset):
        self.next_changeset = changeset

    def _await_changeset(self) -> vxs.WorldChangeset:
        while self.next_changeset is None:
            time.sleep(0.00001)
        ch = self.next_changeset
        self.next_changeset = None
        self.filter_world_upd_ts = None
        return ch

    # --------------- RL API ---------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._gen_world(seed or random.randint(1, 10**6))
        if self.client:
            self.client.send_world_py(self.world)
        self.filter_world = vxs.FilterWorld()
        self.vision = vxs.AgentVisionRenderer(self.world, (160, 100), self.noise)
        self.agent = vxs.Agent(0)
        self.agent.set_hold_py(self.start_pos, 0.0)
        self.world_time = 0.0
        self._last_action[:] = 0.0
        self.chaser = vxs.FixedLookaheadChaser.default_py()
        self.dynamics = vxs.px4.Px4Dynamics.default_py()


        # Prime first dense snapshot
        self._render_agent_vision()
        ch = self._await_changeset()
        ch.update_filter_world_py(self.filter_world)
        obs = self._build_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == 6
        dx, dy, dz, urgency, yaw, priority = action
        urgency = float(np.clip(urgency, 0.0, 1.0))
        # urgency = 0.8
        yaw = float(np.clip(yaw, -math.pi, math.pi))
        # yaw = 0
        priority = float(np.clip(priority, 0.0, 1.0))
        # Compute destination voxel coord relative to current voxel
        origin = np.array(self.agent.get_coord_py())
        # Optional gain to make proposed offsets larger (clamped to bounds)
        scaled = self.action_gain * np.array([dx, dy, dz], dtype=np.float32)
        offset = np.round(np.clip(scaled, -self.max_offset, self.max_offset)).astype(np.int32)
        dst = tuple((origin + offset).tolist())

        # Record last action as the relative offset requested (updated to actual used below)
        self._last_action[:] = [float(offset[0]), float(offset[1]), float(offset[2]), urgency, yaw, priority]

        overridden = False
        failed = False

        ds_len = math.sqrt(dx**2 + dy**2 + dz**2)
        scaled_len = math.sqrt((self.action_gain*dx)**2 + (self.action_gain*dy)**2 + (self.action_gain*dz)**2)
        print(f"requested_offset_len: {ds_len:.3f} | scaled_offset_len: {scaled_len:.3f} (gain {self.action_gain})")
        # Plan if allowed (override or idle)
        curr = self.agent.get_action_py()
        if (not curr):# or (self.allow_override and (priority >= self.override_threshold)):
            if curr and priority >= self.override_threshold:
                overridden = True

            # Try planning, penalise on exceptions and fall back by reducing offset
            # Try progressively smaller offsets if planning fails
            attempts = [np.round(offset * s).astype(np.int32) for s in self.attempt_scales]

            planned = False
            chosen_plan_len = 0
            for off in attempts:
                dst_try = tuple((origin + off).tolist())
                try:
                    dirs = self.astar.plan_action_py(self.world, tuple(origin.tolist()), dst_try, urgency, yaw).get_move_sequence()
                except Exception:
                    failed = True
                    continue
                if len(dirs) > 0:
                    intent = vxs.ActionIntent(float(urgency), float(yaw), dirs)
                    try:
                        self.agent.perform_py(intent)
                        # record the actual relative offset used
                        self._last_action[:3] = [float(off[0]), float(off[1]), float(off[2])]
                        chosen_plan_len = len(dirs)
                        planned = True
                        break
                    except Exception:
                        failed = True
                        continue
            if not planned and not failed:
                # Planning returned empty list without exception -> treat as failure for shaping
                failed = True

        # Advance simulation one tick
        chase = self.chaser.step_chase_py(self.agent, self.delta_time)
        self.dynamics.update_agent_dynamics_py(self.agent, vxs.EnvState.default_py(), chase, self.delta_time)

        self.world_time += self.delta_time

        # Filter world update/render scheduling (delay then process)
        update_fw = False
        if (self.filter_world_upd_ts is not None) and (
            self.world_time - self.filter_world_upd_ts >= self.filter_update_lag
        ):
            ch = self._await_changeset()
            ch.update_filter_world_py(self.filter_world)
            update_fw = True

        fw_ts = self.filter_world.timestamp_py()
        if (fw_ts is None) or (
            (self.world_time - fw_ts >= self.filter_delta) and (self.next_changeset is None)
        ):
            self._render_agent_vision()

        # Collisions
        collisions = self.world.collisions_py(self.agent.get_pos(), self.agent_bounds)
        collided = len(collisions) > 0
        terminated = bool(collided)
        truncated = False

        obs = self._build_observation()
        reward = self.reward_fn.compute(
            collided=collided, overridden=overridden, failed=failed, plan_len=int(locals().get("chosen_plan_len", 0))
        )
        info: Dict[str, Any] = {"overridden": overridden, "collided": collided, "failed": failed}

        # Update render client in the same way as grid_world.py
        self.render(update_fw)
        return obs, reward, terminated, truncated, info

    # --------------- Utilities -----------------------------------------------
    def _build_observation(self) -> Dict[str, Any]:
        hx, hy, hz = self.dense_half_dims
        expected = (2 * hx + 1, 2 * hy + 1, 2 * hz + 1)
        vc = self.agent.get_coord_py()
        grid = self.filter_world.dense_snapshot_py(vc, (hx, hy, hz)).data_py()
        grid = np.asarray(grid, dtype=np.int32).reshape(expected)
        return {"grid": grid, "last_action": self._last_action.copy()}

    def render(self, update_fw: bool):
        if self.client:
            # Send current agent state
            self.client.send_agents_py({0: self.agent})
            # Send POV updates when the filter world was updated this tick
            if update_fw:
                self.filter_world.send_pov_py(
                    self.client, 0, 0, self.camera_proj, self.camera_orientation
                )
        # headless by default
        return None
