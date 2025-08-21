from __future__ import annotations

import math
import random
import time
import threading
from typing import Any, Dict, Optional, Tuple
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import voxelsim as vxs


class RewardBase(ABC):
    """Abstract reward with optional world post-processing after reset."""

    @abstractmethod
    def compute(
        self,
        *,
        collided: bool,
        overridden: bool,
        failed: bool,
        plan_len: int = 0,
    ) -> float:
        """Return scalar reward for the last step."""
        raise NotImplementedError

    def post_reset(self, world: vxs.VoxelGrid) -> None:
        """Hook to edit the world immediately after reset and before rendering.

        Default: no-op. Subclasses can modify `world` (e.g., add targets/obstacles).
        """
        return None


class SimpleReward(RewardBase):
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
        action_gain: float = 1.0,  # deprecated: actions are assumed pre-scaled by the model
        attempt_scales: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),  # unused
        allow_override: bool = False,
        start_pos: Tuple[int, int, int] = (0, 0, 0),
        delta_time: float = 0.01,
        ticks_per_step: int = 25,
        filter_update_lag: float = 0.2,
        filter_delta: float = 0.25,
        reward: Optional[SimpleReward] = None,
        render_client: Optional[vxs.AsyncRendererClient] = None,
        dense_half_dims: Tuple[int, int, int] = (16, 16, 16),
        noise: Optional[vxs.NoiseParams] = None,
    ):
        super().__init__()

        self.override_threshold = float(override_threshold)
        self.max_offset = int(max_offset)
        self.astar = vxs.AStarActionPlanner(int(planner_padding))
        self.delta_time = float(delta_time)
        self.filter_update_lag = float(filter_update_lag)
        self.filter_delta = float(filter_delta)
        self.reward_fn: RewardBase = reward or SimpleReward()
        self.start_pos = tuple(start_pos)
        self.dense_half_dims = tuple(int(x) for x in dense_half_dims)
        self.client = render_client
        self.noise = noise or vxs.NoiseParams.default_with_seed_py([0, 0, 0])
        self.action_gain = 1.0  # no-op; model outputs are used as-is
        self.attempt_scales = tuple(float(s) for s in attempt_scales)  # retained for compatibility
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
        self.ticks_per_step = ticks_per_step

        self.start_rt = time.time()
        self.graphics_time = 0
        self.step_time = 0

        self.render_queue = []
        # Should contain world time and world data.
        self.obs_queue = []

        self._last_action = np.zeros(6, dtype=np.float32)

        self._init_world()

    # --------------- World setup / rendering ---------------------------------
    def _init_world(self):
        self._gen_world(seed=random.randint(1, 10**6))
        # Allow reward to edit the world before renderer/client setup
        try:
            self.reward_fn.post_reset(self.world)
        except Exception as e:
            print(f"Reward post_reset failed, continuing: {e}")
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
        render_time = self.world_time
        # Clone the world so that we can update in parallel.
        self.render_queue.append(render_time)
        self.vision.render_changeset_py(
            self.agent.camera_view_py(self.camera_orientation),
            self.camera_proj,
            self.filter_world,
            render_time,
            lambda ch: self._update_callback(ch),
        )

    def _update_callback(self, changeset: vxs.WorldChangeset):
        # We need to wait for the next world to be ready else we will overwrite previous changes.
        # So we must have our timestamp at the end of the render queue.
        try:
            while self.render_queue[0] != changeset.timestamp_py():
                time.sleep(0.00001)

            changeset.update_filter_world_py(self.filter_world)
            # Write world.
            grid = self._build_grid()
            # Now we push to the queue.
            self.obs_queue.append((changeset.timestamp_py, grid))
            self.render_queue.pop(0)
        except ValueError:
            print(f"Failed to update world: {e}")


    # def _await_changeset(self) -> vxs.WorldChangeset:
    #     while self.next_changeset is None:
    #         time.sleep(0.00001)
    #     ch = self.next_changeset
    #     self.next_changeset = None
    #     self.filter_world_upd_ts = None
    #     return ch

    # --------------- RL API ---------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._gen_world(seed or random.randint(1, 10**6))
        # Reward can modify world prior to any renderer/client setup.
        try:
            self.reward_fn.post_reset(self.world)
        except Exception as e:
            print(f"Reward post_reset failed, continuing: {e}")
        self.filter_world = vxs.FilterWorld()
        self.vision = vxs.AgentVisionRenderer(self.world, (160, 100), self.noise)
        self.agent = vxs.Agent(0)
        self.agent.set_hold_py(self.start_pos, 0.0)
        if self.client:
            self.client.send_world_py(self.world)
        self.world_time = 0.0
        self.start_rt = 0
        self.graphics_time = 0
        self.step_time = 0
        self._last_action[:] = 0.0
        self.chaser = vxs.FixedLookaheadChaser.default_py()
        self.dynamics = vxs.px4.Px4Dynamics.default_py()
        self.start_rt = time.time()


        # Prime first dense snapshot
        # self._render_agent_vision()
        # ch = self._await_changeset()
        # ch.update_filter_world_py(self.filter_world)
        obs = {"grid": self._empty_grid(), "last_action": self._last_action.copy()}                
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        step_start = time.time()
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
        # Use action offsets directly; model outputs are assumed to be correctly scaled
        raw = np.array([dx, dy, dz], dtype=np.float32)
        offset = np.round(np.clip(raw, -self.max_offset, self.max_offset)).astype(np.int32)
        dst = tuple((origin + offset).tolist())

        # Record last action as the relative offset requested (updated to actual used below)
        self._last_action[:] = [float(offset[0]), float(offset[1]), float(offset[2]), urgency, yaw, priority]

        overridden = False
        failed = False

        # No additional scaling; RL policy should emit offsets at the correct scale
        # Plan if allowed (override or idle)
        curr = self.agent.get_action_py()
        if (curr is None) or curr.intent_count() < 3:# or (self.allow_override and (priority >= self.override_threshold)):
            if curr and priority >= self.override_threshold:
                overridden = True

            off = np.round(offset).astype(np.int32)
            # Try planning, penalise on exceptions and fall back by reducing offset
            # Try progressively smaller offsets if planning fails
            dst_try = tuple((origin + off).tolist())
            try:
                dirs = self.astar.plan_action_py(self.world, tuple(origin.tolist()), dst_try, urgency, yaw).get_move_sequence()
                if len(dirs) > 0:
                    intent = vxs.ActionIntent(float(urgency), float(yaw), dirs, None)
                    try:
                        self.agent.push_back_intent_py(intent)
                        # record the actual relative offset used
                        self._last_action[:3] = [float(off[0]), float(off[1]), float(off[2])]
                    except ValueError:
                        failed = True
            except Exception:
                failed = True

        # Advance simulation one tick

        
        # chase = self.chaser.step_chase_py(self.agent, self.delta_time)
        # self.dynamics.update_agent_dynamics_py(self.agent, vxs.EnvState.default_py(), chase, self.delta_time)

        self.dynamics.update_agent_fixed_lookahead_py(self.agent, vxs.EnvState.default_py(), self.chaser, self.delta_time, self.ticks_per_step)
        
        self.world_time += self.delta_time * self.ticks_per_step

        # # Filter world update/render scheduling (delay then process)
        # if (self.filter_world_upd_ts is not None) and (
        #     self.world_time - self.filter_world_upd_ts >= self.filter_update_lag
        # ):
        #     ch = self._await_changeset()
        #     ch.update_filter_world_py(self.filter_world)
        #     update_fw = True

        # fw_ts = self.filter_world.timestamp_py()
        # if (fw_ts is None) or (
        #     (self.world_time - fw_ts >= self.filter_delta) and (self.next_changeset is None)
        # ):
        #     self._render_agent_vision()

        # Collisions
        collisions = self.world.collisions_py(self.agent.get_pos(), self.agent_bounds)
        collided = len(collisions) > 0
        terminated = bool(collided)
        truncated = False

        # obs = self._build_observation()
        reward = self.reward_fn.compute(
            collided=collided, overridden=overridden, failed=failed, plan_len=int(locals().get("chosen_plan_len", 0))
        )
        info: Dict[str, Any] = {"overridden": overridden, "collided": collided, "failed": failed}


        # Now await next frame (if required).
        if len(self.obs_queue) + len(self.render_queue) > 1:
            # We should await.
            # We shouldnt actually need to check the time because there should always be a 2 delay on the queue.
            astart = time.time()
            while len(self.obs_queue) == 0:
                time.sleep(0.000001)

            self.graphics_time += (time.time() - astart)
            obs = {"grid": self.obs_queue[0][1], "last_action": self._last_action.copy()}                
            self.obs_queue.pop(0)
        else:
            obs = {"grid": self._empty_grid(), "last_action": self._last_action.copy()}                
            
        
        # Update render client in the same way as grid_world.py
        self.render(True)
        
        step_end = time.time()
        # Begin a new render pass.
        self._render_agent_vision()

        self.step_time += step_end - step_start
        
        time_ratio = self.world_time / (time.time() - self.start_rt)
        graphics_ratio = self.graphics_time / (time.time() - self.start_rt)
        step_ratio = self.step_time / (time.time() - self.start_rt)

        print(f"speedup: {time_ratio}, graphics: {graphics_ratio}, step: {step_ratio}")
        
        return obs, reward, terminated, truncated, info


    def _build_grid(self) -> Any:
        hx, hy, hz = self.dense_half_dims
        expected = (2 * hx + 1, 2 * hy + 1, 2 * hz + 1)
        vc = self.agent.get_coord_py()
        grid = self.filter_world.dense_snapshot_py(vc, (hx, hy, hz)).data_py()
        grid = np.asarray(grid, dtype=np.int32).reshape(expected)
        return grid

    def _empty_grid(self):
        hx, hy, hz = self.dense_half_dims
        expected = (2 * hx + 1, 2 * hy + 1, 2 * hz + 1)
        grid = np.zeros(expected)
        return grid
        
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
                self.filter_world.send_pov_async_py(
                    self.client, 0, 0, self.camera_proj, self.camera_orientation
                )
        # headless by default
        return None
