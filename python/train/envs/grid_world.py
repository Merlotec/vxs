from __future__ import annotations

import math
import time
import random
from typing import Optional, Tuple, List, Any, Dict

from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import voxelsim as vxs


# We can implement this abstract class to build an actual reward function on top of the baseline environment.
class RewardFunction(ABC):
    @abstractmethod
    def compute(
        self,
        env: "GridWorldEnv",
        observation: Dict[str, Any],
        action: np.ndarray | List[int],
        info: Dict[str, Any],
    ) -> float:
        ...


class BaselineReward(RewardFunction):
    def __init__(self, collision_penalty: float = -1000.0, living_cost: float = -0.01, move_bonus: float = 0.05):
        self.collision_penalty = collision_penalty
        self.living_cost = living_cost
        self.move_bonus = move_bonus

    def compute(self, env, observation, action, info) -> float:
        r = 0.0
        if info.get("collided", False):
            r += self.collision_penalty
        r += self.living_cost
        try:
            cmd = int(action[0])
            if cmd != 6:  # not flush/no-op
                r += self.move_bonus
        except Exception:
            pass
        return r


class GridWorldEnv(gym.Env):
    """VoxelSim-based grid world with pluggable rewards and rich observation."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        agent_dynamics: vxs.QuadDynamics = vxs.QuadDynamics.default_py(),
        chaser: vxs.FixedLookaheadChaser = vxs.FixedLookaheadChaser.default_py(),
        camera_proj: vxs.CameraProjection = vxs.CameraProjection.default_py(),
        camera_orientation: vxs.CameraOrientation = vxs.CameraOrientation.vertical_tilt_py(-0.5),
        agent_bounds: Tuple[float, float, float] = (0.5, 0.5, 0.4),
        filter_update_lag: float = 0.2,
        filter_delta: float = 0.25,
        dense_half_dims: Tuple[int, int, int] = (50, 50, 50),
        start_pos: Tuple[int, int, int] = (0, 0, 0),
        renderer_view_size: Tuple[int, int] = (150, 100),
        noise: vxs.NoiseParams = vxs.NoiseParams.default_with_seed_py([0, 0, 0]),
        delta_time: float = 0.1,
        client: Optional[vxs.RendererClient] = None,
        reward_fn: Optional[RewardFunction] = None,
    ):
        super().__init__()

        # Action: [move(0..6), urgency(1..5), rotation(0..7)]
        self.action_space = gym.spaces.MultiDiscrete([7, 5, 8])

        # Save config
        self.agent_dynamics = agent_dynamics
        self.chaser = chaser
        self.camera_proj = camera_proj
        self.camera_orientation = camera_orientation
        self.agent_bounds = agent_bounds
        self.filter_update_lag = float(filter_update_lag)
        self.filter_delta = float(filter_delta)
        self.dense_half_dims = tuple(int(x) for x in dense_half_dims)
        self.start_pos = tuple(start_pos)
        self.renderer_view_size = tuple(renderer_view_size)
        self.noise = noise
        self.client = client
        self.delta_time = float(delta_time)

        # Time/state
        self.world_time: float = 0.0
        self.next_changeset: Optional[vxs.WorldChangeset] = None
        self.filter_world_upd_ts: Optional[float] = None
        self.current_step: int = 0

        # Buffers and snapshots
        self.action_buffer: Optional[List[vxs.MoveCommand]] = None
        self.action_buffer_codes: List[Tuple[int, int, int]] = []  # (cmd, urgency_idx, yaw_idx)
        self.executing_actions_codes: List[Tuple[int, int, int]] = []
        self.snapshot_grid: Optional[np.ndarray] = None  # int32 grid

        # Reward plugin
        self.reward_fn: RewardFunction = reward_fn or BaselineReward()

        # World & agent
        self.world: Optional[vxs.World] = None
        self.filter_world: Optional[vxs.FilterWorld] = None
        self.agent: Optional[vxs.Agent] = None
        self.agent_vision: Optional[vxs.AgentVisionRenderer] = None
        self.chase_target: Optional[Any] = None
        self.world_env = vxs.EnvState.default_py()

        self._init_world()

        # After agent exists, we can define observation_space
        self.max_cmd_count = int(getattr(self.agent, "max_command_count", lambda: 8)())
        self._build_observation_space()

    def _init_world(self):
        self.gen_world(seed=random.randint(1, 10**6))
        if self.client:
            self.client.send_world_py(self.world)
        self.agent = vxs.Agent(0)
        self.agent.set_pos(self.start_pos)
        self.filter_world = vxs.FilterWorld()
        self.agent_vision = vxs.AgentVisionRenderer(self.world, self.renderer_view_size, self.noise)

    def gen_world(self, seed: int):
        terrain_gen = vxs.TerrainGenerator()
        terrain_gen.generate_terrain_py(vxs.TerrainConfig(seed))
        self.world = terrain_gen.generate_world_py()

    def _build_observation_space(self):
        hx, hy, hz = self.dense_half_dims
        grid_shape = (2 * hx + 1, 2 * hy + 1, 2 * hz + 1)

        # int32 grid with enum values; allow broad range
        grid_space = gym.spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            shape=grid_shape,
            dtype=np.int32,
        )

        # agent voxel coords (unknown world bounds -> wide)
        agent_voxel_space = gym.spaces.Box(
            low=np.array([-10**6, -10**6, -10**6], dtype=np.int32),
            high=np.array([10**6, 10**6, 10**6], dtype=np.int32),
            dtype=np.int32,
        )

        # sequences as (cmd, urgency_idx, yaw_idx), padded with -1
        seq_shape = (self.max_cmd_count, 3)
        low = np.tile(np.array([-1, -1, -1], dtype=np.int32), (self.max_cmd_count, 1))
        high = np.tile(np.array([6, 5, 7], dtype=np.int32), (self.max_cmd_count, 1))

        seq_space = gym.spaces.Box(low=low, high=high, shape=seq_shape, dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "grid": grid_space,
                "executing_actions": seq_space,
                "planned_actions": seq_space,
            }
        )

    def update_callback(self, changeset: vxs.WorldChangeset):
        self.next_changeset = changeset

    def await_changeset(self) -> vxs.WorldChangeset:
        while self.next_changeset is None:
            time.sleep(0.00001)
        ch = self.next_changeset
        self.next_changeset = None
        self.filter_world_upd_ts = None
        return ch

    def update_filter_world(self, changeset: vxs.WorldChangeset):
        changeset.update_filter_world_py(self.filter_world)

    def _render_agent_vision(self):
        self.filter_world_upd_ts = self.world_time
        self.agent_vision.render_changeset_py(
            self.agent.camera_view_py(self.camera_orientation),
            self.camera_proj,
            self.filter_world,
            self.filter_world_upd_ts,
            lambda ch: self.update_callback(ch),
        )

    def _pad_seq_codes(self, codes: List[Tuple[int, int, int]]) -> np.ndarray:
        """Return (max_cmd_count,3) int32 array padded with -1."""
        arr = np.full((self.max_cmd_count, 3), -1, dtype=np.int32)
        n = min(len(codes), self.max_cmd_count)
        if n > 0:
            arr[:n, :] = np.asarray(codes[:n], dtype=np.int32)
        return arr

    def _encode_move_triplet(self, cmd: int, urgency_idx: int, yaw_idx: int) -> Tuple[int, int, int]:
        return int(cmd), int(urgency_idx), int(yaw_idx)

    def _build_observation(self):
        hx, hy, hz = self.dense_half_dims
        expected_shape = (2*hx + 1, 2*hy + 1, 2*hz + 1)

        vc = self.agent.get_voxel_coord()
        grid = self.filter_world.dense_snapshot_py(vc, (hx, hy, hz)).data_py()

        grid = np.asarray(grid, dtype=np.int32)
        if grid.ndim == 1 and grid.size == np.prod(expected_shape):
            grid = grid.reshape(expected_shape)          # <-- key change
        elif grid.shape != expected_shape:
            raise ValueError(f"grid shape {grid.shape} != {expected_shape}")

        self.snapshot_grid = grid

        return {
            "grid": grid,
            "executing_actions": self._pad_seq_codes(self.executing_actions_codes),
            "planned_actions": self._pad_seq_codes(self.action_buffer_codes),
        }

    def _last_or_dummy_obs(self) -> Dict[str, Any]:
        if self.snapshot_grid is not None:
            grid = self.snapshot_grid
        else:
            hx, hy, hz = self.dense_half_dims
            grid = np.zeros((2 * hx + 1, 2 * hy + 1, 2 * hz + 1), dtype=np.int32)
        return {
            "grid": grid,
            "executing_actions": self._pad_seq_codes(self.executing_actions_codes),
            "planned_actions": self._pad_seq_codes(self.action_buffer_codes),
        }


    def _take_action_buffer(self) -> Optional[List[vxs.MoveCommand]]:
        buf = self.action_buffer
        self.action_buffer = None
        self.action_buffer_codes = []
        return buf

    def _decode_action(self, action: np.ndarray | List[int]) -> Optional[List[vxs.MoveCommand]]:
        cmd = int(action[0])             # 0..6 (6 == flush/no-op)
        urgency_idx = int(action[1])     # 0..5
        yaw_idx = int(action[2])         # 0..7

        urgency = 0.2 * (urgency_idx + 1)
        yaw = math.pi * 0.25 * yaw_idx

        if cmd == 6:
            # flush/no-op -> return whatever is in the action buffer
            return self._take_action_buffer()

        mv_cmd = vxs.MoveCommand(vxs.MoveDir.from_code_py(cmd), urgency, yaw)
        if self.action_buffer is None:
            self.action_buffer = []
        self.action_buffer.append(mv_cmd)
        self.action_buffer_codes.append(self._encode_move_triplet(cmd, urgency_idx, yaw_idx))

        max_count = getattr(self.agent, "max_command_count", lambda: 8)()
        if len(self.action_buffer) >= max_count:
            return self._take_action_buffer()
        return None


    def step(self, action: np.ndarray | List[int]):
        info: Dict[str, Any] = {}
        terminated = False
        truncated = False

        mv_commands = self._decode_action(action)
        if mv_commands is None:
            observation = self._last_or_dummy_obs()
            reward = self.reward_fn.compute(self, observation, action, {"waiting_for_more_commands": True})
            info["next"] = "next_command"
            return observation, reward, terminated, truncated, info

        # We have a full sequence to execute
        self.executing_actions_codes = list(self.action_buffer_codes) if self.action_buffer_codes else []
        if len(mv_commands) > 0:
            try:
                self.agent.perform_sequence_py(mv_commands)
            except:
                # Invalid sequence, we must punish.
                observation = self._last_or_dummy_obs()
                info["command_error"] = "duplicate_centroid"
                reward = -100
                return observation, reward, terminated, truncated, info

        # Chase target & dynamics (adjust signatures to your SDK if needed)
        self.chase_target = self.chaser.step_chase_py(self.agent, self.delta_time)
        self.agent_dynamics.update_agent_dynamics_py(self.agent, self.world_env, self.chase_target, self.delta_time)

        # Advance time
        self.world_time += self.delta_time
        self.current_step += 1

        update_fw = False
        # Update filter world if needed
        if self.filter_world_upd_ts != None and self.world_time - self.filter_world_upd_ts >= self.filter_update_lag:
            changeset = self.await_changeset()
            self.update_filter_world(changeset)
            update_fw = True
            # Start a new planning buffer for next step
            self.action_buffer = []
            self.action_buffer_codes = []

        
        # Check whether to start the render pass:
        fw_ts = self.filter_world.timestamp_py()
        if (fw_ts == None) or (self.world_time - fw_ts >= self.filter_delta) and (self.next_changeset == None):
            # We cannot have a frame currently in process/awaiting pop.
            self._render_agent_vision()
        

        # Collisions
        collisions = self.world.collisions_py(self.agent.get_pos(), self.agent_bounds)
        collided = len(collisions) > 0
        info["collided"] = collided
        if collided:
            terminated = True

        # Observation (grid is int32)
        observation = self._build_observation()

        # Reward
        reward = self.reward_fn.compute(self, observation, action, info)

        self.render(update_fw)

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        print("RESET")
        if seed is None:
            seed = random.randint(1, 10**6)

        self.gen_world(seed)
        if self.client:
            self.client.send_world_py(self.world)

        self.filter_world = vxs.FilterWorld()
        self.agent = vxs.Agent(0)
        self.agent.set_pos(self.start_pos)
        self.world_time = 0.0
        self.current_step = 0
        self.next_changeset = None
        self.filter_world_upd_ts = None
        self.action_buffer = []
        self.action_buffer_codes = []
        self.executing_actions_codes = []

        # Sync max command count in case it depends on world/agent
        self.max_cmd_count = int(getattr(self.agent, "max_command_count", lambda: 8)())
        self._build_observation_space()

        # Prime the first observation
        self._render_agent_vision()
        changeset = self.await_changeset()
        self.update_filter_world(changeset)
        observation = self._build_observation()
        
        info: Dict[str, Any] = {}
        return observation, info

    def render(self, update_fw):
        if self.client:
            self.client.send_agents_py({0: self.agent})
            if update_fw:
                self.filter_world.send_pov_py(self.client, 0, 0, self.camera_proj, self.camera_orientation)
        pass

    def close(self):
        pass
