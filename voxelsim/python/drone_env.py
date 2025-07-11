# drone_env.py - All-in-one environment file

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import voxelsim
from typing import Dict, Tuple, Optional, Any
from collections import deque
import time


class DroneEnv(gym.Env):
    """Custom Gym environment for drone navigation in voxel world.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 world_size: int = 100, # We can add proper terain config parameters here when varying the terrain
                 render_client: Optional[voxelsim.RendererClient] = None,
                 planning_horizon: int = 3, 
                 delta_time: float = 0.01,
                 max_steps: int = 1000,
                 agent_id: int = 0,
                 # Feature flags
                 normalize_obs: bool = True,
                 frame_stack: int = 0,  # 0 = disabled
                 reward_shaping: bool = True,
                 continuous_actions: bool = True):
        
        super(DroneEnv, self).__init__()
        
        # Environment parameters
        self.world_size = world_size
        self.delta_time = delta_time
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_id = agent_id
        
        # Features
        self.normalize_obs = normalize_obs
        self.frame_stack = frame_stack
        self.reward_shaping = reward_shaping
        self.continuous_actions = continuous_actions
        
        # Rendering
        self.render_client = render_client
        self._last_render_time = 0
        self._render_fps = 30
        
        # Initialize voxel world and agent
        self._init_world()
        
        # Define action space
        # if continuous_actions:
        #     # Continuous: [forward/back, left/right, up/down] in [-1, 1]
        #     self.action_space = spaces.Box(
        #         low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        #     )
        # else:
        #     # Discrete: 7 actions
        #     self.action_space = spaces.Discrete(7)

                # --- planning horizon: how many primitive moves the agent picks at once
        self.H = int(planning_horizon)          # e.g. pass planning_horizon=3 in __init__
        if self.H < 1:
            raise ValueError("planning_horizon must be ≥ 1")

        # --- each slot can be any of the 7 primitive symbols (0‒6)
        #     so the Gym-visible action is a tuple of length H
        self.action_space = spaces.MultiDiscrete([7] * self.H)

        
        # Define observation space
        # TODO: Add camera/POV data here
        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Dict({
        "position":     spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "velocity":     spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "local_voxels": spaces.Box(0, 255, (5,5,5), np.uint8),
        "camera_occ":   spaces.Box(0, 2, (H, W, k_frames), np.uint8),
        "map_embed":    spaces.Box(0, 2, (self.embed_dim,), np.uint8),
    })
            
        
        # Normalization statistics
        if self.normalize_obs:
            self.obs_mean = {'position': np.zeros(3), 'velocity': np.zeros(3)}
            self.obs_var = {'position': np.ones(3), 'velocity': np.ones(3)}
            self.obs_count = 1e-4
        
        # Reward shaping tracking
        if self.reward_shaping:
            self.previous_distance = None
            self.previous_velocity = None
        
        # Target position
        self.target_position = None
        self._set_random_target()
    
    def _get_obs_shape(self):
        """Get observation shapes based on configuration."""
        shapes = {}
        
        
        if self.frame_stack > 0:
            # Here we are creating shapes for th key observations depending on how big a frame stack we are using.
            shapes['position']     = (self.frame_stack, 3)      # time, feat
            shapes['velocity']     = (self.frame_stack, 3)
            shapes['local_voxels'] = (self.frame_stack, 5, 5, 5)
        else:
            shapes['position']     = (3,)
            shapes['velocity']     = (3,)
            shapes['local_voxels'] = (5, 5, 5)
        
    def _init_world(self):
        """Initialize the voxel world and agent."""
        self.world = voxelsim.VoxelGrid()
        self.world.generate_default_terrain(self.world_size)
        
        self.dynamics = voxelsim.AgentDynamics.default_drone()
        self.agent = voxelsim.Agent(self.agent_id)
        
        # Set initial position
        # TODO: Make this a reandomised position in which we start the agent each time
        self.agent.set_position(50.0, 20.0, 50.0)
        
        self.env = voxelsim.GlobalEnv(self.world, {self.agent_id: self.agent})
        
        # Send world to renderer if connected
        if self.render_client:
            self.env.send_world(self.render_client)
    
    def _set_random_target(self,
                       margin: int = 10,
                       hover_alt: Tuple[float, float] = (10.0, 30.0),
                       chance_air: float = 0.5,
                       max_attempts: int = 50):
        """
        Choose a target that is inside the map, not buried, and not colliding.
        
        Args
        ----
        margin       :  minimum clearance from world boundary on X and Z
        hover_alt    :  (min_y_above_ground, max_y_above_ground) if we pick an airborne goal
        chance_air   :  probability of selecting an airborne goal vs. a ground goal
        max_attempts :  safety loop to avoid infinite retries
        """
        
        for _ in range(max_attempts):
            # --- sample horizontal position inside the safe margin ------------
            x = np.random.uniform(margin, self.world_size - margin)
            z = np.random.uniform(margin, self.world_size - margin)

            # --- query terrain height at (x, z) -------------------------------
            ground_y = self.world.ground_height(int(x), int(z))  # you may need to implement

            # pick air or ground
            if self.rng.random() < chance_air:
                # airborne target
                y = ground_y + np.random.uniform(*hover_alt)
            else:
                # on-ground target
                y = ground_y + 0.5        # half-voxel above to avoid sinking

            # --- collision check ---------------------------------------------
            if not self.world.is_occupied((int(x), int(y), int(z))):
                self.target_position = np.array([x, y, z], dtype=np.float32)

                # mark the voxel so the renderer can highlight it
                self.world.set((int(x), int(y), int(z)), voxelsim.Cell.TARGET)
                return

            raise RuntimeError("Could not find a valid target after many attempts")

    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from the environment."""
        pos_tuple = self.agent.get_position()
        position = np.array(pos_tuple, dtype=np.float32)
        
        # TODO: Get actual velocity from agent
        velocity = np.zeros(3, dtype=np.float32)
        
        # Get local voxel information
        local_voxels = self._get_local_voxels(position)
        
        # TODO: Add camera and embedding data
        # camera_rgb = self._get_camera_view()
        # embedding = self._get_drone_embedding()
        
        obs = {
            'position': position,
            'velocity': velocity,
            'local_voxels': local_voxels,
        }
        
        # Apply normalization if enabled
        if self.normalize_obs:
            obs = self._normalize_observation(obs)
        
        # Apply frame stacking if enabled
        if self.frame_stack > 0:
            obs = self._stack_observation(obs)
        
        return obs
    
    def _normalize_observation(self, obs):
        """Normalize observations using running statistics."""
        for key in ['position', 'velocity']:
            # Update statistics
            delta = obs[key] - self.obs_mean[key]
            self.obs_count += 1
            self.obs_mean[key] += delta / self.obs_count
            self.obs_var[key] = (self.obs_var[key] * (self.obs_count - 1) + 
                                 delta ** 2) / self.obs_count
            
            # Normalize
            obs[key] = (obs[key] - self.obs_mean[key]) / np.sqrt(self.obs_var[key] + 1e-8)
        
        return obs
    
    def _stack_observation(self, obs):
        """Stack observations for temporal information."""
        if len(self.frames) == 0:
            # First observation, fill the buffer
            for _ in range(self.frame_stack):
                self.frames.append(obs)
        else:
            self.frames.append(obs)
        
        # Stack frames
        stacked = {
            'position': np.concatenate([f['position'] for f in self.frames]),
            'velocity': np.concatenate([f['velocity'] for f in self.frames]),
            'local_voxels': np.stack([f['local_voxels'] for f in self.frames])
        }
        
        return stacked
    
    def _get_local_voxels(self, position: np.ndarray, size: int = 5) -> np.ndarray:
        """Get voxel data in a cube around the agent."""
        voxels = np.zeros((size, size, size), dtype=np.uint8)
        # TODO: Implement actual voxel queries
        return voxels
    
    # ---------------------------------------------------------------------
#  NEW helper (replaces _action_to_command)
# ---------------------------------------------------------------------
    def _action_to_sequence(self, action) -> list[voxelsim.MoveCommand]:
        """
        Convert the Gym action into a *list* of MoveCommand objects,
        up to the planning horizon H set in __init__.

        ── Action space ───────────────────────────────────────────────────
        • If continuous_actions = True      ⇒  action is ℝ³  (Box[-1,1]^3)
        • If continuous_actions = False     ⇒  action is an
            ndarray[int] of length H        (MultiDiscrete([7]*H))
            each element   0 1 2 3 4 5 6
            means          · F B L R U D
            (‘0’ = no-op, so the plan can end early)
        """
        cmds: list[voxelsim.MoveCommand] = []

        if self.continuous_actions:
            # -------- continuous: map vector components to axis moves -----
            v = np.clip(action, -1.0, 1.0).astype(float)
            for axis, val in enumerate(v):
                if abs(val) < 0.05:          # tiny thrust → ignore
                    continue
                urgency = abs(val)
                if axis == 0:                # X  (forward/back)
                    cmd = (voxelsim.MoveCommand.forward if val > 0
                        else voxelsim.MoveCommand.back)(urgency)
                elif axis == 1:              # Y  (left/right)
                    cmd = (voxelsim.MoveCommand.right  if val > 0
                        else voxelsim.MoveCommand.left)(urgency)
                else:                        # Z  (up/down)
                    cmd = (voxelsim.MoveCommand.up     if val > 0
                        else voxelsim.MoveCommand.down)(urgency)
                cmds.append(cmd)

            # respect MAX_ACTIONS = 6 in voxelsim
            return cmds[:voxelsim.agent.MAX_ACTIONS]

        # -------- discrete MultiDiscrete: interpret each slot directly ---
        # action arrives as ndarray, list or tuple of ints
        digits = np.asarray(action, dtype=int).flatten()[:self.H]
        fixed_u = 0.8
        mapping = {
            1: voxelsim.MoveCommand.forward,
            2: voxelsim.MoveCommand.back,
            3: voxelsim.MoveCommand.left,
            4: voxelsim.MoveCommand.right,
            5: voxelsim.MoveCommand.up,
            6: voxelsim.MoveCommand.down,
        }
        for d in digits:
            if d == 0:
                break               # early-stop on “no-op”
            cmds.append(mapping[d](fixed_u))
        return cmds

    
    def _calculate_reward(self, observation: Dict[str, np.ndarray]) -> float:
        """Calculate reward based on current state."""
        # Get denormalized position if needed
        if self.normalize_obs:
            position = (observation['position'][:3] * np.sqrt(self.obs_var['position'] + 1e-8) + 
                       self.obs_mean['position'])
            velocity = (observation['velocity'][:3] * np.sqrt(self.obs_var['velocity'] + 1e-8) + 
                       self.obs_mean['velocity'])
        else:
            position = observation['position']
            velocity = observation['velocity']
        
        # Basic distance reward
        distance = np.linalg.norm(position - self.target_position)
        base_reward = -distance / self.world_size - 0.01  # step penalty
        
        # Success bonus
        if distance < 2.0:
            base_reward += 100.0
        
        # Apply reward shaping if enabled
        if self.reward_shaping:
            shaped_reward = self._shape_reward(position, velocity, distance)
            return base_reward + shaped_reward
        
        return base_reward
    
    def _shape_reward(self, position, velocity, distance):
        """Additional reward shaping for better learning."""
        shaped = 0.0
        
        # Progress reward
        if self.previous_distance is not None:
            progress = (self.previous_distance - distance) * 10
            shaped += progress
        
        # Velocity penalty near target
        if distance < 5.0:
            shaped -= np.linalg.norm(velocity) * 0.1
        
        # Smoothness penalty
        if self.previous_velocity is not None:
            acceleration = velocity - self.previous_velocity
            shaped -= np.linalg.norm(acceleration) * 0.01
        
        # Height penalty
        if position[1] < 5:
            shaped -= 0.1 * (5 - position[1])
        elif position[1] > 50:
            shaped -= 0.05 * (position[1] - 50)
        
        # Update tracking
        self.previous_distance = distance
        self.previous_velocity = velocity.copy()
        
        return shaped
    
    def _is_done(self, observation: Dict[str, np.ndarray]) -> bool:
        """Check if episode should terminate."""
        # Get denormalized position
        if self.normalize_obs:
            position = (observation['position'][:3] * np.sqrt(self.obs_var['position'] + 1e-8) + 
                       self.obs_mean['position'])
        else:
            position = observation['position']
        
        # Success
        if np.linalg.norm(position - self.target_position) < 2.0:
            return True
        
        # Failure conditions
        if self.current_step >= self.max_steps:
            return True
        
        if (position < 0).any() or (position > self.world_size).any():
            return True
        
        return False
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self.current_step += 1
        
        # Convert and execute action
        command = self._action_to_command(action)
        if command is not None:
            self.env.perform_sequence_on_agent(self.agent_id, [command])
        
        # Update physics
        def step_callback():
            if self.render_client:
                self.env.send_agents(self.render_client)
        
        self.env.update_with_callback(
            self.dynamics, 
            self.delta_time, 
            step_callback, 
            lambda agent_id: None  # collision callback
        )
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation)
        
        # Check termination
        done = self._is_done(observation)
        
        # Info
        info = {
            'distance_to_target': np.linalg.norm(
                observation['position'][:3] - self.target_position
            ),
            'step': self.current_step,
        }
        
        return observation, reward, done, info
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to initial state."""
        self.current_step = 0
        
        # Reset tracking variables
        if self.reward_shaping:
            self.previous_distance = None
            self.previous_velocity = None
        
        # Clear frame stack
        if self.frame_stack > 0:
            self.frames.clear()
        
        # Reset agent position
        start_x = np.random.uniform(20, self.world_size - 20)
        start_z = np.random.uniform(20, self.world_size - 20)
        self.agent.set_position(start_x, 20.0, start_z)
        self.agent.set_velocity(0.0, 0.0, 0.0)
        
        # Set new target
        self._set_random_target()
        
        # Clear actions
        self.env.perform_sequence_on_agent(self.agent_id, [])
        
        # Send updates
        if self.render_client:
            self.env.send_agents(self.render_client)
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human' and self.render_client:
            current_time = time.time()
            if current_time - self._last_render_time > 1.0 / self._render_fps:
                self.env.send_agents(self.render_client)
                self._last_render_time = current_time
        elif mode == 'rgb_array':
            # TODO: Implement frame capture
            raise NotImplementedError("RGB array rendering not yet implemented")
    
    def close(self):
        """Clean up resources."""
        pass


# Factory function for easy environment creation
def make_drone_env(continuous=True, normalize=True, frame_stack=0, 
                   reward_shaping=True, **kwargs):
    """Create a drone environment with specified features."""
    return DroneEnv(
        continuous_actions=continuous,
        normalize_obs=normalize,
        frame_stack=frame_stack,
        reward_shaping=reward_shaping,
        **kwargs
    )