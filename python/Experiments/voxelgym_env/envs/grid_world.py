from enum import Enum
import gymnasium as gym
import numpy as np
import voxelsim as vxs;
import random 

from dataclasses import dataclass, field

class GridWorldEnv(gym.Env):
        def __init__(self,
                agent_dynamics: vxs.QuadDynamics = vxs.QuadDynamics.default_py(),
                chaser: vxs.FixedLookaheadChaser = vxs.FixedLookaheadChaser.default_py(),
                camera_proj: vxs.CameraProjection = vxs.CameraProjection.default_py(),
                camera_orientation: vxs.CameraOrientation = vxs.CameraOrientation.vertical_tilt_py(-0.5),
                filter_update_lag: float = 0.2,
                start_pos: tuple[int, int, int] = (0, 0, 0),
                renderer_view_size: tuple[int, int] = (150, 100),
                noise: vxs.NoiseParams = vxs.NoiseParams.default_with_seed_py(0),
                delta_time: float = 0.01,
                client: vxs.RendererClient | None = None,
    ):
        # Self initialise the data structure
        super(DroneEnv, self).__init__()
        self.action_space = gym.spaces.MultiDiscrete([6, 10, 8]) # Movement, urgency, rotation
        self.agent_dynamics = agent_dynamics
        self.chaser = chaser
        self.camera_proj = camera_proj
        self.camera_orientation = camera_orientation
        self.filter_update_lag = filter_update_lag
        self.start_pos = start_pos
        self.renderer_view_size = renderer_view_size
        self.noise = noise
        self.client = client
        # Define key variables
        # The delta time of the world.
        self.delta_time = delta_time
        # The current world time of the simulation.
        self.world_time = 0
        # The previous filter world update time.
        self.next_changeset = None
        self.filter_world_upd_ts = None
        self.current_step = 0

        # Self initialise world and agent
        self._init_world()

        return

    def _init_world(self):
        self.gen_world()
        # Create agent.
        self.agent = vxs.Agent(0)
        self.agent_vision = vxs.AgentVisionRenderer(self.world, self.view_size, self.noise)
        self.filter_world = vxs.FilterWorld()
    
    def gen_world(self, seed: int):
        # Randomly generate the world terrain
        terrain_gen = vxs.TerrainGenerator()
        terrain_gen.generate_terrain_py(vxs.TerrainConfig(seed))

        # Generate the actual world.
        world = terrain_gen.generate_world_py()
        return
      
    def update_callback(self, changeset):
        # Set the update timestamp to none to tell anything awaiting this result that the world has updated.
        self.next_changeset = changeset
        return
    
    def await_changeset(self) -> vxs.WorldChangeset:
        while self.changeset == None:
            time.sleep(0.00001)
        cch = self.changeset
        self.changeset = None
        self.filter_world_upd_ts = None
        return self.changeset
    
    def update_filter_world(self, changeset: vxs.WorldChangeset):
        changeset.update_filter_world_py(self.filter_world)
    
    def observations(self):
        # Take observations from environment
        self.filter_world_upd_ts = self.world_time
        self.agent_vision.render_changeset_py(self.agent.camera_view_py, self.camera_proj, self.filter_world, self.filter_world_upd_ts, lambda ch: self.update_callback(self, ch))
        return 
        
    def reward(self):
        # Calculate the reward based on the agent's state and actions
        return 
    
    def value_function(self):
        # Calculate the value function based on the agent's state and reward history
        return

    def decode_action(action) -> list[vxs.MoveCommand]:
        
    
    def step(self, action):
        if self.filter_world_upd_ts and self.filter_world_update_ts - self.world_time >= self.filter_update_lag:
            changeset = self.await_changeset()
            self.update_filter_world(changeset)

        # The point in space that the drone should be chasing.
        self.chase_target = self.chaser.step_chase_py(self.agent, self.delta_time)
        self.dynamics.update_agent_dynamics_py(agent, env, chase_target, self.delta_time)

        self.world_time += self.delta_time                 
        return observation, reward, terminated, False, info
    
   # Reset between episodes
    def reset(self, seed=random.randint(1, 100000), options=None):
        self.gen_world(seed)

        self.agent_filter_world = vxs.FilterWorld()
        self.agent = vxs.Agent(0)
        self.agent.set_pos(self.start_pos)
        return 


    # TODO: Add rendering window for monitoring
