from enum import Enum
import gymnasium as gym
import numpy as np




class GridWorldEnv(gym.Env):

    def __init__(self, 
                 #TODO: We can add options to changte kind of terrain config we are iterating over

                #TODO: implement a planning horizon e.g.  planning_horizon: int = 3, 
                 delta_time: float = 0.01,
                 max_steps: int = 1000,
                 agent_id: int = 0,
                 # Feature flags
                #  normalize_obs: bool = True,
                #  frame_stack: int = 0,
                #  reward_shaping: bool = True):
    ):
        # Self initialise the data structure
        super(DroneEnv, self).__init__()
        print("hi")
        # Define key variables
        self.delta_time = delta_time
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_id = agent_id

        # Self initialise world and agent
        self._init_world()


        # Define observation space

        # Define action space

        # Define mapping of action names to directions

        # Initialise rendering/moonitoring
        return
    
    def gen_world(self):
        # Randomly generate the world terrain``
        return
      

    def observations(self):
        # Take observations from environment
        return 
        
    def reward(self):
        # Calculate the reward based on the agent's state and actions
        return 
    
    def value_function(self):
        # Calculate the value function based on the agent's state and reward history
        return
    
    def step(self, action):
    # Map the action to a direction

    # Setup safety checks
    
    # An episode is done iff the agent has reached the target
    
        return observation, reward, terminated, False, info
    
   # Reset between episodes
    def reset(self, seed=None, options=None):
        # Reset the terrain seed 

        # Choose the agent's location uniformly at random


        return 


    # TODO: Add rendering window for monitoring