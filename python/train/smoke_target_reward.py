from __future__ import annotations

import math
import time

import numpy as np

from envs.grid_world_astar import GridWorldAStarEnv
from rewards.target_locate import RewardTargetLocate


def main():
    # Headless env to quickly verify reward triggering
    env = GridWorldAStarEnv(
        reward=RewardTargetLocate(reach_reward=10.0, respawn_on_reach=False, max_distance=20, z_level=-40),
        render_client=None,
        start_pos=(100, 100, -40),
        allow_override=True,
    )
    obs, info = env.reset()

    steps = 0
    total_reward = 0.0
    reached = False
    max_steps = 200

    while steps < max_steps and not reached:
        # Get agent and target voxels
        ac = env.agent.get_coord_py()
        # Access the reward internals for testing purposes
        target = getattr(env.reward_fn, "_target_voxel", None)
        if target is None:
            # No target yet; take a no-op-ish action
            act = np.array([0.0, 0.0, 0.0, 0.9, 0.0, 1.0], dtype=np.float32)
        else:
            dx = target[0] - ac[0]
            dy = target[1] - ac[1]
            dz = target[2] - ac[2]
            # Clamp to env action bounds
            lim = env.max_offset
            dx = float(max(-lim, min(lim, dx)))
            dy = float(max(-lim, min(lim, dy)))
            dz = float(max(-lim, min(lim, dz)))
            act = np.array([dx, dy, dz, 0.9, 0.0, 1.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(act)
        total_reward += float(reward)
        steps += 1

        if reward >= 9.5:  # reached (approx reach_reward)
            reached = True
            print(f"Reached target at step {steps} with reward {reward}")
            break

        if terminated or truncated:
            print("Episode ended early.")
            break

    print(f"Steps: {steps}, total_reward: {total_reward}, reached: {reached}")


if __name__ == "__main__":
    main()
