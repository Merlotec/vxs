from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import gymnasium as gym


class MoveMaskingWrapper(gym.Wrapper):
    """
    Enforces a mask on the first dimension (move) of a MultiDiscrete([7, 5, 8]) action.

    - Reads `info["action_mask_move"]` from the underlying env's reset()/step() output.
    - Before passing an action to env.step(), coerces the move index into an allowed one.

    This requires the base env to populate `info["action_mask_move"]` with a
    length-7 binary mask indicating valid moves for the NEXT step. The provided
    GridWorldEnv does this in both reset() and step().
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._current_move_mask: Optional[np.ndarray] = None

        # Validate action space shape
        if not isinstance(env.action_space, gym.spaces.MultiDiscrete):
            raise TypeError("MoveMaskingWrapper expects a MultiDiscrete action space")
        if env.action_space.nvec[0] != 7:
            raise ValueError("Expected first dimension (move) size 7")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        mask = info.get("action_mask_move", None)
        if mask is None:
            # Default to all-allowed on reset if not provided
            mask = np.ones(7, dtype=np.int8)
        self._current_move_mask = np.asarray(mask, dtype=np.int8)
        return obs, info

    def step(self, action: np.ndarray | Sequence[int]):
        # Ensure action is mutable numpy array
        a = np.array(action, dtype=np.int64, copy=True)

        # Apply mask to the first (move) dimension if available
        if self._current_move_mask is not None:
            mask = self._current_move_mask
            if mask.shape != (7,):
                raise ValueError("action_mask_move must be shape (7,)")
            if mask.sum() <= 0:
                # Fallback: allow no-op if everything else is forbidden
                a[0] = 6
            elif mask[int(a[0])] == 0:
                # Coerce to a valid move; prefer no-op (6) if allowed
                if mask[6] == 1:
                    a[0] = 6
                else:
                    # pick first allowed index
                    a[0] = int(np.flatnonzero(mask)[0])

        obs, reward, terminated, truncated, info = self.env.step(a)

        # Store mask for the next step
        next_mask = info.get("action_mask_move", None)
        if next_mask is not None:
            self._current_move_mask = np.asarray(next_mask, dtype=np.int8)
        else:
            self._current_move_mask = None

        return obs, reward, terminated, truncated, info

