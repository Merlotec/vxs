from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class VoxGridExtractor(BaseFeaturesExtractor):
    """
    Features extractor for Dict observations with keys:
      - "grid": (C,D,H,W) float in [0,1] if semantic, or (D,H,W) int -> converted to 1-channel float
      - "last_action": (6,)
      - optional "goal_rel": (3,)

    Encodes the 3D grid with small Conv3D stack + GAP, then concatenates low-dim vectors.
    """

    def __init__(self, observation_space: spaces.Dict, grid_key: str = "grid"):
        # Compute total features dim after building the model
        super().__init__(observation_space, features_dim=1)

        assert isinstance(observation_space, spaces.Dict)
        self.grid_key = grid_key

        grid_space = observation_space[self.grid_key]
        last_action_space = observation_space["last_action"]
        # Accept either legacy "goal_rel" or new "target_rel"
        if "goal_rel" in observation_space:
            self.goal_key = "goal_rel"
        elif "target_rel" in observation_space:
            self.goal_key = "target_rel"
        else:
            self.goal_key = None

        if isinstance(grid_space, spaces.Box) and len(grid_space.shape) == 4:
            # (C,D,H,W)
            in_channels = grid_space.shape[0]
            depth, height, width = grid_space.shape[1:]
            self.semantic = True
        elif isinstance(grid_space, spaces.Box) and len(grid_space.shape) == 3:
            # (D,H,W)
            in_channels = 1
            depth, height, width = grid_space.shape
            self.semantic = False
        else:
            raise ValueError("Unsupported grid space shape")

        # Small Conv3D encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # -> (N, 32, 1, 1, 1)
        )

        # Compute conv output dim
        dummy = torch.zeros(1, in_channels, depth, height, width)
        with torch.no_grad():
            enc = self.encoder(dummy)
        conv_dim = enc.view(1, -1).shape[1]

        low_dim = last_action_space.shape[0]
        if self.goal_key is not None:
            low_dim += observation_space[self.goal_key].shape[0]

        self._features_dim = conv_dim + low_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations[self.grid_key]
        # Convert to channels-first 5D: (N,C,D,H,W)
        if grid.dim() == 4:
            # (N, D, H, W) -> add channel
            grid = grid.unsqueeze(1).float()
            # Normalize ints to {0,1}
            grid = (grid != 0).float()
        else:
            # Assume already (N, C, D, H, W)
            grid = grid.float()

        enc = self.encoder(grid)
        enc = torch.flatten(enc, 1)

        low = observations["last_action"].float()
        if self.goal_key is not None:
            low = torch.cat([low, observations[self.goal_key].float()], dim=1)

        return torch.cat([enc, low], dim=1)
