"""
Octmap Embedding Testbed
A modular framework for testing various embedding approaches for drone navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv                           # v2.3+
from spconv.pytorch import SparseConvTensor
import csv, datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import voxelsim
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, IterableDataset
import os

from losses import make_recon_loss, make_aux_loss
from scipy.ndimage import distance_transform_edt as edt
import time, collections

# ============== Data Structures ==============
@dataclass
class VoxelData:
    """Container for voxel octmap data"""
    occupied_coords: torch.Tensor  # [N, 3] coordinates of occupied voxels
    values: torch.Tensor           # [N] values (filled, sparse, etc)``
    bounds: torch.Tensor           # [3] max bounds of the octmap TODO: Not sure if we can have this, it might have to infer itself/ be adaptive
    drone_pos: torch.Tensor        # [3] current drone position
    
    def to_device(self, device):
        return VoxelData(
            occupied_coords=self.occupied_coords.to(device),
            values=self.values.to(device),
            bounds=self.bounds.to(device),
            drone_pos=self.drone_pos.to(device)
        )


# ============== Base Classes ==============
class EmbeddingEncoder(ABC):
    """Base class for all encoders"""
    @abstractmethod
    def encode(self, voxel_data: VoxelData) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass


class EmbeddingDecoder(ABC):
    """Base class for all decoders"""
    @abstractmethod
    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pass


class LossHead(ABC, nn.Module):
    variable_length_target: bool = False  # override in subclasses if needed

    def __init__(self, logits_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.logits_fn = logits_fn

    def bind_logits(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.logits_fn = fn
        return self

    @abstractmethod
    def forward(self, embedding: torch.Tensor, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return 'prediction' used by compute_loss.
        If logits is None and self.logits_fn is set, the head may call self.logits_fn(embedding).
        """

    @abstractmethod
    def compute_loss(self, prediction: torch.Tensor, target) -> torch.Tensor:
        pass


# ============== Simple CNN Autoencoder ==============
class SimpleCNNEncoder(EmbeddingEncoder, nn.Module):
    def __init__(self, voxel_size=48, embedding_dim=128):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        
        # 3D CNN encoder
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Calculate flattened size
        self.flat_size = 128 * (voxel_size // 8) ** 3
        self.fc = nn.Linear(self.flat_size, embedding_dim)
        
    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        # Build one dense grid per sample, then stack
        grids = [self._sparse_to_dense(vd) for vd in voxel_batch]  # each [1,1,D,D,D]
        x = torch.cat(grids, dim=0)                                # [B,1,D,D,D]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)                                           # [B, F]
        return self.fc(x) 
    
    def _sparse_to_dense(self, voxel_data: VoxelData) -> torch.Tensor:
        """Convert sparse voxel coordinates to dense grid"""

        grid = torch.zeros((1, 1, self.voxel_size, self.voxel_size, self.voxel_size),
                          device=voxel_data.occupied_coords.device, dtype=torch.float32)
        
        if voxel_data.occupied_coords.shape[0] > 0:
            # Normalize coordinates to grid size
            coords = voxel_data.occupied_coords.float()
            coords = coords / voxel_data.bounds.float() * (self.voxel_size - 1)
            coords = coords.long().clamp(0, self.voxel_size - 1)
            
            # Fill grid
            grid[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = voxel_data.values
     

        return grid
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class SimpleCNNDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=128, voxel_size=48):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        
        # Calculate sizes
        self.init_size = voxel_size // 8
        self.flat_size = 128 * self.init_size ** 3
        
        self.fc = nn.Linear(embedding_dim, self.flat_size)
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        
    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None, skips=None) -> Dict[str, torch.Tensor]:
        x = self.fc(embedding)
        x = x.view(-1, 128, self.init_size, self.init_size, self.init_size)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        logits = self.deconv3(x)
        
        return {"logits": logits}


# Mein Encoder

class MeinEncoder(EmbeddingEncoder, nn.Module):
    def __init__(self, voxel_size=48, embedding_dim=128):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        
        # 3D CNN encoder
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size
        self.flat_size = 128 * (voxel_size // 8) ** 3
        self.fc = nn.Linear(self.flat_size, embedding_dim)
        
    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        # Build one dense grid per sample, then stack
        grids = [self._sparse_to_dense(vd) for vd in voxel_batch]  # each [1,1,D,D,D]
        x = torch.cat(grids, dim=0)                                # [B,1,D,D,D]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)                                           # [B, F]
        return self.fc(x) 
    
    def _sparse_to_dense(self, voxel_data: VoxelData) -> torch.Tensor:
        """Convert sparse voxel coordinates to dense grid"""

        grid = torch.zeros((1, 1, self.voxel_size, self.voxel_size, self.voxel_size),
                          device=voxel_data.occupied_coords.device, dtype=torch.float32)
        
        if voxel_data.occupied_coords.shape[0] > 0:
            # Normalize coordinates to grid size
            coords = voxel_data.occupied_coords.float()
            coords = coords / voxel_data.bounds.float() * (self.voxel_size - 1)
            coords = coords.long().clamp(0, self.voxel_size - 1)
            
            # Fill grid
            grid[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = voxel_data.values
     

        return grid
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class MeinDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=128, voxel_size=48):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        
        # Calculate sizes
        self.init_size = voxel_size // 8
        self.flat_size = 128 * self.init_size ** 3
        
        self.fc = nn.Linear(embedding_dim, self.flat_size)
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        
    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None, skips=None) -> Dict[str, torch.Tensor]:
        x = self.fc(embedding)
        x = x.view(-1, 128, self.init_size, self.init_size, self.init_size)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        logits = self.deconv3(x)
        
        return {"logits": logits}


# ---------------------------------------------------------------------------


class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm3d(c), nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm3d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class UNet3DEncoder(EmbeddingEncoder, nn.Module):
    def __init__(self, voxel_size=48, embedding_dim=512):   # pick any ≤1000
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(                 # 48³ → 24³ (32ch)
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            ResBlock3D(32),
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            ResBlock3D(64)
        )
        self.down = nn.Sequential(                 # 24³ → 12³ (128ch)
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            ResBlock3D(128)
        )

        self.fc = nn.Linear(128, embedding_dim)

    # same helper you already have
    def _sparse_to_dense(self, vd):
        g = torch.zeros((1,1,self.voxel_size,self.voxel_size,self.voxel_size),
                        device=vd.occupied_coords.device)
        if vd.occupied_coords.numel():
            xyz = (vd.occupied_coords.float()/vd.bounds.float()*(self.voxel_size-1)).long()
            xyz = xyz.clamp(0, self.voxel_size-1)  # ← add this
            g[0,0, xyz[:,0], xyz[:,1], xyz[:,2]] = vd.values
        return g

    def encode(self, voxel_batch):
        x = torch.cat([self._sparse_to_dense(v) for v in voxel_batch], 0)
        x = self.stem(x)
        x = self.down(x)                 # [B,128,12,12,12]
        x = x.mean(dim=[2,3,4])          # global average-pool
        return self.fc(x)                # **only latent**, no skips

    def get_embedding_dim(self):
        return self.embedding_dim


# ------------------------------------------------------------------
#  🄱 Decoder: latent → 12³ → 24³ → 48³ (no skip cat)
# ------------------------------------------------------------------
class UNet3DDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=512, voxel_size=48):
        super().__init__()
        self.voxel_size = voxel_size
        self.init_size  = voxel_size // 4       # 12
        self.fc = nn.Linear(embedding_dim, 128 * self.init_size**3)

        self.up1 = nn.Sequential(               # 12³ → 24³
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            ResBlock3D(64)
        )
        self.up0 = nn.Sequential(               # 24³ → 48³
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            ResBlock3D(32)
        )
        self.out = nn.Conv3d(32, 3, 3, padding=1)

    def decode(self, embedding, skips=None):    # skips ignored
        x = self.fc(embedding)
        x = x.view(-1, 128, self.init_size, self.init_size, self.init_size)
        x = self.up1(x)
        x = self.up0(x)
        return {"logits": self.out(x)}
    # ----------------------------------------------------------------------

# ===== ResNet-style 3D encoder/decoder (GroupNorm) =====
class _GN3d(torch.nn.GroupNorm):
    def __init__(self, num_channels, groups=32):
        super().__init__(num_groups=min(groups, num_channels), num_channels=num_channels)

class _ResBlock3D_GN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.gn1   = _GN3d(c); self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.gn2   = _GN3d(c)
    def forward(self, x):
        return self.act(x + self.gn2(self.conv2(self.act(self.gn1(self.conv1(x))))))
    
class ResNet3DEncoder(EmbeddingEncoder, nn.Module):
    """48/96/etc → 24 → 12 → 6; GAP → latent. No skips."""
    def __init__(self, voxel_size=48, embedding_dim=1024, widths=(32, 64, 128)):
        super().__init__()
        assert voxel_size % 8 == 0, "ResNet3DEncoder expects voxel_size divisible by 8"
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        c0, c1, c2 = widths

        self.stem = nn.Sequential(
            nn.Conv3d(1, c0, 5, stride=2, padding=2, bias=False), _GN3d(c0), nn.ReLU(True),
            _ResBlock3D_GN(c0),
        )  # x/2
        self.stage1 = nn.Sequential(
            nn.Conv3d(c0, c1, 3, stride=2, padding=1, bias=False), _GN3d(c1), nn.ReLU(True),
            _ResBlock3D_GN(c1),
        )  # x/4
        self.stage2 = nn.Sequential(
            nn.Conv3d(c1, c2, 3, stride=2, padding=1, bias=False), _GN3d(c2), nn.ReLU(True),
            _ResBlock3D_GN(c2),
        )  # x/8

        self.proj = nn.Linear(c2, embedding_dim)

    def _sparse_to_dense(self, vd: VoxelData) -> torch.Tensor:
        g = torch.zeros((1,1,self.voxel_size,self.voxel_size,self.voxel_size),
                        device=vd.occupied_coords.device, dtype=torch.float32)
        if vd.occupied_coords.numel():
            xyz = (vd.occupied_coords.float()/vd.bounds.float()*(self.voxel_size-1)).long().clamp(0,self.voxel_size-1)
            g[0,0, xyz[:,0], xyz[:,1], xyz[:,2]] = vd.values
        return g

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        x = torch.cat([self._sparse_to_dense(v) for v in voxel_batch], 0)  # [B,1,D,H,W]
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)           # [B,c2,D/8,H/8,W/8]
        x = x.mean(dim=(2,3,4))                                            # GAP → [B,c2]
        return self.proj(x)                                                # [B,E]

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class ResNet3DDecoder(EmbeddingDecoder, nn.Module):
    """latent → 1/8 grid → 1/4 → 1/2 → full; 3-class logits."""
    def __init__(self, embedding_dim=1024, voxel_size=48, widths=(128, 64, 32)):
        super().__init__()
        assert voxel_size % 8 == 0, "ResNet3DDecoder expects voxel_size divisible by 8"
        self.voxel_size = voxel_size
        self.init_size  = voxel_size // 8
        c2, c1, c0 = widths

        self.fc = nn.Linear(embedding_dim, c2 * self.init_size**3)

        self.up1 = nn.Sequential(  # 1/8 → 1/4
            nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1, bias=False), _GN3d(c1), nn.ReLU(True),
            _ResBlock3D_GN(c1),
        )
        self.up2 = nn.Sequential(  # 1/4 → 1/2
            nn.ConvTranspose3d(c1, c0, 4, stride=2, padding=1, bias=False), _GN3d(c0), nn.ReLU(True),
            _ResBlock3D_GN(c0),
        )
        self.up3 = nn.Sequential(  # 1/2 → 1
            nn.ConvTranspose3d(c0, 32, 4, stride=2, padding=1, bias=False), _GN3d(32), nn.ReLU(True),
        )
        self.out = nn.Conv3d(32, 3, 3, padding=1)

    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None, skips=None) -> Dict[str, torch.Tensor]:
        x = self.fc(embedding).view(-1, self.up1[0].in_channels, self.init_size, self.init_size, self.init_size)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        logits = self.out(x)
        return {"logits": logits}

# ===== Implicit MLP decoder (chunked) =====
class ImplicitMLPDecoder(EmbeddingDecoder, nn.Module):
    """
    Predicts per-voxel logits from (z, xyz) with an MLP; returns dense [B,3,D,H,W].
    Good for smooth surfaces; can be slower — use chunking.
    """
    def __init__(self, embedding_dim=1024, voxel_size=48, hidden=256, depth=4, chunk=65536):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.depth = depth
        self.chunk = chunk

        layers = []
        in_dim = embedding_dim + 3
        for i in range(depth):
            out = hidden if i < depth - 1 else 3
            layers.append(nn.Linear(in_dim, out, bias=True))
            if i < depth - 1:
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden
        self.mlp = nn.Sequential(*layers)

        # precompute normalized grid in [-1, 1]³
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            indexing='ij'
        ), dim=-1).view(-1, 3)  # [V,3]
        self.register_buffer("grid", coords, persistent=False)

    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None, skips=None) -> Dict[str, torch.Tensor]:
        # embedding: [B,E]
        B, E = embedding.shape
        V = self.grid.shape[0]
        outs = []
        for b in range(B):
            z = embedding[b].unsqueeze(0).expand(V, -1)   # [V,E]
            feats = torch.cat([z, self.grid.to(z.dtype)], dim=1)  # [V,E+3]
            preds = []
            for i in range(0, V, self.chunk):
                preds.append(self.mlp(feats[i:i+self.chunk]))
            pred = torch.cat(preds, dim=0)  # [V,3]
            outs.append(pred.view(self.voxel_size, self.voxel_size, self.voxel_size, 3)
                             .permute(3,0,1,2))  # [3,D,H,W]
        logits = torch.stack(outs, dim=0)  # [B,3,D,H,W]
        return {"logits": logits}
    
# Point MLP Encoder

class PointMLPEncoder(EmbeddingEncoder, nn.Module):
    """
    Encodes directly from sparse points: [x,y,z] (normalized) + value.
    Robust at coarse grids; no dense voxel tensor needed.
    """
    def __init__(self, voxel_size=48, embedding_dim=512,
                 fourier_feats: int = 0, max_points: int = 8192):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        self.max_points = max_points
        self.k = fourier_feats

        in_dim = 4  # x,y,z,val
        if self.k > 0:
            in_dim += 6 * self.k  # sin/cos on 3 coords with k bands

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True),
            nn.Linear(128, 256),     nn.ReLU(True),
            nn.Linear(256, 256),     nn.ReLU(True),
        )
        self.head = nn.Linear(256 * 2, embedding_dim)  # mean+max concat

        # precompute Fourier frequencies if requested
        if self.k > 0:
            bands = 2.0 ** torch.arange(self.k).float() * np.pi
            self.register_buffer("bands", bands, persistent=False)

    def get_embedding_dim(self): return self.embedding_dim

    def _featify(self, vd: VoxelData) -> torch.Tensor:
        if vd.occupied_coords.numel() == 0:
            return torch.zeros((1, 4 + (6*self.k if self.k>0 else 0)),
                               device=self.head.weight.device)

        xyz = vd.occupied_coords.float() / vd.bounds.float()  # [0,1]
        val = vd.values.float().unsqueeze(1)                  # [N,1]
        x = torch.cat([xyz, val], dim=1)                      # [N,4]

        if self.k > 0:
            # Fourier features on coords mapped to [-1,1]
            uvw = (xyz * 2 - 1)                               # [-1,1]
            # [N,3] -> [N,3,k]
            ang = uvw.unsqueeze(-1) * self.bands
            sins = torch.sin(ang); coss = torch.cos(ang)
            # concat along last dim -> [N,3,2k], then flatten 3*(2k)
            fourier = torch.cat([sins, coss], dim=-1).reshape(uvw.size(0), -1)
            x = torch.cat([x, fourier], dim=1)

        # random or strided subsample if too many points
        if x.size(0) > self.max_points:
            idx = torch.randperm(x.size(0), device=x.device)[:self.max_points]
            x = x[idx]
        return x

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        embs = []
        for vd in voxel_batch:
            x = self._featify(vd)              # [N,d]
            f = self.mlp(x)                    # [N,256]
            mean = f.mean(0)
            mx, _ = f.max(0)
            embs.append(self.head(torch.cat([mean, mx], dim=0)))
        return torch.stack(embs, dim=0)        # [B,E]


class PointNetPPLiteFPEncoder(EmbeddingEncoder, nn.Module):
    """
    Hierarchical point-set encoder (PointNet++-lite) with FEATURE PROPAGATION.
    Stages:
      SA0: sample K0 centers from raw points; group from raw points; per-group MLP on [Δxyz, val, ||Δ||, Fourier(Δ)] → pool → C
      SA1+: sample Ki from previous centers; group from previous centers; MLP on [Δxyz, ||Δ||, prev_center_feat, Fourier(Δ)] → pool → C
    Final: global mean+max over last-stage tokens, linear head → embedding_dim.
    Notes:
      - Sampling is voxel-grid (fast, uniform-ish). Flip use_fps=True for FPS.
      - Neighborhood query is brute-force (pairwise) with radius + top-k cap; fine at N≤~8k.
    """
    def __init__(
        self,
        voxel_size: int = 48,
        embedding_dim: int = 512,
        max_points: int = 8192,                 # cap raw N
        stages=((2048, 2.5), (512, 5.0), (128, 9.0)),  # (K, radius_in_voxels)
        nbrs_cap: int = 64,                     # max neighbors per center
        fourier_feats: int = 8,                 # 0 disables Fourier(Δxyz)
        width: int = 128,                       # per-stage hidden (C)
        use_fps: bool = False                   # default voxel-grid subsample
    ):
        super().__init__()
        self.voxel_size    = voxel_size
        self.embedding_dim = embedding_dim
        self.max_points    = max_points
        self.stages        = list(stages)
        self.nbrs_cap      = int(nbrs_cap)
        self.k_fourier     = int(fourier_feats)
        self.width         = int(width)
        self.use_fps       = bool(use_fps)

        # Stage-specific MLPs: SA0 sees val, later stages see prev feature (C)
        in0 = 3 + 1 + 1 + (6*self.k_fourier if self.k_fourier > 0 else 0)          # Δxyz, val, ||Δ||, PE
        inL = 3 + 1 + self.width + (6*self.k_fourier if self.k_fourier > 0 else 0) # Δxyz, ||Δ||, prev_feat, PE
        self.stage_mlps = nn.ModuleList()
        self.post_mlps  = nn.ModuleList()
        for si in range(len(self.stages)):
            ind = in0 if si == 0 else inL
            self.stage_mlps.append(nn.Sequential(
                nn.Linear(ind, self.width), nn.ReLU(True),
                nn.Linear(self.width, self.width), nn.ReLU(True),
            ))
            self.post_mlps.append(nn.Sequential(
                nn.Linear(2*self.width, self.width), nn.ReLU(True)  # after mean+max concat
            ))

        # Final head after global mean+max over tokens
        self.head = nn.Linear(2*self.width, embedding_dim)

        # Fourier bands
        if self.k_fourier > 0:
            bands = (2.0 ** torch.arange(self.k_fourier).float()) * np.pi
            self.register_buffer("bands", bands, persistent=False)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    # -------- sampling helpers --------
    @staticmethod
    def _pairwise_dist2(a: torch.Tensor, b: torch.Tensor):
        # a: [M,3], b: [N,3] -> [M,N]
        a2 = (a**2).sum(1, keepdim=True)
        b2 = (b**2).sum(1).unsqueeze(0)
        return (a2 + b2 - 2.0 * (a @ b.t())).clamp_min_(0.0)

    @staticmethod
    def _fps(xyz: torch.Tensor, K: int):
        N = xyz.size(0); K = min(K, N)
        if K == N:
            return torch.arange(N, device=xyz.device, dtype=torch.long)
        sel = torch.empty(K, dtype=torch.long, device=xyz.device)
        sel[0] = torch.randint(0, N, (1,), device=xyz.device)
        dists = torch.full((N,), float('inf'), device=xyz.device)
        for i in range(1, K):
            p = xyz[sel[i-1]].unsqueeze(0)
            d2 = ((xyz - p)**2).sum(1)
            dists = torch.minimum(dists, d2)
            sel[i] = torch.argmax(dists)
        return sel

    def _voxel_grid_subsample(self, xyz_int: torch.Tensor, target_k: int):
        """
        Keep ≤1 point per coarse cell; choose cell size to hit ~target_k total.
        xyz_int: [N,3] int voxel coords in [0, side-1].
        """
        N = xyz_int.size(0)
        if N <= target_k:
            return torch.arange(N, device=xyz_int.device, dtype=torch.long)
        side = int(self.voxel_size)
        cell = max(1, int(np.ceil(side / (target_k ** (1/3)))))
        cells = (xyz_int // cell).to(torch.int64)  # [N,3]
        # 3D hash → 1D
        h = cells[:, 0] + cells[:, 1] * 73856093 + cells[:, 2] * 19349663
        # keep first occurrence per hash
        _, keep = torch.unique(h, return_index=True)
        if keep.numel() > target_k:
            perm = torch.randperm(keep.numel(), device=keep.device)
            keep = keep[perm[:target_k]]
        return keep

    def _fourier(self, x: torch.Tensor):
        # x: [*,3] roughly in [-1,1]
        ang = x.unsqueeze(-1) * self.bands  # [*,3,B]
        s, c = torch.sin(ang), torch.cos(ang)
        return torch.cat([s, c], dim=-1).reshape(*x.shape[:-1], -1)

    # -------- one SA stage with feature propagation --------
    def _stage(self, src_xyz: torch.Tensor, src_feat: torch.Tensor,
               base_xyz: torch.Tensor, K: int, radius: float, si: int):
        """
        src_xyz:  [Ns,3] points to search neighbors from
        src_feat: [Ns,1] (SA0) or [Ns,C] (SA1+)
        base_xyz: [Nb,3] points to sample centers from (usually same as src_xyz)
        Returns:
          centers_xyz:  [K,3]
          centers_feat: [K,C]
        """
        device = src_xyz.device; C = self.width
        # choose centers
        if self.use_fps:
            idx = self._fps(base_xyz, K)
        else:
            idx = self._voxel_grid_subsample(base_xyz.long().to(torch.int64), K)
        centers_xyz = base_xyz[idx]  # [K,3]

        # neighbor gather
        d2 = self._pairwise_dist2(centers_xyz, src_xyz)   # [K,Ns]
        r2 = float(radius * radius)
        d2_masked = d2.clone()
        d2_masked[d2_masked > r2] = float('inf')
        k_take = min(self.nbrs_cap, src_xyz.size(0))
        vals, nn_idx = torch.topk(d2_masked, k=k_take, dim=1, largest=False)
        # fallback to true NN where all inf
        none_mask = torch.isinf(vals[:, 0])
        if none_mask.any():
            fallback = d2.argmin(dim=1)
            nn_idx[none_mask, 0] = fallback[none_mask]
            vals[none_mask, 0] = d2[none_mask, fallback[none_mask]]

        nbr_xyz  = src_xyz[nn_idx]                           # [K,k,3]
        delta    = (nbr_xyz - centers_xyz.unsqueeze(1)) / max(radius, 1e-6)  # [-1,1]-ish
        distn    = torch.sqrt(vals.clamp_min(0.0)).unsqueeze(-1) / max(radius, 1e-6)  # [K,k,1]
        nbr_feat = src_feat[nn_idx]                          # [K,k, 1 or C]

        parts = [delta, distn, nbr_feat]
        if self.k_fourier > 0:
            parts.append(self._fourier(delta))
        x = torch.cat(parts, dim=-1)                        # [K,k,F_in]

        # per-neighbor MLP → [K,k,C]
        Kk = x.shape[0] * x.shape[1]
        x = self.stage_mlps[si](x.view(Kk, -1)).view(x.shape[0], x.shape[1], -1)

        # masked mean + max
        valid = ~torch.isinf(vals)                          # [K,k]
        valid_f = valid.float().unsqueeze(-1)
        counts = valid_f.sum(dim=1).clamp_min_(1.0)         # [K,1]
        mean = (x * valid_f).sum(dim=1) / counts            # [K,C]
        neg_big = torch.finfo(x.dtype).min
        x_masked = torch.where(valid.unsqueeze(-1), x, neg_big)
        mx, _ = x_masked.max(dim=1)                         # [K,C]
        out = torch.cat([mean, mx], dim=-1)                 # [K,2C]
        out = self.post_mlps[si](out)                       # [K,C]
        return centers_xyz, out

    # -------- encode one sample --------
    def _encode_one(self, vd: VoxelData) -> torch.Tensor:
        dev = self.head.weight.device
        if vd.occupied_coords.numel() == 0:
            return torch.zeros(self.embedding_dim, device=dev)

        xyz = vd.occupied_coords.to(device=dev, dtype=torch.float32)        # [N,3] in voxel units
        val = vd.values.to(device=dev, dtype=torch.float32).unsqueeze(1)    # [N,1]

        # optional cap N
        if xyz.size(0) > self.max_points:
            keep = self._voxel_grid_subsample(xyz.long().to(torch.int64), self.max_points)
            xyz, val = xyz[keep], val[keep]

        # SA0: from raw points
        K0, r0 = self.stages[0]
        c_xyz, c_feat = self._stage(src_xyz=xyz, src_feat=val, base_xyz=xyz, K=K0, radius=r0, si=0)

        # SA1..n: from previous centers (propagate features)
        for si in range(1, len(self.stages)):
            Ki, ri = self.stages[si]
            c_xyz, c_feat = self._stage(
                src_xyz=c_xyz, src_feat=c_feat, base_xyz=c_xyz,
                K=Ki, radius=ri, si=si
            )

        # Global mean+max over tokens → embedding
        mean = c_feat.mean(dim=0)
        mx, _ = c_feat.max(dim=0)
        return self.head(torch.cat([mean, mx], dim=0))

    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        return torch.stack([self._encode_one(vd) for vd in voxel_batch], dim=0)



class CrossAttnTokensEncoder(EmbeddingEncoder, nn.Module):
    """
    Conv ↓ to 1/8 grid → tokens; learnable queries attend to tokens; pooled to latent.
    """
    def __init__(self, voxel_size=48, embedding_dim=1024,
                 token_channels=256, num_queries=20, heads=4, layers=2):
        super().__init__()
        assert voxel_size % 8 == 0
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        self.proj = nn.Identity()
        g = voxel_size // 8

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 5, stride=2, padding=2), nn.ReLU(True),  # /2
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(True), # /4
            nn.Conv3d(64, token_channels, 3, stride=2, padding=1), nn.ReLU(True), # /8
        )

        self.queries = nn.Parameter(torch.randn(num_queries, token_channels))

        self.mhas = nn.ModuleList([
            nn.MultiheadAttention(token_channels, heads, batch_first=True)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_channels, 4*token_channels), nn.ReLU(True),
                nn.Linear(4*token_channels, token_channels)
            ) for _ in range(layers)
        ])
        self.norm_q = nn.LayerNorm(token_channels)
        self.norm_t = nn.LayerNorm(token_channels)
   

    def _sparse_to_dense(self, vd):
        g = torch.zeros((1,1,self.voxel_size,self.voxel_size,self.voxel_size),
                        device=vd.occupied_coords.device)
        if vd.occupied_coords.numel():
            xyz = (vd.occupied_coords.float()/vd.bounds.float()*(self.voxel_size-1)).long().clamp(0,self.voxel_size-1)
            g[0,0, xyz[:,0], xyz[:,1], xyz[:,2]] = vd.values
        return g

    def encode(self, voxel_batch):
        x = torch.cat([self._sparse_to_dense(v) for v in voxel_batch], 0)  # [B,1,D,H,W]
        x = self.conv(x)                                                   # [B,C,g,g,g]
        B, C, g, _, _ = x.shape
        tokens = x.view(B, C, g*g*g).transpose(1, 2)                       # [B,T,C]
        tokens = self.norm_t(tokens)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)      # [B,Q,C]
        for mha, ffn in zip(self.mhas, self.ffns):
            q2,_ = mha(self.norm_q(q), tokens, tokens)
            q = q + q2
            q = q + ffn(q)                                    # still [B,Q,C]
        z = q.reshape(B, -1)                                  # [B,Q*C] = [B,5120]
        return z

    def get_embedding_dim(self): return self.embedding_dim

class SepConv3D(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_in, kernel_size=(3,1,1), padding=(1,0,0), groups=c_in, bias=False),
            nn.Conv3d(c_in, c_in, kernel_size=(1,3,1), padding=(0,1,0), groups=c_in, bias=False),
            nn.Conv3d(c_in, c_in, kernel_size=(1,1,3), padding=(0,0,1), groups=c_in, bias=False),
            nn.Conv3d(c_in, c_out, kernel_size=1, bias=False),
            nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class Factorised3DDecoder(EmbeddingDecoder, nn.Module):
    """
    latent → 1/8 grid → 1/4 → 1/2 → full using separable 3D convs
    """
    def __init__(self, embedding_dim=512, voxel_size=48, base=96):
        super().__init__()
        assert voxel_size % 8 == 0
        self.voxel_size = voxel_size
        s = voxel_size // 8
        self.fc = nn.Linear(embedding_dim, base * s * s * s)

        self.up1 = nn.Sequential(
        nn.ConvTranspose3d(base, base//2, 4, stride=2, padding=1, bias=False),
        SepConv3D(base//2, base//2),   # ← add c_out
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base//2, base//4, 4, stride=2, padding=1, bias=False),
            SepConv3D(base//4, base//4),   # ← add c_out
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(base//4, base//8, 4, stride=2, padding=1, bias=False),
            SepConv3D(base//8, base//8),   # ← add c_out
        )

        self.out = nn.Conv3d(base//8, 3, 1)

    def decode(self, embedding, query_points=None, skips=None):
        B = embedding.size(0)
        s = self.voxel_size // 8
        x = self.fc(embedding).view(B, -1, s, s, s)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        return {"logits": self.out(x)}

class ImplicitFourierDecoder(EmbeddingDecoder, nn.Module):
    """
    Stronger implicit field: concat latent with multi-scale Fourier features of xyz.
    Returns dense logits [B,3,D,H,W]. Chunked to fit memory.
    """
    def __init__(self, embedding_dim=512, voxel_size=48, fourier_bands=8, hidden=256, depth=4, chunk=65536):
        super().__init__()
        self.voxel_size = voxel_size
        self.chunk = chunk

        self.register_buffer("xyz", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            torch.linspace(-1, 1, voxel_size),
            indexing='ij'
        ), dim=-1).view(-1,3), persistent=False)  # [V,3]

        freqs = 2.0 ** torch.arange(fourier_bands)
        self.register_buffer("freqs", freqs * np.pi, persistent=False)
        in_dim = embedding_dim + 3 + 6*fourier_bands  # xyz + sin/cos
        layers = []
        d = in_dim
        for i in range(depth-1):
            layers += [nn.Linear(d, hidden), nn.ReLU(True)]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.mlp = nn.Sequential(*layers)

    def _encode_xyz(self, xyz):
        # xyz: [N,3] in [-1,1]
        ang = xyz.unsqueeze(-1) * self.freqs  # [N,3,B]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).view(xyz.size(0), -1)
        return torch.cat([xyz, pe], dim=1)   # [N, 3+6B]

    def decode(self, embedding, query_points=None, skips=None):
        B, E = embedding.shape
        V = self.xyz.size(0)
        xyz_pe = self._encode_xyz(self.xyz)

        outs = []
        for b in range(B):
            z = embedding[b].unsqueeze(0).expand(V, -1)           # [V,E]
            feats = torch.cat([z, xyz_pe.to(z.dtype)], dim=1)     # [V,E+pos]
            preds = []
            for i in range(0, V, self.chunk):
                preds.append(self.mlp(feats[i:i+self.chunk]))
            logits = torch.cat(preds, dim=0)                      # [V,3]
            outs.append(logits.view(self.voxel_size, self.voxel_size, self.voxel_size, 3)
                              .permute(3,0,1,2))
        return {"logits": torch.stack(outs, 0)}  # [B,3,D,H,W]






# ============== Loss Heads ==============
class AuxFnHead(LossHead):
    """Wraps a function from losses.make_aux_loss to behave like a LossHead."""
    variable_length_target: bool = False  # set True if the fn needs lists (e.g. chamfer)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, embedding, logits=None):
        # We just pass logits through; compute_loss will call the fn with logits + targets
        if logits is None and self.logits_fn is not None:
            logits = self.logits_fn(embedding)
        return logits

    def compute_loss(self, prediction, targets):
        # prediction == logits (from forward)
        return self.fn(prediction, targets)
    
class ContourHead(LossHead, nn.Module):
    """Predicts max altitude contour map from top-down view"""
    def __init__(self, embedding_dim=128, map_size=32):
        super().__init__()
        self.map_size = map_size
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, map_size * map_size)

        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(embedding))
        x = self.fc2(x)
        return x.view(-1, self.map_size, self.map_size)
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)


class RelativeOffsetHead(LossHead, nn.Module):
    """Predicts relative offsets to K nearest obstacles/cover"""
    def __init__(self, embedding_dim=128, k_nearest=5):
        super().__init__()
        self.k_nearest = k_nearest
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, k_nearest * 3)  # 3D offsets
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(embedding))
        x = self.fc2(x)
        return x.view(-1, self.k_nearest, 3)
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)


class CoverMaskHead(LossHead, nn.Module):
    """Predicts binary mask of cover within radius"""
    def __init__(self, embedding_dim=128, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, grid_size * grid_size * grid_size)
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(embedding))
        x = self.fc2(x)
        return torch.sigmoid(x.view(-1, self.grid_size, self.grid_size, self.grid_size))
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(prediction, target)
class TopDownHeightHead(LossHead, nn.Module):
    """Predict 2D max-height map (x,z) normalized to [0,1]."""
    def __init__(self, embedding_dim=128, map_size=32):
        super().__init__()
        self.map_size = map_size
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(True),
            nn.Linear(256, map_size*map_size)
        )

    def forward(self, embedding):                    # [B,E] -> [B,S,S]
        return self.net(embedding).view(-1, self.map_size, self.map_size)

    def compute_loss(self, prediction, target):
        return F.l1_loss(prediction, target)         # robust to outliers

class DistanceTransformHead(LossHead, nn.Module):
    """Predict a coarse 3D distance transform to nearest occupancy."""
    def __init__(self, embedding_dim=128, grid=24):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.ReLU(True),
            nn.Linear(512, grid*grid*grid)
        )

    def forward(self, embedding):
        return self.net(embedding).view(-1, self.grid, self.grid, self.grid)

    def compute_loss(self, prediction, target_dt):   # expect normalized DT
        return F.smooth_l1_loss(prediction, target_dt)

class OccupancyProjectionHead(LossHead, nn.Module):
    """Predict 3D occupancy at lower res for multi-scale supervision."""
    def __init__(self, embedding_dim=128, grid=24, out_ch=1):
        super().__init__()
        self.grid = grid
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 1024), nn.ReLU(True),
            nn.Linear(1024, out_ch*grid*grid*grid)
        )

    def forward(self, embedding):
        x = self.fc(embedding).view(-1, 1, self.grid, self.grid, self.grid)
        return torch.sigmoid(x)  # prob of occupancy

    def compute_loss(self, prediction, target_occ):  # [B,1,G,G,G] float∈[0,1]
        return F.binary_cross_entropy(prediction, target_occ)

# ============== Training Framework ==============
class EmbeddingTrainer:
    def __init__(self, encoder: EmbeddingEncoder, decoder: EmbeddingDecoder, 
                 loss_heads: Dict[str, LossHead], device='cuda', lr=1e-3, recon_loss = None):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_heads = loss_heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        

        
        # Move models to device and collect parameters TODO: Not too sure what this does, I think it moves something to GPU if possible
        if isinstance(encoder, nn.Module):
            encoder.to(self.device)

        if isinstance(decoder, nn.Module):
            decoder.to(self.device)

        # ↓↓↓ wrap and move once
        self.loss_heads = nn.ModuleDict(loss_heads)
        self.loss_heads.to(self.device)

        # collect params cleanly
        all_params = []
        if isinstance(encoder, nn.Module):
            all_params += list(encoder.parameters())
        if isinstance(decoder, nn.Module):
            all_params += list(decoder.parameters())
        all_params += list(self.loss_heads.parameters())

        self.recon_loss_fn = recon_loss or (lambda l,t,extra=None: F.cross_entropy(l, t))
        
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
    
    def train_step(self, voxel_batch: List[VoxelData], target_batch: List[Dict[str, torch.Tensor]],
                   loss_weights: Dict[str, float]) -> Dict[str, float]:
        """Single training step"""
        # Zeros stored gradients
        self.optimizer.zero_grad()
        
        # Encode
        out = self.encoder.encode(voxel_batch)
        if isinstance(out, tuple):
            embedding, skips = out
        else:
            embedding, skips = out, None

        # Decode for reconstruction loss
        reconstruction = self.decoder.decode(embedding, skips=skips)
        logits = reconstruction["logits"]    
        # Compute losses
        losses = {}
        
        # Reconstruction loss
        if "reconstruction" in loss_weights:
            recon_targets = torch.cat(
            [self._create_reconstruction_target(vd) for vd in voxel_batch], dim=0
        )   

            # TODO: Check adaptive histogram weighting
            
            # hist = torch.bincount(recon_target.view(-1), minlength=3).float() 
            # inv = 1.0 / (hist + 1e-6)
            # weights = (inv / inv.sum()) * 3.0

            #weights = torch.tensor([0.2, 1.0, 1.5], device=logits.device)

            # --- dynamic class weights (median-frequency + EMA) ---
            num_classes = 3
            counts = torch.bincount(recon_targets.view(-1), minlength=num_classes).float()
            freq = counts / counts.sum().clamp_min(1)

            # median-frequency balancing: w_c = median(freq)/freq_c
            med = freq[freq > 0].median()
            w = med / freq.clamp_min(1e-6)

            # optional: clamp extremes so a missing class doesn't explode the loss
            w = w.clamp(0.25, 4.0)

            # smooth over time (create the buffers the first time)
            if not hasattr(self, "class_weight_ema"):
                self.class_weight_ema = w.to(logits.device)
                self.class_weight_momentum = 0.9
            else:
                self.class_weight_ema = (
                    self.class_weight_momentum * self.class_weight_ema
                    + (1 - self.class_weight_momentum) * w.to(logits.device)
                )

            weights = self.class_weight_ema

            losses["reconstruction"] = self.recon_loss_fn(
            logits, recon_targets,
            extra={"class_weights": weights}
        )

        # inside EmbeddingTrainer.train_step, for aux heads loop

        merged_targets = {}
        if len(target_batch) > 0:
            keys = set().union(*[t.keys() for t in target_batch])
            for k in keys:
                if k == "points":
                    merged_targets[k] = [t[k] for t in target_batch if k in t]
                else:
                    merged_targets[k] = torch.stack(
                        [t[k] for t in target_batch if k in t], dim=0
                    ).to(self.device)

        # ---- auxiliary heads ----
        for name, head in self.loss_heads.items():
            if name not in loss_weights:
                continue

            # Function-based aux heads (use merged_targets, skip name check)
            if isinstance(head, AuxFnHead):
                preds = head(embedding, logits=logits)  # logits reused from recon
                losses[name] = head.compute_loss(preds, merged_targets)
                continue

            # Legacy per-head targets (keep old key check)
            if not all(name in tb for tb in target_batch):
                continue
            preds = head(embedding, logits=logits)
            if getattr(head, "variable_length_target", False):
                t_for_head = [tb[name] for tb in target_batch]
            else:
                t_for_head = torch.stack([tb[name] for tb in target_batch], dim=0)
            losses[name] = head.compute_loss(preds, t_for_head)

        # Total loss
        total_loss = sum(loss_weights.get(name, 0) * loss for name, loss in losses.items())
        losses["total"] = total_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def _create_reconstruction_target(self, voxel_data: VoxelData) -> torch.Tensor:
        voxel_size = self.encoder.voxel_size if hasattr(self.encoder, 'voxel_size') else 64
        # default empty = 0
        target = torch.zeros(
        (1, self.encoder.voxel_size, self.encoder.voxel_size, self.encoder.voxel_size),
        device=self.device, dtype=torch.long
        )

        if voxel_data.occupied_coords.shape[0] > 0:
            coords = voxel_data.occupied_coords.float()
            coords = coords / voxel_data.bounds.float() * (voxel_size - 1)
            coords = coords.long().clamp(0, voxel_size - 1)

            # map your current values (1.0 filled, 0.5 sparse) to labels 2 / 1
            filled_mask = voxel_data.values >= 0.8
            sparse_mask = (~filled_mask) & (voxel_data.values >= 0.5)

            target[0, coords[filled_mask, 0], coords[filled_mask, 1], coords[filled_mask, 2]] = 2
            target[0, coords[sparse_mask, 0], coords[sparse_mask, 1], coords[sparse_mask, 2]] = 1

        return target


# ============== Data Generation ==============
class TerrainBatch(IterableDataset):
    """Infinite generator of full cubic worlds (no sub-volume)."""
    def __init__(self, world_size: int = 120,
                 build_dt: bool = True,
                 build_low: bool = True,
                 low_scale: int = 4,
                 build_com: bool = True,
                 build_points: bool = True):
        self.world_size = int(world_size)
        self.build_dt   = bool(build_dt)
        self.build_low  = bool(build_low)
        self.low_scale  = int(low_scale)
        self.build_com  = bool(build_com)
        self.build_pts  = bool(build_points)

    @staticmethod
    def world_to_voxeldata_np(world: voxelsim.VoxelGrid, side: int) -> VoxelData:
        coords_np, vals_np = world.as_numpy()  # (N,3) in (x,z,y) per your lib note; you already treat as (x,y,z)
        coords = torch.from_numpy(coords_np)
        vals   = torch.from_numpy(vals_np)
        return VoxelData(
            occupied_coords = coords,
            values          = vals,
            bounds          = torch.tensor([side]*3, dtype=torch.float32),
            drone_pos       = torch.tensor([side//2]*3, dtype=torch.float32),
        )

    def __iter__(self):
        side = self.world_size
        while True:
            g   = voxelsim.TerrainGenerator()
            cfg = voxelsim.TerrainConfig.default_py()
            cfg.set_seed_py(int(np.random.randint(0, 2**31)))
            # cfg.set_seed_py(42)
            cfg.set_world_size_py(side)
            g.generate_terrain_py(cfg)
            world = g.generate_world_py()

            voxel_data = self.world_to_voxeldata_np(world, side)
            targets = self._generate_targets(voxel_data)  # fast path
            yield voxel_data, targets

    # ---------- helpers ----------
    @staticmethod
    def _block_mean_3d(x: np.ndarray, scale: int) -> np.ndarray:
        """Average pool by integer factor `scale` in 3D."""
        if scale <= 1:
            return x
        D, H, W = x.shape
        assert D % scale == 0 and H % scale == 0 and W % scale == 0, "low_scale must divide each dim"
        x = x.reshape(D//scale, scale, H//scale, scale, W//scale, scale)
        return x.mean(axis=(1,3,5))

    def _generate_targets(self, voxel_data: VoxelData) -> Dict[str, torch.Tensor]:
        """
        Builds:
          - labels   : Long [D,H,W] {0 empty, 1 sparse, 2 filled}
          - occ_dense: Float [D,H,W] in [0,1] (filled=1.0, sparse=0.5)
          - occ_low  : Float [D/s,H/s,W/s] (if build_low)
          - dt       : Float [D,H,W] distance-to-boundary normalized (if build_dt)
          - com      : Float [3] center of mass (x,y,z) (if build_com)
          - points   : Tensor [N,3] GT occupied coords (if build_points; variable length)
        All returned on CPU (dataloader workers), moved to device later by your trainer.
        """
        side = int(voxel_data.bounds[0].item())
        D = H = W = side

        # ---- dense labels (vectorized) ----
        labels_np = np.zeros((D, H, W), dtype=np.uint8)  # 0 empty
        if voxel_data.occupied_coords.numel() > 0:
            coords = voxel_data.occupied_coords.cpu().numpy().astype(np.int64)  # [N,3] (x,y,z)
            vals   = voxel_data.values.cpu().numpy().astype(np.float32)

            # clamp just in case
            np.clip(coords, 0, side-1, out=coords)

            filled_mask = vals >= 0.8
            sparse_mask = (~filled_mask) & (vals >= 0.5)

            if filled_mask.any():
                xyz = coords[filled_mask]
                labels_np[xyz[:,0], xyz[:,1], xyz[:,2]] = 2
            if sparse_mask.any():
                xyz = coords[sparse_mask]
                labels_np[xyz[:,0], xyz[:,1], xyz[:,2]] = 1

        # ---- occupancy float (0, 0.5, 1.0) ----
        occ_np = (labels_np == 2).astype(np.float32) + 0.5 * (labels_np == 1).astype(np.float32)

        # ---- low-res occupancy by avg pooling ----
        if self.build_low:
            occ_low_np = self._block_mean_3d(occ_np, self.low_scale)
        else:
            occ_low_np = None

        # ---- distance transform to boundary (0 at boundary/occupied) ----
        if self.build_dt:
            # edt returns distance to zero; build both sides and take min for boundary distance
            occ_bool = occ_np > 0
            dt_to_empty = edt(occ_bool.astype(np.uint8))      # distance to empty (zeros)
            dt_to_occ   = edt((~occ_bool).astype(np.uint8))   # distance to occ (zeros where occ_bool=1)
            dt_bound    = np.minimum(dt_to_empty, dt_to_occ)
            # normalize (avoid div by zero)
            mx = float(dt_bound.max()) if dt_bound.size else 1.0
            dt_np = (dt_bound / (mx + 1e-6)).astype(np.float32)
        else:
            dt_np = None

        # ---- center of mass (weighted by occ_np) ----
        if self.build_com:
            mass = occ_np.sum()
            if mass > 1e-6:
                xs = np.arange(W, dtype=np.float32)
                ys = np.arange(H, dtype=np.float32)
                zs = np.arange(D, dtype=np.float32)

                # sum over planes
                sum_x = (occ_np.sum(axis=(0,1)) * xs).sum()
                sum_y = (occ_np.sum(axis=(0,2)) * ys).sum()
                sum_z = (occ_np.sum(axis=(1,2)) * zs).sum()

                com_np = np.array([sum_x/mass, sum_y/mass, sum_z/mass], dtype=np.float32)
            else:
                com_np = np.array([W/2.0, H/2.0, D/2.0], dtype=np.float32)  # fallback
        else:
            com_np = None

        # ---- points (variable length) ----
        if self.build_pts and voxel_data.occupied_coords.numel() > 0:
            pts_t = voxel_data.occupied_coords.clone().detach()  # [N,3] torch (CPU)
        else:
            pts_t = torch.zeros((0,3), dtype=torch.float32)

        # ---- pack to torch tensors (CPU) ----
        targets: Dict[str, torch.Tensor] = {
            "labels"   : torch.from_numpy(labels_np.astype(np.int64)),   # Long
            "occ_dense": torch.from_numpy(occ_np),                       # Float
        }
        if occ_low_np is not None:
            targets["occ_low"] = torch.from_numpy(occ_low_np)            # Float
        if dt_np is not None:
            targets["dt"] = torch.from_numpy(dt_np)                      # Float
        if com_np is not None:
            targets["com"] = torch.from_numpy(com_np)                    # Float [3]
        if self.build_pts:
            targets["points"] = pts_t.float()                            # Float [N,3]
        return targets
    






# ============== Main Test Runner ==============
def collate_fn(batch):
    """Custom collate function for batching"""
    voxel_list, target_list = zip(*batch)       
    return list(voxel_list), list(target_list)


def run_experiment(encoder_class, decoder_class, loss_heads, recon_loss, embedding_dim=128, num_epochs=100, batch_size=1, visualize_every=10, size = 48, ckpt_every=100):
    """Run a complete experiment with given encoder/decoder"""
    # Initialize components
    encoder  = encoder_class(voxel_size=size, embedding_dim=embedding_dim)
    decoder  = decoder_class(voxel_size=size, embedding_dim=embedding_dim)
    
        
    # one-step logits function
    logits_fn = lambda z: decoder.decode(z)["logits"]

    # bind it to all heads that need logits
    for h in loss_heads.values():
        if hasattr(h, "bind_logits"):
            h.bind_logits(logits_fn)

    loss_keys = {"total", "reconstruction", *loss_heads.keys()}

    logger = RunLogger(encoder_class.__name__, decoder_class.__name__,
                       loss_keys=loss_keys, ckpt_every=ckpt_every)
    print("▶ logging to:", logger.dir)
    
    trainer = EmbeddingTrainer(encoder, decoder, loss_heads, recon_loss=recon_loss)
    
    # Initialize renderer clients for before/after visualization
    client_input = voxelsim.RendererClient("sampo", 8080, 8081, 8090, 9090)
    client_output = voxelsim.RendererClient("sampo", 8082, 8083, 8090, 9090)  
    client_input.connect_py(0)
    client_output.connect_py(0)
    
    # Set up agents for camera positions
    agent_input = voxelsim.Agent(0)
    agent_output = voxelsim.Agent(0)
    # Position cameras above the voxel volume
    agent_input.set_pos([size/2, size/2, size/2])  # Adjust based on sub_volume_size
    agent_output.set_pos([size/2, size/2, size/2])
    client_input.send_agents_py({0: agent_input})
    client_output.send_agents_py({0: agent_output})



    # Create dataloader
    dataset    = TerrainBatch(world_size=size)
    dataloader = DataLoader(
        dataset,
        batch_size          = batch_size,
        collate_fn          = collate_fn,
        num_workers         = os.cpu_count(),
        pin_memory          = True,
        persistent_workers  = True,
        prefetch_factor     = 1,
    )
    
    # Training loop
    loss_history = defaultdict(list)
    
    steps_per_epoch = 10  # Since we have an infinite dataset
    

    # Store a sample for visualization
    viz_sample = None
    stats = collections.defaultdict(list)
    last_viz_wall_t = time.perf_counter()
    dataloader_iter = iter(dataloader)
    for epoch in range(num_epochs):
        epoch_losses = defaultdict(float)
        
        for step in range(steps_per_epoch):
            with Timer() as t_fetch:
                voxel_batch, target_batch = next(dataloader_iter)
            if viz_sample is None:
                viz_sample = (voxel_batch[0], target_batch[0])
                
            # Move to device
            with Timer() as t_move:
                voxel_batch  = [vd.to_device(trainer.device) for vd in voxel_batch]
                target_batch = [{k: v.to(trainer.device) for k, v in t.items()} 
                                for t in target_batch]
            
            # Train step
            with Timer() as t_model:
                loss_weights = {
                "reconstruction": 1.0,
                "soft_iou": 0.5,
                "tv": 0.05,
                "boundary": 0.5,
                "com": 0.2,
                "class_balance": 0.05,
                "ms_occ": 0.3,
                "chamfer": 0.2,
            }
                losses = trainer.train_step(voxel_batch, target_batch, loss_weights)

            stats["fetch"].append(t_fetch.dt)
            stats["move" ].append(t_move.dt)
            stats["model"].append(t_model.dt)
            

            # Accumulate losses
            for name, value in losses.items():
                epoch_losses[name] += value
        
        # Average losses
        for name in epoch_losses:
            epoch_losses[name] /= steps_per_epoch
            loss_history[name].append(epoch_losses[name])
        f = np.mean(stats["fetch"][-steps_per_epoch:])
        m = np.mean(stats["move" ][-steps_per_epoch:])
        g = np.mean(stats["model"][-steps_per_epoch:])
        if epoch % visualize_every == 0:
            print(f"Epoch {epoch}: {dict(epoch_losses)}")

            
            # --- averaged timings for the last “epoch” (10 steps) ---
 
            print(
                f"Epoch {epoch:04d}  "
                f"| fetch {f*1e3:6.1f} ms   "
                f"move {m*1e3:6.1f} ms   "
                f"model {g*1e3:6.1f} ms   "
                f"total {(f+m+g)*1e3:6.1f} ms"
            )

            # --- wall‑clock since last visualisation ---
            now          = time.perf_counter()
            delta        = now - last_viz_wall_t
            last_viz_wall_t = now
            print(f"⏲  {delta:6.2f} s since previous visualisation\n")


            if viz_sample is not None:
                # Show input (before)
                show_voxels(viz_sample[0], client_input)
                with Timer() as t_disp:
                    # Generate reconstruction (after)
                    with torch.no_grad():
                        viz_data = [viz_sample[0].to_device(trainer.device)]
                        out = encoder.encode(viz_data)
        

                        # handle (latent)   vs   (latent, skips)
                        if isinstance(out, tuple):
                            embedding, skips = out
                        else:
                            embedding, skips = out, None
                        reconstruction = decoder.decode(embedding,skips)
                        logits = reconstruction["logits"]                  # [1,3,D,H,W]
                        
                    
                    # Show output (after)
           
                    show_voxels(logits, client_output)
                    
                print(
                    f"Visualization updated  –  "
                    f"Input:8090 Output:8091   "
                    f"(render {t_disp.dt*1e3:6.1f} ms)"
                )
            # log & checkpoint
        logger.log_epoch(epoch, epoch_losses,
                            {"fetch": f, "move": m, "model": g})
        logger.maybe_ckpt(epoch, encoder, decoder, trainer.optimizer)
    logger.close()
    return loss_history

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter()-self.t0



def show_voxels(sample, client):
    """
    sample: either
      - VoxelData (GT sparse coords), or
      - torch.Tensor logits with shape [B, 3, D, H, W] or [3, D, H, W]
    """
    cell_dict = {}

    # ---------- Ground truth case ----------
    if isinstance(sample, VoxelData):
        coords = sample.occupied_coords.long().cpu().numpy()
        vals   = sample.values.cpu().numpy()  # 0.5 sparse, 1.0 filled
        for (x,y,z), v in zip(coords, vals):
            cell_dict[(int(x), int(y), int(z))] = (
                voxelsim.Cell.filled() if v > 0.9 else voxelsim.Cell.sparse()
            )

    # ---------- Prediction case ----------
    else:
        logits = sample
        if logits.dim() == 5:   # [B,3,D,H,W]
            logits = logits[0]
        # logits now [3,D,H,W]
        pred_class = logits.argmax(0).cpu().numpy()  # 0 empty, 1 sparse, 2 filled

        filled = np.argwhere(pred_class == 2)
        sparse = np.argwhere(pred_class == 1)

        for x,y,z in filled:
            cell_dict[(int(x),int(y),int(z))] = voxelsim.Cell.filled()
        for x,y,z in sparse:
            cell_dict[(int(x),int(y),int(z))] = voxelsim.Cell.sparse()

    world = voxelsim.VoxelGrid.from_dict_py(cell_dict)
    client.send_world_py(world)



class RunLogger:
    """
    Logs any combination of loss heads + timing into CSV and TensorBoard.
    loss_keys : iterable of strings you plan to log, e.g.
                {"total","reconstruction","contour","relative_offset"}
    """

    def __init__(self, enc_name, dec_name, *,
                 loss_keys,                     # ← NEW (set/list)
                 root="runs", ckpt_every=50):
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.dir = os.path.join(root, f"{stamp}-{enc_name}_{dec_name}")
        os.makedirs(self.dir, exist_ok=True) 
        os.makedirs(f"{self.dir}/checkpoints", exist_ok=True)

        # ---- CSV header -------------------------------------------------
        self.loss_keys = list(loss_keys)          # keep a stable order
        header = (["epoch"] +
                  [f"loss_{k}" for k in self.loss_keys] +
                  ["t_fetch_ms", "t_move_ms", "t_model_ms"])
        self.csv_f = open(os.path.join(self.dir, "losses.csv"), "w", newline="")
        self.csv_w = csv.writer(self.csv_f); self.csv_w.writerow(header)

        # ---- TensorBoard ------------------------------------------------
        self.tb = SummaryWriter(self.dir)
        self.tb.flush()
        self.ckpt_every = ckpt_every

    # ---------------------------------------------------------------------
    def log_epoch(self, epoch:int, losses:dict, t:dict):
        row = [epoch] + [losses.get(k, 0.0) for k in self.loss_keys] + [
            t.get("fetch",0)*1e3, t.get("move",0)*1e3, t.get("model",0)*1e3
        ]
        self.csv_w.writerow(row); self.csv_f.flush()

        for k in self.loss_keys:
            self.tb.add_scalar(f"loss/{k}", losses.get(k,0.0), epoch)
        for k,v in t.items():
            self.tb.add_scalar(f"time/{k}", v, epoch)

    # ---------------------------------------------------------------------
    def maybe_ckpt(self, epoch:int, enc, dec, opt):
        if (epoch+1) % self.ckpt_every == 0:
            torch.save({
                "epoch": epoch+1,
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "optim"  : opt.state_dict()
            }, os.path.join(self.dir, "checkpoints",
                            f"epoch-{epoch+1:04d}.pt"))

    def close(self):
        self.csv_f.close(); self.tb.close()

# ============== Usage Example ==============
# if __name__ == "__main__":
#     # Test simple CNN autoencoder
#     print("Testing CNN Autoencoder...")
#     print("CUDA available:", torch.cuda.is_available())
#     print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        
#     loss_heads = {
#         # # function losses wrapped as heads
#         # "soft_iou":      AuxFnHead(make_aux_loss("soft_iou")),
#         # "tv":            AuxFnHead(make_aux_loss("tv")),
#         # "boundary":      AuxFnHead(make_aux_loss("boundary")),
#         # "com":           AuxFnHead(make_aux_loss("com")),
#         # "class_balance": AuxFnHead(make_aux_loss("class_balance")),
#         # "ms_occ":        AuxFnHead(make_aux_loss("ms_occ", scale=4)),
#         # "chamfer":       AuxFnHead(make_aux_loss("chamfer", topk=2048)),
#     }
#     # loss_heads["chamfer"].variable_length_target = True
#     recon_loss = make_recon_loss("ce") 
#     torch.cuda.empty_cache()
#     # dims = [128, 256, 512, 1024]          # sweep list
#     # for d in dims:
#     #     print(f"\n=== 🔵 latent_dim = {d} ===")
#     #     run_experiment(SimpleCNNEncoder, SimpleCNNDecoder,
#     #                    embedding_dim=d,
#     #                    num_epochs=500,            # shorter for quick sweep
#     #                    batch_size=8,
#     #                    visualize_every=50,
#     #                    size=48,
#     #                    ckpt_every=50)
#     run_experiment(SimpleCNNEncoder, SimpleCNNDecoder, loss_heads,
#                        embedding_dim=512,
#                        num_epochs=5000,            # shorter for quick sweep
#                        recon_loss=recon_loss,
#                        batch_size=1,
#                        visualize_every=50,
#                        size=48,
#                        ckpt_every=50)

def make_aux_heads(embedding_dim):
    heads = {
        # "soft_iou":      AuxFnHead(make_aux_loss("soft_iou")), # For now
        # "tv":            AuxFnHead(make_aux_loss("tv")),
        # "boundary":      AuxFnHead(make_aux_loss("boundary")), # For now 
        # "com":           AuxFnHead(make_aux_loss("com")), # For now
        # "class_balance": AuxFnHead(make_aux_loss("class_balance")),
        # "ms_occ":        AuxFnHead(make_aux_loss("ms_occ", scale=4)), # For now
        # "chamfer":       AuxFnHead(make_aux_loss("chamfer", topk=2048)),
        # Add navigation heads with correct dim:
        # "topdown":       TopDownHeightHead(embedding_dim=embedding_dim),
        # "dt":            DistanceTransformHead(embedding_dim=embedding_dim, grid=24),
        # "rel_offset":    RelativeOffsetHead(embedding_dim=embedding_dim, k_nearest=5),
    }
    # heads["chamfer"].variable_length_target = True
    return heads

def sweep():
    recon_loss = make_recon_loss("ce")

    encoder_decoder_pairs = [
        (SimpleCNNEncoder,        SimpleCNNDecoder),
        (PointNetPPLiteFPEncoder,           ImplicitFourierDecoder),
        # (ResNet3DEncoder,         ResNet3DDecoder),
        # (CrossAttnTokensEncoder,  ImplicitFourierDecoder),
        # (PointMLPEncoder,         ImplicitFourierDecoder),   # sparse-in / implicit-out
    ]

    emb_dims = [512, 1024,2048,5120]
    size = 48

     # Regimes are factories now
    regimes = [
        ("recon_only",    lambda dim: {}),
        ("recon_plus_aux", make_aux_heads),   # will be called with dim
    ]

    for regime_name, heads_factory in regimes:
        for Enc, Dec in encoder_decoder_pairs:
            for d in emb_dims:
                loss_heads = heads_factory(d)
                print(f"\n=== {regime_name} | {Enc.__name__} → {Dec.__name__} | dim={d} ===")
                run_experiment(
                    encoder_class=Enc,
                    decoder_class=Dec,
                    loss_heads=loss_heads,
                    recon_loss=recon_loss,
                    embedding_dim=d,
                    num_epochs=5000,
                    batch_size=1,
                    visualize_every=250,
                    size=size,
                    ckpt_every=500,
                )

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    torch.cuda.empty_cache()
    sweep()

