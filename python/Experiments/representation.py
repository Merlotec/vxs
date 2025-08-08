"""
Octmap Embedding Testbed
A modular framework for testing various embedding approaches for drone navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv, datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import voxelsim
from collections import defaultdict
from losses import make_recon_loss
from torch.utils.data import DataLoader, IterableDataset
import os
from scipy.spatial import KDTree


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

# ============== Loss Heads ==============
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

            weights = torch.tensor([0.2, 1.0, 1.5], device=logits.device)

            losses["reconstruction"] = self.recon_loss_fn(
            logits, recon_targets,
            extra={"class_weights": weights}
        )
            
        # Auxiliary head losses, currently not in use
        # inside EmbeddingTrainer.train_step, for aux heads loop
        for name, head in self.loss_heads.items():
            if name not in loss_weights: 
                continue
            if not all(name in tb for tb in target_batch):
                continue

            # reuse decoder output once
            preds = head(embedding, logits=logits)

            if getattr(head, "variable_length_target", False):
                # e.g. Chamfer/RDP lists of points per sample
                t_for_head = [tb[name] for tb in target_batch]   # list (no stack)
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
    def __init__(self, world_size: int = 120):
        self.world_size = int(world_size)

    # ---------- helper: fast Rust → NumPy → Torch ------------------------
    @staticmethod
    def world_to_voxeldata_np(world: voxelsim.VoxelGrid, side: int) -> VoxelData:
        coords_np, vals_np = world.as_numpy()                 # (N,3) in (x,z,y)
        coords = torch.from_numpy(coords_np)  
        vals   = torch.from_numpy(vals_np)

        return VoxelData(
            occupied_coords = coords,
            values          = vals,
            bounds          = torch.tensor([side]*3, dtype=torch.float32),
            drone_pos       = torch.tensor([side//2]*3, dtype=torch.float32),
        )

    # ---------- iterator -------------------------------------------------
    def __iter__(self):
        side = self.world_size
        while True:
            # build world in Rust
            g   = voxelsim.TerrainGenerator()
            cfg = voxelsim.TerrainConfig.default_py()
            cfg.set_seed_py(int(np.random.randint(0, 2**31)))
            cfg.set_world_size_py(side)
            g.generate_terrain_py(cfg)
            world = g.generate_world_py()

            voxel_data = self.world_to_voxeldata_np(world, side)
            targets = self._generate_targets(voxel_data)
            yield voxel_data, targets          # empty target dict for now
            
    def _generate_targets(self, voxel_data: VoxelData) -> Dict[str, torch.Tensor]:
        """Generate ground truth targets for loss heads"""
        targets = {}
        
        # Contour map (max height projection)
        if voxel_data.occupied_coords.shape[0] > 0:
            coords = voxel_data.occupied_coords
            map_size = 32
            contour = torch.zeros(map_size, map_size)
            
            # Project to 2D and find max heights
            coords_2d = (coords[:, [0, 2]] / voxel_data.bounds[[0, 2]] * (map_size - 1)).long()
            for i, (x, z) in enumerate(coords_2d):
                if 0 <= x < map_size and 0 <= z < map_size:
                    contour[x, z] = max(contour[x, z], coords[i, 1])
            
            targets["contour"] = contour / voxel_data.bounds[1]  # Normalize
            
            # Relative offsets to K nearest obstacles
            k_nearest = 5
            drone_pos_np = voxel_data.drone_pos.cpu().numpy()
            coords_np = voxel_data.occupied_coords.cpu().numpy()
            
            if coords_np.shape[0] >= k_nearest:
                # Build KDTree for efficient nearest neighbor search
                tree = KDTree(coords_np)
                distances, indices = tree.query(drone_pos_np, k=min(k_nearest, coords_np.shape[0]))
                indices = np.atleast_1d(indices)
                
                # Compute relative offsets
                nearest_coords = coords_np[indices]
                offsets = nearest_coords - drone_pos_np
                targets["relative_offset"] = torch.tensor(offsets, dtype=torch.float32)
            else:
                # Not enough points, pad with zeros
                targets["relative_offset"] = torch.zeros(k_nearest, 3)
        
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
    client_input = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client_output = voxelsim.RendererClient("127.0.0.1", 8082, 8083, 8090, 9090)  
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
                    "contour": 0.5,
                    "relative_offset": 0.3,
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
        
        if epoch % visualize_every == 0:
            print(f"Epoch {epoch}: {dict(epoch_losses)}")

            
            # --- averaged timings for the last “epoch” (10 steps) ---
            f = np.mean(stats["fetch"][-steps_per_epoch:])
            m = np.mean(stats["move" ][-steps_per_epoch:])
            g = np.mean(stats["model"][-steps_per_epoch:])
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
                # show_voxels(viz_sample[0], client_input)
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
           
                    # show_voxels(logits, client_output)
                    
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






# def show_voxels(voxel_data: Union[VoxelData, torch.Tensor], 
#                 client: voxelsim.RendererClient) -> None:
#     """Send voxel data to the renderer"""
    
#     cell_dict = {}
    
#     if hasattr(voxel_data, 'occupied_coords'):
#         coords = voxel_data.occupied_coords.long().cpu().numpy()
#         values = voxel_data.values.cpu().numpy()
        
#         for i in range(coords.shape[0]):
#             coord_tuple = (int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))
#             # Use values to determine cell type
#             if values[i] > 0.8:
#                 cell_dict[coord_tuple] = voxelsim.Cell.filled()
#             else:
#                 cell_dict[coord_tuple] = voxelsim.Cell.sparse()
#     else:
#         # Handle dense tensor
#         dense = voxel_data
#         if dense.dim() == 5:  # [B, C, D, H, W]
#             dense = dense[0, 0]
#         elif dense.dim() == 4:  # [C, D, H, W]
#             dense = dense[0]
        
#         voxel_array = dense.cpu().numpy()
#         occupied = np.where(voxel_array > 0.5)
        
#         for i in range(len(occupied[0])):
#             coord_tuple = (int(occupied[0][i]), int(occupied[1][i]), int(occupied[2][i]))
#             value = voxel_array[occupied[0][i], occupied[1][i], occupied[2][i]]
            
#             if value > 0.8:
#                 cell_dict[coord_tuple] = voxelsim.Cell.filled()
#             else:
#                 cell_dict[coord_tuple] = voxelsim.Cell.sparse()
    
#     # Create world from dictionary
#     world = voxelsim.VoxelGrid.from_dict_py(cell_dict)
#     client.send_world_py(world)


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
if __name__ == "__main__":
    # Test simple CNN autoencoder
    print("Testing CNN Autoencoder...")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    loss_heads = {
        # "contour": ContourHead(),
        # "relative_offset": RelativeOffsetHead(),
        # "cover_mask": CoverMaskHead()  # Disabled until we have semantic labels
    }
    recon_loss = make_recon_loss("ce") 
    torch.cuda.empty_cache()
    # dims = [128, 256, 512, 1024]          # sweep list
    # for d in dims:
    #     print(f"\n=== 🔵 latent_dim = {d} ===")
    #     run_experiment(SimpleCNNEncoder, SimpleCNNDecoder,
    #                    embedding_dim=d,
    #                    num_epochs=500,            # shorter for quick sweep
    #                    batch_size=8,
    #                    visualize_every=50,
    #                    size=48,
    #                    ckpt_every=50)
    run_experiment(SimpleCNNEncoder, SimpleCNNDecoder, loss_heads,
                       embedding_dim=512,
                       num_epochs=500,            # shorter for quick sweep
                       recon_loss=recon_loss,
                       batch_size=1,
                       visualize_every=10,
                       size=48,
                       ckpt_every=50)



    # Simple test: Generate terrain, extract subvolume, and visualize



    # print("Testing terrain generation and visualization...")
    
    # # Initialize renderer client
    # client = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    # client.connect_py(0)
    
    # # Set up camera
    # agent = voxelsim.Agent(0)
    # agent.set_pos([50, 40.0, 50])  # Position above center of 48x48x48 volume
    # client.send_agents_py({0: agent})
    
    # # Generate terrain sample
    # dataset = TerrainBatch(world_size=100, sub_volume_size=100)
    # voxel_data, targets = dataset.generate_terrain_sample()
    
    # print(f"Generated voxel data with {len(voxel_data.occupied_coords)} voxels")
    # print(f"Bounds: {voxel_data.bounds}")
    # print(f"Drone position: {voxel_data.drone_pos}")
    
    # # Visualize the extracted subvolume
    # show_voxels(voxel_data, client)
    # print("Visualization sent to renderer on port 8090")
    
    # # Keep the program running to view the visualization
    # input("Press Enter to exit...")
    
    
