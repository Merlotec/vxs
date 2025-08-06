"""
Octmap Embedding Testbed
A modular framework for testing various embedding approaches for drone navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import voxel_grid, max_pool
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import voxelsim
from collections import defaultdict
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, IterableDataset
import os

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


class LossHead(ABC):
    """Base class for auxiliary loss heads"""
    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


# ============== Sparse CNN Autoencoder ==============
class SparseCNNEncoder(EmbeddingEncoder, nn.Module):
    """Ultra-simple sparse encoder using only MLPs on voxel coordinates"""
    def __init__(self, voxel_size=48, embedding_dim=128):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        
        # Simple MLP that processes each voxel independently
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 64),  # 3 coords + 1 value
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Attention-based aggregation (more efficient than graph convs)
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        
        # Final embedding
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
    
    def encode(self, voxel_batch: List[VoxelData]) -> torch.Tensor:
        batch_embeddings = []
        
        for voxel_data in voxel_batch:
            if voxel_data.occupied_coords.shape[0] == 0:
                batch_embeddings.append(torch.zeros(self.embedding_dim, device=voxel_data.occupied_coords.device))
                continue
            
            # Normalize coordinates to [0, 1]
            coords_norm = voxel_data.occupied_coords.float() / voxel_data.bounds.float()
            
            # Concatenate normalized coords with values
            features = torch.cat([coords_norm, voxel_data.values.unsqueeze(-1)], dim=-1)  # [N, 4]
            
            # Encode each point
            point_features = self.point_encoder(features)  # [N, 128]
            
            # Self-attention to capture relationships
            point_features = point_features.unsqueeze(0)  # [1, N, 128]
            attn_out, _ = self.attention(point_features, point_features, point_features)
            
            # Global pooling
            global_feature = attn_out.mean(dim=1)  # [1, 128]
            
            # Final embedding
            embedding = self.final_mlp(global_feature.squeeze(0))
            batch_embeddings.append(embedding)
        
        return torch.stack(batch_embeddings)
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class SparseCNNDecoder(EmbeddingDecoder, nn.Module):
    def __init__(self, embedding_dim=128, voxel_size=48, max_output_voxels=5000, mem_tokens=8):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        self.num_queries = min(512, max_output_voxels // 4)
        self.mem_tokens = mem_tokens

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, 128))

        self.embedding_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # project embedding into M distinct KV tokens
        self.kv_proj = nn.Linear(128, 128 * self.mem_tokens)

        self.cross_attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)

        self.dec_ffn = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.voxel_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )

    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        B = embedding.shape[0]
        dev = embedding.device

        # Q queries (learned)
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,Q,128]

        # M distinct KV tokens derived from embedding
        emb = self.embedding_mlp(embedding)                                # [B,128]
        kv = self.kv_proj(emb).view(B, self.mem_tokens, 128).contiguous()  # [B,M,128]

        attended, _ = self.cross_attention(q, kv, kv)                      # [B,Q,128]
        x = q + attended
        x = x + self.dec_ffn(x)

        preds = self.voxel_head(x)                                         # [B,Q,5]
        coords = torch.sigmoid(preds[..., :3]) * self.voxel_size           # [B,Q,3]
        logits = preds[..., 3]
        vals   = torch.sigmoid(preds[..., 4])

        return {"sparse_coords": coords, "sparse_logits": logits, "sparse_values": vals}



# ============== Loss Heads ,Currently not in use ==============
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


# ============== Training Framework ==============
class SparseEmbeddingTrainer:
    def __init__(self, encoder: EmbeddingEncoder, decoder: EmbeddingDecoder, 
                 device='cuda', lr=1e-3):
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        if isinstance(encoder, nn.Module):
            encoder.to(self.device)
        if isinstance(decoder, nn.Module):
            decoder.to(self.device)
        
        # Create optimizer
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
    
    def train_step(self, voxel_batch: List[VoxelData]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        # Encode
        embedding = self.encoder.encode(voxel_batch)
        
        # Decode to sparse predictions
        predictions = self.decoder.decode(embedding)
        
        # Compute sparse reconstruction loss
        total_loss = 0
        batch_size = len(voxel_batch)
        
        for i in range(batch_size):
            # Get predictions for this sample
            pred_coords = predictions["sparse_coords"][i]  # [Q, 3]
            pred_logits = predictions["sparse_logits"][i]  # [Q]
            pred_values = predictions["sparse_values"][i]  # [Q]
            
            # Get ground truth
            gt_coords = voxel_batch[i].occupied_coords.to(self.device)  # [N, 3]
            gt_values = voxel_batch[i].values.to(self.device)  # [N]
            
            # Compute losses
            loss = self.sparse_chamfer_loss(
                pred_coords, pred_logits, pred_values,
                gt_coords, gt_values
            )
            total_loss += loss
        
        total_loss = total_loss / batch_size
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        return {"total": total_loss.item()}
    
    def sparse_chamfer_loss(
    self,
    pred_coords: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_values: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_values: torch.Tensor,
    threshold: float = 0.5,
    chunk: int = 4096,
):
        """Fixed loss that always trains spatial coordinates"""
        
        probs = torch.sigmoid(pred_logits)
        
        # Occupancy regularization
        target_occ = (torch.arange(pred_logits.numel(), device=pred_logits.device)
                    < gt_coords.shape[0]).float()
        occ_loss = F.binary_cross_entropy_with_logits(pred_logits, target_occ)
        
        if gt_coords.numel() == 0:
            return occ_loss
        
        # CRITICAL FIX: Always compute spatial loss, weighted by probability
        # This ensures coordinates learn even when occupancies are low
        
        # Use soft weighting instead of hard thresholding
        weights = torch.clamp(probs, min=0.01)  # Minimum weight to keep gradients
        
        # Weighted chamfer distance
        def weighted_min_dist(A, B, W, chunk_sz):
            """Find weighted minimum distance from A to B"""
            P = A.size(0)
            min_d = torch.full((P,), float('inf'), device=A.device, dtype=A.dtype)
            min_j = torch.zeros((P,), dtype=torch.long, device=A.device)
            
            for s in range(0, B.size(0), chunk_sz):
                e = min(s + chunk_sz, B.size(0))
                d = torch.cdist(A, B[s:e], p=2)
                d_chunk, j = d.min(dim=1)
                better = d_chunk < min_d
                min_d = torch.where(better, d_chunk, min_d)
                min_j = torch.where(better, j + s, min_j)
            
            # Weight distances by probability
            weighted_d = min_d * W
            return weighted_d, min_j
        
        # Forward: all predictions to GT, weighted by their probability
        d_fg, gt_idx = weighted_min_dist(pred_coords, gt_coords, weights, chunk)
        forward_loss = d_fg.mean()
        
        # Backward: GT to predictions (find best prediction for each GT)
        # Weight prediction targets by their probability
        expanded_weights = weights.unsqueeze(0).expand(gt_coords.size(0), -1)
        dist_matrix_chunks = []
        
        for s in range(0, pred_coords.size(0), chunk):
            e = min(s + chunk, pred_coords.size(0))
            d_chunk = torch.cdist(gt_coords, pred_coords[s:e], p=2)
            # Weight by probability of each prediction
            d_chunk = d_chunk / (expanded_weights[:, s:e] + 1e-6)
            dist_matrix_chunks.append(d_chunk)
        
        if dist_matrix_chunks:
            weighted_dists = torch.cat(dist_matrix_chunks, dim=1)
            d_gf, pred_idx = weighted_dists.min(dim=1)
            backward_loss = d_gf.mean()
        else:
            backward_loss = torch.tensor(0.0, device=pred_coords.device)
        
        # Value loss - weighted by probability and distance
        close = d_fg < 2.0
        if close.any():
            matched_gt_vals = gt_values[gt_idx[close]]
            matched_pred_vals = pred_values[close]
            # Weight by probability of being occupied
            value_weights = weights[close]
            value_loss = (F.mse_loss(matched_pred_vals, matched_gt_vals, reduction='none') * value_weights).mean()
        else:
            value_loss = torch.tensor(0.0, device=pred_coords.device)
        
        # Balance losses carefully
        total = (
            forward_loss * 1.0 +      # Spatial accuracy
            backward_loss * 1.0 +     # Coverage
            value_loss * 0.5 +        # Value prediction
            occ_loss * 0.2            # Occupancy (reduced weight)
        )
        
        return total
        

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
            yield voxel_data, {}          # empty target dict for now
            
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
    


# ============== Visualization Functions ==============
def show_voxels(voxel_data: VoxelData, client) -> None:
    """Show ground truth voxels"""
    cell_dict = {}
    
    coords = voxel_data.occupied_coords.long().cpu().numpy()
    values = voxel_data.values.cpu().numpy()
    
    for i in range(coords.shape[0]):
        x, y, z = coords[i]
        if values[i] > 0.8:
            cell_dict[(int(x), int(y), int(z))] = voxelsim.Cell.filled()
        else:
            cell_dict[(int(x), int(y), int(z))] = voxelsim.Cell.sparse()
    
    world = voxelsim.VoxelGrid.from_dict_py(cell_dict)
    client.send_world_py(world)


def show_sparse_voxels(predictions: Dict[str, torch.Tensor], client, 
                       threshold=0.3, voxel_size=48):
    """Visualize sparse predictions"""
    cell_dict = {}
    
    # Get first batch item
    coords = predictions["sparse_coords"][0].cpu().numpy()
    logits = predictions["sparse_logits"][0].cpu().numpy()
    values = predictions["sparse_values"][0].cpu().numpy()
    
    # Filter by occupancy probability
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    occupied = probs > threshold
    
    coords = coords[occupied]
    values = values[occupied]
    
    # Round to integer positions
    coords = np.round(coords).astype(int)
    
    # Add to visualization
    for i in range(coords.shape[0]):
        x, y, z = coords[i]
        if 0 <= x < voxel_size and 0 <= y < voxel_size and 0 <= z < voxel_size:
            if values[i] > 0.8:
                cell_dict[(int(x), int(y), int(z))] = voxelsim.Cell.filled()
            else:
                cell_dict[(int(x), int(y), int(z))] = voxelsim.Cell.sparse()
    
    world = voxelsim.VoxelGrid.from_dict_py(cell_dict)
    client.send_world_py(world)


# ============== Main Test Runner ==============
def collate_fn(batch):
    """Custom collate function for batching"""
    voxel_list, target_list = zip(*batch)       
    return list(voxel_list), list(target_list)


def run_sparse_experiment(num_epochs=100, batch_size=1, visualize_every=10, size=48):
    """Run experiment with fully sparse architecture"""
    
    # Initialize sparse components
    encoder = SparseCNNEncoder(voxel_size=size)
    decoder = SparseCNNDecoder(voxel_size=size, max_output_voxels=5000)
    
    trainer = SparseEmbeddingTrainer(encoder, decoder)
    
    # Initialize renderer clients
    client_input = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client_output = voxelsim.RendererClient("127.0.0.1", 8082, 8083, 8090, 9090)
    client_input.connect_py(0)
    client_output.connect_py(0)
    
    # Create dataloader
    dataset = TerrainBatch(world_size=size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )
    
    # Training loop
    dataloader_iter = iter(dataloader)
    viz_sample = None
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        steps_per_epoch = 10
        
        for step in range(steps_per_epoch):
            voxel_batch, _ = next(dataloader_iter)
            
            if viz_sample is None:
                viz_sample = voxel_batch[0]
            
            # Move to device
            voxel_batch = [vd.to_device(trainer.device) for vd in voxel_batch]
            
            # Train step (no target needed - using sparse chamfer loss)
            losses = trainer.train_step(voxel_batch)
            epoch_loss += losses["total"]
        
        epoch_loss /= steps_per_epoch
        
        if epoch % visualize_every == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
            
            if viz_sample is not None:
                # Show input
                show_voxels(viz_sample, client_input)
                
                # Generate and show sparse reconstruction
                with torch.no_grad():
                    viz_data = [viz_sample.to_device(trainer.device)]
                    embedding = encoder.encode(viz_data)
                    predictions = decoder.decode(embedding)
                    show_sparse_voxels(predictions, client_output)
                
                print(f"Visualization updated - Input:8090 Output:8091")
    
    return {"loss": epoch_loss}

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter()-self.t0






# ============== Usage Example ==============
if __name__ == "__main__":
    # Test simple CNN autoencoder
    print("Testing CNN Autoencoder...")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    results = run_sparse_experiment(num_epochs=1000, batch_size=1, visualize_every=10, size=48) 

