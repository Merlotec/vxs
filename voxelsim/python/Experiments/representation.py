"""
Octmap Embedding Testbed
A modular framework for testing various embedding approaches for drone navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import voxelsim
from collections import defaultdict
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, IterableDataset


# ============== Data Structures ==============
@dataclass
class VoxelData:
    """Container for voxel octmap data"""
    occupied_coords: torch.Tensor  # [N, 3] coordinates of occupied voxels
    values: torch.Tensor           # [N] values (filled, sparse, etc)``
    bounds: torch.Tensor           # [3] max bounds of the octmap TODO: Not sure if we can have this, it might have to infer itself
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
        
    def encode(self, voxel_data: VoxelData) -> torch.Tensor:
        # Convert sparse voxel data to dense grid
        grid = self._sparse_to_dense(voxel_data)
        
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def _sparse_to_dense(self, voxel_data: VoxelData) -> torch.Tensor:
        """Convert sparse voxel coordinates to dense grid"""
        batch_size = 1  # For now, single sample
        grid = torch.zeros((batch_size, 1, self.voxel_size, self.voxel_size, self.voxel_size),
                          device=voxel_data.occupied_coords.device, dtype=torch.float32)
        
        if voxel_data.occupied_coords.shape[0] > 0:
            # Normalize coordinates to grid size
            coords = voxel_data.occupied_coords.float()
            coords = coords / voxel_data.bounds.float() * (self.voxel_size - 1)
            coords = coords.long().clamp(0, self.voxel_size - 1)
            
            # Fill grid
            grid[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = voxel_data.values
     

        return grid.to(next(self.parameters()).device)
    
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
        self.deconv3 = nn.ConvTranspose3d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        
    def decode(self, embedding: torch.Tensor, query_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.fc(embedding)
        x = x.view(-1, 128, self.init_size, self.init_size, self.init_size)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        
        return {"reconstruction": x}


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


# ============== Training Framework ==============
class EmbeddingTrainer:
    def __init__(self, encoder: EmbeddingEncoder, decoder: EmbeddingDecoder, 
                 loss_heads: Dict[str, LossHead], device='cuda', lr=1e-3):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_heads = loss_heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Collect all parameters
        all_params = []
        
        # Move models to device and collect parameters TODO: Not too sure what this does, I think it moves# something to GPU if possible
        if isinstance(encoder, nn.Module):
            encoder.to(self.device)
            all_params.extend(encoder.parameters())
        if isinstance(decoder, nn.Module):
            decoder.to(self.device)
            all_params.extend(decoder.parameters())
        for head in loss_heads.values():
            if isinstance(head, nn.Module):
                head.to(self.device)
                all_params.extend(head.parameters())
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
    
    def train_step(self, voxel_data: VoxelData, targets: Dict[str, torch.Tensor], 
                   loss_weights: Dict[str, float]) -> Dict[str, float]:
        """Single training step"""
        # Zeros stored gradients
        self.optimizer.zero_grad()
        
        # Encode
        embedding = self.encoder.encode(voxel_data)
        
        # Decode for reconstruction loss
        reconstruction = self.decoder.decode(embedding)
        
        # Compute losses
        losses = {}
        
        # Reconstruction loss
        if "reconstruction" in loss_weights:
            recon_target = self._create_reconstruction_target(voxel_data)
            losses["reconstruction"] = F.mse_loss(reconstruction["reconstruction"], recon_target)
        
        # Auxiliary head losses
        for name, head in self.loss_heads.items():
            if name in targets and name in loss_weights:
                prediction = head.forward(embedding)
                losses[name] = head.compute_loss(prediction, targets[name])
        
        # Total loss
        total_loss = sum(loss_weights.get(name, 0) * loss for name, loss in losses.items())
        losses["total"] = total_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def _create_reconstruction_target(self, voxel_data: VoxelData) -> torch.Tensor:
        """Create dense reconstruction target from sparse voxel data"""
        voxel_size = self.encoder.voxel_size if hasattr(self.encoder, 'voxel_size') else 64
        batch_size = 1
        
        grid = torch.zeros((batch_size, 1, voxel_size, voxel_size, voxel_size),
                  device=self.device, requires_grad=False)
        
        if voxel_data.occupied_coords.shape[0] > 0:
            # Normalize coordinates to grid size
            coords = voxel_data.occupied_coords.float()
            coords = coords / voxel_data.bounds.float() * (voxel_size - 1)
            coords = coords.long().clamp(0, voxel_size - 1)
            
            # Fill grid
            grid[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = voxel_data.values
        
        return grid


# ============== Data Generation ==============
class TerrainBatch(IterableDataset):
    def __init__(self, world_size=100, sub_volume_size=64):
        self.world_size = world_size
        self.sub_volume_size = sub_volume_size
        
    def generate_terrain_sample(self) -> Tuple[VoxelData, Dict[str, torch.Tensor]]:
        """Generate a terrain sample with ground truth targets"""
        # Create world
        generator = voxelsim.TerrainGenerator()
        generator.generate_terrain_py(voxelsim.TerrainConfig.default_py())
        world = generator.generate_world_py()
        
        # Extract random sub-volume
        center = np.random.randint(20, self.world_size - 20, size=3)
        voxel_data, targets = self._extract_subvolume(world, center)
        
        return voxel_data, targets
    
    def _extract_subvolume(self, world: voxelsim.VoxelGrid, center: np.ndarray) -> Tuple[VoxelData, Dict[str, torch.Tensor]]:
        """Extract sub-volume around center point"""
        # Get voxels in sub-volume, later on we should switch to using Minkowski dense tensor for much better performance
        occupied_coords = []
        values = []
        
        half_size = self.sub_volume_size // 2
        for x in range(center[0] - half_size, center[0] + half_size):
            for y in range(center[1] - half_size, center[1] + half_size):
                for z in range(center[2] - half_size, center[2] + half_size):
                    cell = world.get_cell(x, y, z) 
                    if cell is not None:
                        occupied_coords.append([x - center[0] + half_size, 
                                              y - center[1] + half_size, 
                                              z - center[2] + half_size])
                        values.append(1.0)  # Simplified for now
        if len(occupied_coords) == 0:
            values.append(0.0)
            return self.generate_terrain_sample()
            
        voxel_data = VoxelData(
            occupied_coords=torch.tensor(occupied_coords, dtype=torch.float32),
            values=torch.tensor(values, dtype=torch.float32),
            bounds=torch.tensor([self.sub_volume_size] * 3, dtype=torch.float32),
            drone_pos=torch.tensor([half_size, half_size, half_size], dtype=torch.float32)
        )
        
        # Generate targets
        targets = self._generate_targets(voxel_data)
        
        
        return voxel_data, targets
    
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
    
    def __iter__(self):
        while True:
            yield self.generate_terrain_sample()


# ============== Main Test Runner ==============
def collate_fn(batch):
    """Custom collate function for batching"""
    # For now, just return single samples as-is
    # In future, implement proper batching
    return batch[0]


def run_experiment(encoder_class, decoder_class, num_epochs=10, batch_size=1, visualize_every=10):
    """Run a complete experiment with given encoder/decoder"""
    # Initialize components
    encoder = encoder_class()
    decoder = decoder_class()
    
    loss_heads = {
        "contour": ContourHead(),
        "relative_offset": RelativeOffsetHead(),
        # "cover_mask": CoverMaskHead()  # Disabled until we have semantic labels
    }
    
    trainer = EmbeddingTrainer(encoder, decoder, loss_heads)
    
    # Initialize renderer clients for before/after visualization
    client_input = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client_output = voxelsim.RendererClient("127.0.0.1", 8082, 8083, 8090, 9090)  
    client_input.connect_py(0)
    client_output.connect_py(0)
    
    # Set up agents for camera positions
    agent_input = voxelsim.Agent(0)
    agent_output = voxelsim.Agent(0)
    # Position cameras above the voxel volume
    agent_input.set_pos([24.0, 40.0, 24.0])  # Adjust based on sub_volume_size
    agent_output.set_pos([24.0, 40.0, 24.0])
    client_input.send_agents_py({0: agent_input})
    client_output.send_agents_py({0: agent_output})



    # Create dataloader
    dataset = TerrainBatch(world_size=100, sub_volume_size=48)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)  # batch_size=1 for now
    
    # Training loop
    loss_history = defaultdict(list)
    
    steps_per_epoch = 10  # Since we have an infinite dataset

    # Store a sample for visualization
    viz_sample = None
    
    for epoch in range(num_epochs):
        epoch_losses = defaultdict(float)
        
        for step, (voxel_data, targets) in enumerate(dataloader):
            if step >= steps_per_epoch:
                break

            if viz_sample is None:
                viz_sample = (voxel_data, targets)
                
            # Move to device
            voxel_data = voxel_data.to_device(trainer.device)
            targets = {k: v.to(trainer.device) for k, v in targets.items()}
            
            # Train step
            loss_weights = {
                "reconstruction": 1.0,
                "contour": 0.5,
                "relative_offset": 0.3,
            }
            
            losses = trainer.train_step(voxel_data, targets, loss_weights)
            
            # Accumulate losses
            for name, value in losses.items():
                epoch_losses[name] += value
        
        # Average losses
        for name in epoch_losses:
            epoch_losses[name] /= steps_per_epoch
            loss_history[name].append(epoch_losses[name])
        
        if epoch % visualize_every == 0:
            print(f"Epoch {epoch}: {dict(epoch_losses)}")
            
            if viz_sample is not None:
                # Show input (before)
                dataset.show_voxels(viz_sample[0], client_input)
                
                # Generate reconstruction (after)
                with torch.no_grad():
                    viz_data = viz_sample[0].to_device(trainer.device)
                    embedding = encoder.encode(viz_data)
                    reconstruction = decoder.decode(embedding)["reconstruction"]
                    
                # Show output (after)
                dataset.show_voxels(reconstruction[0], client_output)
                
                print(f"Visualization updated - Input on port 8090, Output on port 8091")
    
    return loss_history


def show_voxels(voxel_data: Union[VoxelData, torch.Tensor], 
                client: voxelsim.RendererClient) -> None:
    """Send voxel data to the renderer"""
    # Create empty world
    world = voxelsim.VoxelGrid.new()  # This should work based on the Rust code
    
    if hasattr(voxel_data, 'occupied_coords'):
        coords = voxel_data.occupied_coords.long().cpu().numpy()
        for i in range(coords.shape[0]):
            world.set_cell(int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]), 
                          voxelsim.CellType.FilledDirt)
    else:
        dense = voxel_data
        if dense.dim() == 4:
            dense = dense[0]
        
        voxel_array = dense.cpu().numpy()
        occupied = np.where(voxel_array > 0.5)
        
        for i in range(len(occupied[0])):
            world.set_cell(int(occupied[0][i]), int(occupied[1][i]), int(occupied[2][i]), 
                          voxelsim.CellType.FilledDirt)
    
    client.send_world_py(world)

# ============== Usage Example ==============
if __name__ == "__main__":
    # Test simple CNN autoencoder
    print("Testing CNN Autoencoder...")
    results = run_experiment(SimpleCNNEncoder, SimpleCNNDecoder, num_epochs=50, visualize_every=10)
    
    # Future: Add transformer, GNN, and other architectures
