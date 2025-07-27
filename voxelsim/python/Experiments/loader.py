# Here we are experimenting with ways to rapidly load and manipulate voxel data in Python.

# ============== Data Structures ==============
@dataclass
class VoxelData:
    """Container for voxel octmap data"""
    occupied_coords: torch.Tensor  # [N, 3] coordinates of occupied voxels
    values: torch.Tensor           # [N] values (filled, sparse, etc)
    bounds: torch.Tensor           # [3] max bounds of the octmap TODO: This is in a sense a field of perception for the mac bounds that we waznt the map to proccess, on the rust side we might have to make this within a certain range
    drone_pos: torch.Tensor        # [3] current drone position
    
    def to_device(self, device):
        return VoxelData(
            occupied_coords=self.occupied_coords.to(device),
            values=self.values.to(device),
            bounds=self.bounds.to(device),
            drone_pos=self.drone_pos.to(device)
        )
    

class TerrainBatch(IterableDataset):
    def __init__(self, world_size=100, sub_volume_size=64):
        self.world_size = world_size
        self.sub_volume_size = sub_volume_size
        
    def generate_terrain_sample(self) -> Tuple[VoxelData, Dict[str, torch.Tensor]]:
        """Generate a terrain sample with ground truth targets"""
        # Create world
        world = voxelsim.VoxelGrid()
        world.generate_default_terrain(np.random.randint(1000))
        
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
            return self.generate_terrain_sample()
            values.append(0.0)
        voxel_data = VoxelData(
            occupied_coords=torch.tensor(occupied_coords, dtype=torch.float32),
            values=torch.tensor(values, dtype=torch.float32),
            bounds=torch.tensor([self.sub_volume_size] * 3, dtype=torch.float32),
            drone_pos=torch.tensor([half_size, half_size, half_size], dtype=torch.float32)
        )
        
        # Generate targets
        targets = self._generate_targets(voxel_data)
        
        
        return voxel_data, targets
        
    def __iter__(self):
        while True:
            yield self.generate_terrain_sample()
