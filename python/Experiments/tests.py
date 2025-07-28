
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
        # Get all cells from the world as a dictionary
        world_cells = world.to_dict_py()
        
        # Extract cells in sub-volume
        occupied_coords = []
        values = []
        
        half_size = self.sub_volume_size // 2
        
        # Iterate through world cells and extract those in our sub-volume
        for (x, y, z), cell in world_cells.items():
            # Check if this cell is within our sub-volume
            if (center[0] - half_size <= x < center[0] + half_size and
                center[1] - half_size <= y < center[1] + half_size and
                center[2] - half_size <= z < center[2] + half_size):
                
                # Convert to sub-volume relative coordinates
                rel_x = x - center[0] + half_size
                rel_y = y - center[1] + half_size
                rel_z = z - center[2] + half_size
                
                occupied_coords.append([rel_x, rel_y, rel_z])
                
                # Determine value based on cell type
                if cell == voxelsim.Cell.filled():
                    values.append(1.0)
                elif cell == voxelsim.Cell.sparse():
                    values.append(0.5)
                else:
                    values.append(1.0)  # Default
        
        if len(occupied_coords) == 0:
            # No voxels in this sub-volume, try again
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




def show_voxels(voxel_data: Union[VoxelData, torch.Tensor], 
                client: voxelsim.RendererClient) -> None:
    """Send voxel data to the renderer"""
    # Create empty world
    world = voxelsim.VoxelGrid()  # This should work based on the Rust code
    
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
