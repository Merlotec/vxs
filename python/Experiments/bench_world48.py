# bench_world48.py
#
# Quick micro‑benchmark:
#   1. build a 48×48×48 world                (t_gen)
#   2. convert the world → VoxelData         (t_conv)
#   3. push it to the renderer via show_voxels (t_show)

import time, numpy as np, torch
import voxelsim                                # your Rust bindings
from representation import VoxelData
import matplotlib.pyplot as plt
from representation import (
    VoxelData, show_voxels,
    SimpleCNNEncoder, SimpleCNNDecoder,
)


class Timer:
    def __enter__(self):  self.t0 = time.perf_counter(); return self
    def __exit__(self,*_): self.dt = time.perf_counter() - self.t0



def build_world(side=48):
    g   = voxelsim.TerrainGenerator()
    cfg = voxelsim.TerrainConfig.default_py()             # enforce 48³
    cfg.set_seed_py(int(np.random.randint(0, 2**31))) # new seed
    cfg.set_world_size_py(int(side))
    g.generate_terrain_py(cfg)
    return g.generate_world_py()

def world_to_voxeldata(world, side=48) -> VoxelData:
    items = world.to_dict_py_tolistpy()         # Vec<((x,y,z), Cell)>
    coords, vals = [], []

    for (x,y,z), cell in items:
        coords.append([x, y, z])
        vals.append(1.0 if cell.is_filled_py() else 0.5)

    return VoxelData(
        occupied_coords=torch.tensor(coords, dtype=torch.float32),
        values=torch.tensor(vals,   dtype=torch.float32),
        bounds=torch.tensor([side, side, side], dtype=torch.float32),
        drone_pos=torch.tensor([side//2]*3,      dtype=torch.float32),
    )

def world_to_voxeldata_np(world, side=48) -> VoxelData:
    coords_np, vals_np = world.as_numpy()       # two NumPy arrays
    coords = torch.from_numpy(coords_np)         # (N,3) float32 – zero copy
    vals   = torch.from_numpy(vals_np)           # (N,)  float32

    return VoxelData(
        occupied_coords=coords,
        values         =vals,
        bounds         =torch.tensor([side, side, side], dtype=torch.float32),
        drone_pos      =torch.tensor([side//2]*3,        dtype=torch.float32),
    )

def test1():
    side = 48
    with Timer() as t_gen:
        world = build_world(side)
    print(f"🌍 generate   {t_gen.dt*1e3:7.2f} ms")


    with Timer() as t_conv_np:
        vd_np = world_to_voxeldata_np(world, side)
    print(f"⚡ convert (numpy) {t_conv_np.dt*1e3:7.2f} ms")

    # ----------------------------------  renderer  ----------------
    client = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client.connect_py(0)

    with Timer() as t_show:
        show_voxels(vd_np, client)
        time.sleep(1)
    print(f"🖼  render     {t_show.dt*1e3:7.2f} ms")
    print("Done – open port 8090 to view.")




test1()