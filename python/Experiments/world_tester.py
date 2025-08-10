
# bench_world48.py
#
# Quick micro‑benchmark:
#   1. build a 48×48×48 world                (t_gen)
#   2. convert the world → VoxelData         (t_conv)
#   3. push it to the renderer via show_voxels (t_show)

import time, numpy as np, torch
import voxelsim                                # your Rust bindings
from representation import VoxelData, show_voxels   # reuse helpers
import matplotlib.pyplot as plt


# --- add these near the top, after your imports ---
from representation import (
    # pick the pair you trained; these defaults match your sweep example
    CrossAttnTokensEncoder, Factorised3DDecoder,
    # if you used others, import them instead:
    # SimpleCNNEncoder, SimpleCNNDecoder,
    # UNet3DEncoder, UNet3DDecoder,
    # ResNet3DEncoder, ResNet3DDecoder,
    # PointMLPEncoder, ImplicitFourierDecoder,
)

CKPT_PATH = "/home/box/drone/vxs/runs/2025-08-10T15-33-27-CrossAttnTokensEncoder_Factorised3DDecoder/checkpoints/epoch-5000.pt"
EncClass  = CrossAttnTokensEncoder          # <-- swap if needed
DecClass  = Factorised3DDecoder                   # <-- swap if needed

def _infer_emb_dim(enc_sd, dec_sd):
    for k in ("fc.weight", "proj.weight", "head.weight"):
        if k in enc_sd:
            return enc_sd[k].shape[0]
    if "fc.weight" in dec_sd:                     # decoder fc [out, E]
        return dec_sd["fc.weight"].shape[1]
    return 512  # fallback

def _load_models(ckpt_path, world_side, model_side, device):
    with Timer() as t_load:
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        enc_sd, dec_sd = ckpt["encoder"], ckpt["decoder"]
        E = _infer_emb_dim(enc_sd, dec_sd)
        enc = EncClass(voxel_size=world_side, embedding_dim=E).to(device).eval()
        dec = DecClass(voxel_size=model_side, embedding_dim=E).to(device).eval()
        enc.load_state_dict(enc_sd); dec.load_state_dict(dec_sd)
    return enc, dec, t_load.dt





class Timer:
    def __enter__(self):  self.t0 = time.perf_counter(); return self
    def __exit__(self,*_): self.dt = time.perf_counter() - self.t0

def build_world(side=48):
    g   = voxelsim.TerrainGenerator()
    cfg = voxelsim.TerrainConfig.default_py()             # enforce 48³
    # cfg.set_seed_py(int(np.random.randint(0, 2**31))) # new seed
    cfg.set_seed_py(42)
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

# --- inside your test1() (minimal edits) ---
def test1():
    world_side = 48
    model_side = 48
    with Timer() as t_gen:
        world = build_world(world_side)
    print(f"🌍 generate          {t_gen.dt*1e3:7.2f} ms")

    with Timer() as t_conv_np:
        vd_np = world_to_voxeldata_np(world, world_side)
    print(f"⚡ convert (numpy)   {t_conv_np.dt*1e3:7.2f} ms")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # two viewers: input on 8090, output on 8091
    client_in  = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client_out = voxelsim.RendererClient("127.0.0.1", 8082, 8083, 8090, 9090)

    client_in.connect_py(0); client_out.connect_py(0)

    # show input
    with Timer() as t_show_in:
        show_voxels(vd_np, client_in)
    print(f"🖼  render (input)    {t_show_in.dt*1e3:7.2f} ms")

    # load -> embed -> reconstruct+display (profiled)
    enc, dec, t_load = _load_models(CKPT_PATH, model_side, world_side, device)

    with Timer() as t_embed:
        with torch.no_grad():
            latent = enc.encode([vd_np.to_device(device)])
            if isinstance(latent, tuple):         # handle (latent, skips)
                latent = latent[0]

    with Timer() as t_recon_disp:
        with torch.no_grad():
            out = dec.decode(latent)
            logits = out["logits"]
        show_voxels(logits, client_out)
    

    print(f"📦 load ckpt         {t_load*1e3:7.2f} ms")
    print(f"🧠 embed (encode)    {t_embed.dt*1e3:7.2f} ms")
    print(f"🎯 recon+display     {t_recon_disp.dt*1e3:7.2f} ms")
    print("Open 8090 (input) and 8091 (output). Done.")
    while True:
        continue
    
    
test1()