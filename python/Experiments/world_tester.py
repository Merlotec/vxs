# bench_world48.py
#
# Quick micro-benchmark:
#   1. build a 48×48×48 world                (t_gen)
#   2. convert the world → VoxelData         (t_conv)
#   3. push it to the renderer via show_voxels (t_show)

import time, numpy as np, torch, sys, select, tty, termios
import voxelsim                                # your Rust bindings
import matplotlib.pyplot as plt

# use the same datastructures + helpers from representation_parallel
from representation import (
    VoxelData, show_voxels, TerrainBatch,
    SimpleCNNEncoder, SimpleCNNDecoder,
)

CKPT_PATH = "/home/box/drone/vxs/runs/2025-08-13T12-46-36-SimpleCNNEncoder_SimpleCNNDecoder/checkpoints/epoch-5000.pt"
EncClass  = SimpleCNNEncoder          # <-- swap if needed
DecClass  = SimpleCNNDecoder          # <-- swap if needed


def _infer_emb_dim(enc_sd, dec_sd):
    for k in ("fc.weight", "proj.weight", "head.weight"):
        if k in enc_sd:
            return enc_sd[k].shape[0]
    if "fc.weight" in dec_sd:                     # decoder fc [out, E]
        return dec_sd["fc.weight"].shape[1]
    return 512  # fallback

def _load_models(ckpt_path, world_side, model_side, device):
    class Timer:
        def __enter__(self): self.t0=time.perf_counter(); return self
        def __exit__(self,*_): self.dt=time.perf_counter()-self.t0
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

class KeyPoller:
    """Non-blocking single-key reader (R to refresh, Q to quit)."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def poll(self, timeout=0.05):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            try:
                ch = sys.stdin.read(1)
                return ch
            except Exception:
                return None
        return None

# --- bench runner with live refresh ---
def test1():
    world_side = 150
    model_side = 120
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data generator (no reordering here)
    tb = TerrainBatch(
        world_size=world_side,
        build_dt=False, build_low=False, build_com=False, build_points=False
    )
    tb_iter = iter(tb)

    # viewers: input on 8090, output on 8091? (keeping your original args)
    client_in  = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client_out = voxelsim.RendererClient("127.0.0.1", 8082, 8083, 8090, 9090)
    client_in.connect_py(0); client_out.connect_py(0)

    # load models once
    enc, dec, t_load = _load_models(CKPT_PATH, model_side, world_side, device)
    print(f"📦 load ckpt         {t_load*1e3:7.2f} ms")

    def refresh():
        # generate + show input
        with Timer() as t_gen_conv:
            vd_np, _ = next(tb_iter)  # VoxelData from TerrainBatch
        print(f"🌍 generate+convert   {t_gen_conv.dt*1e3:7.2f} ms")

        with Timer() as t_show_in:
            show_voxels(vd_np, client_in)
        print(f"🖼  render (input)    {t_show_in.dt*1e3:7.2f} ms")

        # encode → decode → show output
        with Timer() as t_embed:
            with torch.no_grad():
                latent = enc.encode([vd_np.to_device(device)])
                if isinstance(latent, tuple):
                    latent = latent[0]
        print(f"🧠 embed (encode)    {t_embed.dt*1e3:7.2f} ms")

        with Timer() as t_recon_disp:
            with torch.no_grad():
                out = dec.decode(latent)
                logits = out["logits"]
            show_voxels(logits, client_out)
        print(f"🎯 recon+display     {t_recon_disp.dt*1e3:7.2f} ms")

    # initial render
    refresh()
    print("Controls: press R to refresh world, Q to quit.")

    # live loop
    with KeyPoller() as keys:
        while True:
            ch = keys.poll(0.1)
            if not ch:
                continue
            if ch in ("q", "Q"):
                print("bye.")
                break
            if ch in ("r", "R"):
                refresh()

if __name__ == "__main__":
    test1()
