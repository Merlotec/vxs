from pynput import keyboard, mouse
import voxelsim as vxs
import time
import threading
import torch
import numpy as np

# ============ IMPORTS FROM bench_world48.py ============
from representation import (VoxelData, show_voxels, SimpleCNNEncoder, SimpleCNNDecoder,)

# ============ MODEL CONFIG FROM bench_world48.py ============
CKPT_PATH = "/home/box/drone/vxs/runs/2025-08-13T12-46-36-SimpleCNNEncoder_SimpleCNNDecoder/checkpoints/epoch-5000.pt"
EncClass = SimpleCNNEncoder
DecClass = SimpleCNNDecoder

# ============ MODEL LOADING FUNCTIONS FROM bench_world48.py ============
def _infer_emb_dim(enc_sd, dec_sd):
    for k in ("fc.weight", "proj.weight", "head.weight"):
        if k in enc_sd:
            return enc_sd[k].shape[0]
    if "fc.weight" in dec_sd:
        return dec_sd["fc.weight"].shape[1]
    return 512  # fallback

def _load_models(ckpt_path, world_side, model_side, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_sd, dec_sd = ckpt["encoder"], ckpt["decoder"]
    
    E = _infer_emb_dim(enc_sd, dec_sd)
    enc = EncClass(voxel_size=model_side, embedding_dim=E).to(device).eval()
    dec = DecClass(voxel_size=model_side, embedding_dim=E).to(device).eval()
    
    def _prune_mismatched(sd, module):
        current = module.state_dict()
        removed = []
        for k in list(sd.keys()):
            if (k not in current) or (sd[k].shape != current[k].shape):
                removed.append(k)
                sd.pop(k)
        return removed
    
    dropped_enc = _prune_mismatched(enc_sd, enc)
    dropped_dec = _prune_mismatched(dec_sd, dec)
    
    enc.load_state_dict(enc_sd, strict=False)
    dec.load_state_dict(dec_sd, strict=False)
    
    print(f"encoder: loaded {len(enc_sd)} keys, skipped {len(dropped_enc)} → {dropped_enc}")
    print(f"decoder: loaded {len(dec_sd)} keys, skipped {len(dropped_dec)} → {dropped_dec}")
    return enc, dec, 0.0

# ============ SIMPLE QUEUE CLASS ============
class LatestJob:
    """Single-slot queue that only keeps the latest item"""
    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
    
    def set(self, item):
        with self.lock:
            self.data = item
    
    def get_and_clear(self):
        with self.lock:
            data = self.data
            self.data = None
            return data

latest = LatestJob()

# ============ MODEL SETUP ============
MODEL_SIDE = 120  # or 64, depending on your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading models on {device}...")
encoder, decoder, t_load = _load_models(CKPT_PATH, MODEL_SIDE, MODEL_SIDE, device)
print(f"📦 Models loaded in {t_load*1e3:.2f} ms")

# ============ HELPER FUNCTIONS ============
def send_sparse_coords(coords_np, vals_np, client):
    """Send sparse coordinates to renderer client"""
    # Filter out empty cells
    m = vals_np > 0.0
    if m.sum() == 0:
        return
    
    # Build cell dictionary
    cell_dict = {}
    for (x, y, z), v in zip(coords_np[m], vals_np[m]):
        coord = (int(x), int(y), int(z))
        # FILLED if > 0.75, else SPARSE
        cell = vxs.Cell.filled() if v > 0.75 else vxs.Cell.sparse()
        cell_dict[coord] = cell
    
    # Create and send world
    if hasattr(vxs.VoxelGrid, "from_dict_py"):
        world = vxs.VoxelGrid.from_dict_py(cell_dict)
    else:
        # Fallback if from_dict_py doesn't exist
        world = vxs.VoxelGrid()
        for coord, cell in cell_dict.items():
            world.set_cell_py(coord, cell)
    
    client.send_world_py(world)

def make_voxeldata(coords_np, vals_np, side_hint):
    """Convert sparse numpy arrays to VoxelData for model - matching bench_world48 structure"""
    # Create VoxelData using the same structure as in representation.py
    # Filter for occupied cells
    m = vals_np > 0.0
    if m.sum() == 0:
        # Return empty VoxelData
        return VoxelData(
            occupied_coords=torch.zeros((0, 3), dtype=torch.float32),
            values=torch.zeros(0, dtype=torch.float32),
            bounds=torch.tensor([side_hint, side_hint, side_hint], dtype=torch.float32),
            drone_pos=torch.tensor([side_hint//2, side_hint//2, side_hint//2], dtype=torch.float32),
        )
    
    return VoxelData(
        occupied_coords=torch.from_numpy(coords_np[m].astype(np.float32)),
        values=torch.from_numpy(vals_np[m].astype(np.float32)),
        bounds=torch.tensor([side_hint, side_hint, side_hint], dtype=torch.float32),
        drone_pos=torch.tensor([side_hint//2, side_hint//2, side_hint//2], dtype=torch.float32),
    )

# ============ WORKER THREAD ============
def worker_encode_decode():
    """Worker thread that processes belief maps without blocking sim"""
    while not stop_worker.is_set():
        data = latest.get_and_clear()
        if data is None:
            time.sleep(0.01)
            continue
        
        coords_np, vals_np, ts = data
        
        # 1. Send sparse overlay to filter world client
        send_sparse_coords(coords_np, vals_np, client_fw)
        
        # 2. Run model
        try:
            # Create VoxelData
            vd = make_voxeldata(coords_np, vals_np, MODEL_SIDE)
            
            # Run inference (matching bench_world48 approach)
            with torch.no_grad():
                # Move to device and encode
                latent = encoder.encode([vd.to_device(device)])
                if isinstance(latent, tuple):
                    latent = latent[0]
                
                # Decode
                out = decoder.decode(latent)
                logits = out["logits"] if isinstance(out, dict) else out
            
            # Visualize reconstruction
            show_voxels(logits, client_pred)
            
        except Exception as e:
            print(f"Model inference error: {e}")
            import traceback
            traceback.print_exc()

# ============ EXISTING SETUP CODE ============
agent = vxs.Agent(0)
agent.set_pos([50.0, 50.0, -20.0])

fw = vxs.FilterWorld()
dynamics = vxs.px4.PX4Dynamics.default_py()
chaser = vxs.FixedLookaheadChaser.default_py()

generator = vxs.TerrainGenerator()
generator.generate_terrain_py(vxs.TerrainConfig.default_py())
world = generator.generate_world_py()
proj = vxs.CameraProjection.default_py()
env = vxs.EnvState.default_py()

AGENT_CAMERA_TILT = -0.5
camera_orientation = vxs.CameraOrientation.vertical_tilt_py(AGENT_CAMERA_TILT)

# Renderer
noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

# Main client
client = vxs.RendererClient.default_localhost_py()
client.connect_py(1)
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

client.send_world_py(world)
client.send_agents_py({0: agent})

# Filter world client (belief map)
client_fw = vxs.RendererClient("127.0.0.1", 8084, 8085, 8090, 9090)
client_fw.connect_py(0)

# Prediction client (model output)
client_pred = vxs.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
client_pred.connect_py(0)

# ============ START WORKER THREAD ============
stop_worker = threading.Event()
worker_thread = threading.Thread(target=worker_encode_decode, daemon=True)
worker_thread.start()

# ============ KEYBOARD HANDLING ============
pressed = set()
just_pressed = set()

def on_press(key):
    try:
        key_str = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            key_str = 'space'
        elif key == keyboard.Key.shift:
            key_str = 'shift'
        elif key == keyboard.Key.esc:
            return False
        else:
            return
    
    if key_str not in pressed:
        just_pressed.add(key_str)
    pressed.add(key_str)

def on_release(key):
    try:
        key_str = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            key_str = 'space'
        elif key == keyboard.Key.shift:
            key_str = 'shift'
        else:
            return
    pressed.discard(key_str)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ============ MODIFIED WORLD UPDATE CALLBACK ============
upd_start = 0.0
def world_update(fw_arg, timestamp):
    """Minimal callback - just queue the data and return"""
    dtime = time.time() - upd_start
    print(f"upd_time: {dtime:.3f}s")
    
    # Get sparse numpy arrays
    coords, vals = fw.as_numpy()
    
    # Queue for worker thread (only if we have data)
    if coords.shape[0] > 0:
        latest.set((coords.copy(), vals.copy(), timestamp))
    
    # That's it! Worker thread handles everything else

# ============ MAIN LOOP ============
delta = 0.01
last_view_time = time.time()
FRAME_DELTA_MAX = 0.13
YAW_STEP = 0.3

print("\nStarting simulation...")
print("Three renderer windows should open:")
print("  1. Main view (default ports)")
print("  2. Belief map (port 8084)")
print("  3. Model reconstruction (port 8086)")

while listener.running:
    t0 = time.time()
    view_delta = t0 - last_view_time
    
    # Rendering
    if fw.is_updating_py(last_view_time):
        if view_delta >= FRAME_DELTA_MAX:
            continue
    else:
        if view_delta >= FRAME_DELTA_MAX:
            fw.send_pov_py(client, 0, 0, proj, camera_orientation)
            upd_start = time.time()
            renderer.update_filter_world_py(
                agent.camera_view_py(camera_orientation), 
                proj, fw, t0, world_update
            )
            last_view_time = t0
    
    # Handle actions
    action = agent.get_action()
    commands = []
    if action:
        commands = action.get_commands()
    commands_cl = list(commands)
    
    # Compute yaw delta
    yaw_delta = 0.0
    if 'q' in just_pressed:
        yaw_delta -= YAW_STEP
    if 'e' in just_pressed:
        yaw_delta += YAW_STEP
    
    # Apply movement commands
    if 'w' in just_pressed: 
        commands.append(vxs.MoveCommand(vxs.MoveDir.Forward, 0.8, yaw_delta))
    if 's' in just_pressed: 
        commands.append(vxs.MoveCommand(vxs.MoveDir.Back, 0.8, yaw_delta))
    if 'a' in just_pressed: 
        commands.append(vxs.MoveCommand(vxs.MoveDir.Left, 0.8, -3.14))
    if 'd' in just_pressed: 
        commands.append(vxs.MoveCommand(vxs.MoveDir.Right, 0.8, 3.14))
    if 'space' in just_pressed: 
        commands.append(vxs.MoveCommand(vxs.MoveDir.Up, 0.8, yaw_delta))
    if 'shift' in just_pressed: 
        commands.append(vxs.MoveCommand(vxs.MoveDir.Down, 0.8, yaw_delta))
    
    if not action or commands != commands_cl:
        if len(commands) > 0:
            agent.perform_sequence_py(commands)
    
    # Update dynamics
    chase_target = chaser.step_chase_py(agent, delta)
    dynamics.update_agent_dynamics_py(agent, env, chase_target, delta)
    just_pressed.clear()
    
    # Check collisions
    collisions = world.collisions_py(agent.get_pos(), [0.5, 0.5, 0.3])
    if len(collisions) > 0:
        print(f"{len(collisions)} collisions")
    
    client.send_agents_py({0: agent})
    
    # Frame timing
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)

# Cleanup
print("\nShutting down...")
stop_worker.set()
worker_thread.join()
listener.join()