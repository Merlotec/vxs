from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
import os
import multiprocessing as mp

from gymnasium.envs.registration import register
import torch
from envs.grid_world_astar import GridWorldAStarEnv
from rewards.target_locate import RewardTargetLocate
from models.vox_features import VoxGridExtractor
import voxelsim as vxs
# register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point="gymnasium_env.envs:GridWorldEnv",
# )

def _truthy_env(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    return str(val).lower() in ("1", "true", "yes", "y", "on")


run_rendered = _truthy_env("VXS_RUN_RENDERED", "1")
# run_rendered = True
# Force single-env, single-process to avoid GPU device loss with multiprocessing
num_envs = int(os.getenv("VXS_NUM_ENVS", "1"))
# Explicitly disable SubprocVecEnv usage
use_subproc = False


def make_env_fn(seed_offset: int = 0):
    def _thunk():
        env = GridWorldAStarEnv(
            reward=RewardTargetLocate(max_distance=1),
            render_client=None,
            start_pos=[100, 100, -40],
            action_gain=3.0,
            attempt_scales=(1.0, 0.85, 0.6, 0.4),
            allow_override=True,
            semantic_grid=True,
            max_world_time=30.0,
        )
        return env
    return _thunk


if run_rendered:
    # Single rendered env (legacy behavior)
    client = vxs.AsyncRendererClient.default_localhost_py(1)
    env = GridWorldAStarEnv(
        reward=RewardTargetLocate(max_distance=1),
        render_client=client,
        start_pos=[100, 100, -40],  # NED coordinate system
        action_gain=3.0,
        attempt_scales=(1.0, 0.85, 0.6, 0.4),
        allow_override=True,
        semantic_grid=True,
        max_world_time=30.0,
    )
    train_env = Monitor(env)
else:
    # Headless single-env training (parallel disabled to prevent GPU device loss)
    train_env = DummyVecEnv([make_env_fn(0)])
    train_env = VecMonitor(train_env)
    print("[sb3_train] Parallel training disabled: using single DummyVecEnv")


logdir = "logs/voxelsim_ppo"
os.makedirs(logdir, exist_ok=True)

# Optional eval env (no rendering for speed)
# eval_env = Monitor(GridWorldEnv(reward_fn=SimpleReqard(), start_pos = [100, 100, 20]))

# # Callbacks (periodic eval + checkpoints)
# eval_cb = EvalCallback(
#     eval_env,
#     best_model_save_path=os.path.join(logdir, "best"),
#     log_path=logdir,
#     eval_freq=10_000,
#     deterministic=True,
#     render=False,
# )
ckpt_cb = CheckpointCallback(
    save_freq=50_000, save_path=os.path.join(logdir, "ckpts"), name_prefix="ppo_vox"
)

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # MPS (Metal) on macOS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = _select_device()
print(f"[sb3_train] Using device: {device}")

model = PPO(
    policy="MultiInputPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=logdir,
    # You can tune these:
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(
        log_std_init=1.0,
        features_extractor_class=VoxGridExtractor,
        features_extractor_kwargs=dict(grid_key="grid"),
    ),
    device=device,
)

model.learn(total_timesteps=1_000_000, callback=[ckpt_cb])
model.save(os.path.join(logdir, "final_model"))
