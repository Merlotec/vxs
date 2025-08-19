from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

from gymnasium.envs.registration import register
from envs.grid_world_astar import GridWorldAStarEnv, SimpleReward
import voxelsim as vxs;
# register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point="gymnasium_env.envs:GridWorldEnv",
# )

client = vxs.AsyncRendererClient.default_localhost_py(1)

# one agent

env = GridWorldAStarEnv(
    reward=SimpleReward(
        plan_success_bonus=0.2,
        distance_bonus_per_step=0.02,
    ),
    render_client=client,
    start_pos=[100, 100, -40],  # NED coordinate system
    action_gain=3.0,
    attempt_scales=(1.0, 0.85, 0.6, 0.4),
    allow_override=True,
)



logdir = "logs/voxelsim_ppo"
os.makedirs(logdir, exist_ok=True)
train_env = Monitor(env)

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
    policy_kwargs=dict(log_std_init=1.0),
    device="mps",
)

model.learn(total_timesteps=1_000_000, callback=[ckpt_cb])
model.save(os.path.join(logdir, "final_model"))
