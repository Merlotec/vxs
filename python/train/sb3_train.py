from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

from gymnasium.envs.registration import register
from envs.grid_world import GridWorldEnv, BaselineReward
import voxelsim as vxs;
# register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point="gymnasium_env.envs:GridWorldEnv",
# )

client = vxs.RendererClient.default_localhost_py()

# one agent
client.connect_py(1)

env = GridWorldEnv(
    reward_fn =BaselineReward(),
    client = client,
    start_pos=[100, 100, 20]
)



logdir = "logs/voxelsim_ppo"
os.makedirs(logdir, exist_ok=True)
train_env = Monitor(env)

# Optional eval env (no rendering for speed)
eval_env = Monitor(GridWorldEnv(reward_fn=BaselineReward(), start_pos = [100, 100, 20]))

# Callbacks (periodic eval + checkpoints)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(logdir, "best"),
    log_path=logdir,
    eval_freq=10_000,
    deterministic=True,
    render=False,
)
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
)

model.learn(total_timesteps=1_000_000, callback=[eval_cb, ckpt_cb])
model.save(os.path.join(logdir, "final_model"))
