import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise
from .DynamicPositioningEnv import VesselEnv  # must be importable at top-level

# Directories
run_id = int(time.time())
models_dir = f"DynamicPositioning/dynamicPositioningModels/TD3-{run_id}"
logdir = f"DynamicPositioning/dynamicPositioningLogs/TD3-{run_id}"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Number of parallel envs
n_envs = 16  # adjust to CPU cores

def make_env(rank: int, seed: int = 0):
    def _init():
        env = VesselEnv(render_mode="none")
        env = Monitor(env, filename=None)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    base_seed = 0
    set_random_seed(base_seed)

    # Vectorized multiprocessing env
    env = SubprocVecEnv([make_env(i, base_seed) for i in range(n_envs)])

    # Normalize observations and rewards (TD3 typically normalizes obs; reward normalization is optional)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Action noise for exploration (Normal or Ornstein-Uhlenbeck). Match action_dim to env.action_space
    # After VecNormalize, env is wrapped; use the unwrapped action_space from a temp env or create a dummy env
    tmp_env = VesselEnv(render_mode="none")
    action_dim = tmp_env.action_space.shape[0]
    tmp_env.close()
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),     # was ("episode")
        gradient_steps=1,           # consider 2–4 for more updates per step
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        tensorboard_log=logdir,
        verbose=1,
        )

    # Off-policy total_timesteps can be larger; keep checkpoints as before
    Timesteps = 6000
    try:
        for i in range(1, 500):
            model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="TD3", log_interval=1)
            model.save(f"{models_dir}/{Timesteps * i}")
            # Save VecNormalize statistics
            env.save(os.path.join(models_dir, f"vecnormalize_{Timesteps * i}.pkl"))
    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model...")
        model.save(f"{models_dir}/latest_interrupt")
        env.save(os.path.join(models_dir, "vecnormalize_latest.pkl"))

    env.close()