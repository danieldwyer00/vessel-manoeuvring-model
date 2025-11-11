import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from .DynamicPositioningEnv import VesselEnv  # must be importable at top-level

# Directories
run_id = int(time.time())
models_dir = f"DynamicPositioning/dynamicPositioningModels/PPO-{run_id}"
logdir = f"DynamicPositioning/dynamicPositioningLogs/PPO-{run_id}"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Number of parallel envs
n_envs = 16  # adjust to CPU cores

def make_env(rank: int, seed: int = 0):
    def _init():
        env = VesselEnv(render_mode="none")
        env = Monitor(env, filename=None)  # per-env Monitor; logs aggregate via TensorBoard
        # Gymnasium seeding via reset; different seed per rank
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # Good practice: set a base seed
    base_seed = 0
    set_random_seed(base_seed)

    # Create vectorized multiprocessing env
    env = SubprocVecEnv([make_env(i, base_seed) for i in range(n_envs)])

    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Keep overall per-update rollout size roughly similar:
    # n_envs * n_steps ~= previous single-env n_steps (1920) -> n_steps ≈ 240 for 8 envs
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=int(4096/n_envs),
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.1,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        use_sde=True,
        sde_sample_freq=64,
        verbose=1,
        normalize_advantage=True,
        tensorboard_log=logdir,
    )

    Timesteps = 4000
    try:
        for i in range(1, 250):
            model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="PPO")
            model.save(f"{models_dir}/{Timesteps * i}")
            # Save VecNormalize statistics for later evaluation/retraining
            env.save(os.path.join(models_dir, f"vecnormalize_{Timesteps * i}.pkl"))
    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model...")
        model.save(f"{models_dir}/latest_interrupt")
        env.save(os.path.join(models_dir, "vecnormalize_latest.pkl"))

    env.close()