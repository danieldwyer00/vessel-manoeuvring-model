from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os
import time
from VesselEnv import VesselEnv

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Wrap the environment
env = DummyVecEnv([lambda: Monitor(VesselEnv(render_mode="none"))])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy",  # or CnnPolicy if you're using visual input
            env,
            n_steps=2048,              # larger helps smooth learning for slow dynamics
            batch_size=512,            # large batches help stability in continuous envs
            gae_lambda=0.95,           # common default; balances bias/variance
            gamma=0.99,                # longer-term rewards; adjust if vessel takes time to respond
            n_epochs=10,               # multiple passes over data
            learning_rate=3e-4,        # or a schedule like linear or constant
            ent_coef=0.01,             # encourage exploration (reduce if jittery)
            clip_range=0.2,            # PPO clipping
            max_grad_norm=0.5,         # gradient clipping for stability
            vf_coef=0.5,               # value function loss weight
            use_sde=True,              # State-dependent exploration (helps in low-noise envs)
            sde_sample_freq=4,         # how often to resample noise
            verbose=1,# Print training progress
            tensorboard_log=logdir)

Timesteps = 10000
try:
    for i in range(1,1000):
        model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{Timesteps*i}")
        env.save(os.path.join(models_dir, f"vecnormalize_{Timesteps*i}.pkl"))

except KeyboardInterrupt:
    print("Training interrupted. Saving latest model...")
    model.save(f"{models_dir}/latest_interrupt")
    env.save(f"{models_dir}/vecnormalize_latest.pkl")

env.close()