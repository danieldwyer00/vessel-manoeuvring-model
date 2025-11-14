from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from .DynamicPositioningEnv import VesselEnv
import os

models_dir = "DynamicPositioning/dynamicPositioningModels/PPO-1763011259" 
model_path = os.path.join(models_dir, "latest_interrupt")  # path to the saved .zip
vecnorm_path = os.path.join(models_dir, "vecnormalize_latest.pkl")

def make_env():
    return Monitor(VesselEnv(render_mode="human"))

# Build single-env vectorized wrapper for evaluation
eval_venv = DummyVecEnv([make_env])

# Load normalization stats onto the eval VecEnv and freeze them
eval_venv = VecNormalize.load(vecnorm_path, eval_venv)
eval_venv.training = False
eval_venv.norm_reward = False

# Load the model with the new env (changes n_envs safely)
model = PPO.load(model_path, env=eval_venv)

# Evaluate deterministically
obs = eval_venv.reset()
for _ in range(10_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_venv.step(action)
    if done[0]:
        obs = eval_venv.reset()