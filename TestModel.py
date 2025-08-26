from stable_baselines3 import PPO
from VesselEnv import VesselEnv
import os

# Load environment (with rendering on for visualization)
env = VesselEnv(render_mode="human")


# Load the trained model
model = PPO.load("models/PPO-1754096311/2510000", env=env)

obs, info = env.reset()
done = False
truncated = False

episodes = 10
for i in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()