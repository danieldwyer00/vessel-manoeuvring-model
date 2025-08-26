from stable_baselines3.common.env_checker import check_env
from VesselEnv import VesselEnv

env = VesselEnv()
episodes = 10

for episode in range (episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("action: ",random_action)
        obs, reward, done, truncated, info = env.step(random_action)
        print("reward: ", reward)