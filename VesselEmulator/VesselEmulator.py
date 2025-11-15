import numpy as np
import cloudpickle
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from manoeuvringModelLibrary import *

class FakeEnv(gym.Env):
    def __init__(self,obs_shape, action_shape):
        super().__init__()
        self.observation_space = spaces.Box(low= -np.inf, high=np.inf, shape = obs_shape, dtype = np.float32)
        self.action_space = spaces.Box(low= -np.inf, high=np.inf, shape = action_shape, dtype = np.float32)
    
    def reset(self):
        return np.zeros(self.observation_space.shape, dtype= np.float32)

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype= np.float32),
            0.0,
            True,
            {},
        )

class AutonomyLoop:
    def __init__(self, shared_store):
        self.running = False
        self.store = shared_store
    def loop(self):
        self.running = True

        Position = np.zeros([3,1])
        Velocity = np.zeros([3,1])
        Acceleration = np.zeros([3,1])

        Position[0,0] = 200 #Initial y Position m
        Position[1,0] = 0 #Initial x Position m
        Position[2,0] = np.deg2rad(0) #Initial Yaw Position rad

        Velocity[0,0] = 0 #Initial Surge Velocity m/s
        Velocity[1,0] = 0 #Initial Sway Velocity m/s
        Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        self.Density = 1025  #kg/m3
        self.dt = 0.1

        self.vessel = vessel_from_file("VesselData/WAM-V",Position, Velocity)

        #load models
        #load DP model
        models_dir = "VesselData/WAM-V/DynamicPositioningModel" 
        model_path = os.path.join(models_dir, "latest_interrupt")  # path to the saved .zip
        vecnorm_path = os.path.join(models_dir, "vecnormalize_latest.pkl")

        model = PPO.load(model_path)
        fake_env = DummyVecEnv([lambda: FakeEnv((8,),(4,))])
        vecnorm = VecNormalize.load(vecnorm_path, fake_env)

        vecnorm.training = False
        vecnorm.norm_reward = False


        self.MyWaypoint = waypoint(None,[np.random.uniform(30),np.random.uniform(30)],np.random.uniform(-np.pi,np.pi))
        self.MaxDistance = 100
        self.MaxSpeed = 6

        while self.running:
            Mode = "DynamicPositioning"

            if Mode == "DynamicPositioning":
                #do DP
                #observations
                #distance to wp
                self.DistanceToWaypoint = self.MyWaypoint.Distance(self.vessel)
                #velocity to wp
                self.VelocityToWaypoint = self.MyWaypoint.VelocityToWaypoint(self.vessel)
                #heading to wp error
                self.HeadingToWaypointError = self.MyWaypoint.HeadingError(self.vessel)
                #desired heading error
                self.DesiredHeadingError = self.MyWaypoint.DesiredHeadingError(self.vessel)


                observation = np.array([
                    float(np.clip(self.DistanceToWaypoint / float(self.MaxDistance), -1, 1)),
                    float(self.VelocityToWaypoint / float(self.MaxSpeed)),
                    float(np.sin(self.HeadingToWaypointError)),
                    float(np.cos(self.HeadingToWaypointError)),
                    float(np.sin(self.DesiredHeadingError)),
                    float(np.cos(self.DesiredHeadingError)),
                    0.0,
                    0.0
                ], dtype=np.float32)

                obs_norm = vecnorm.normalize_obs(observation)
                #controls
                action,_ = model.predict(obs_norm)

                ThrottleCommands = [action[0], action[1]]
                RudderCommands = [action[0], action[1]]

                pass
            elif Mode == "ManualControl":
                #do Manual control
                ThrottleCommands = [0, 0]
                RudderCommands = [0, 0]
            
            

            #update vessel
            self.vessel.Update(ThrottleCommands, RudderCommands, self.dt, self.Density)

            #upload data
            self.store['latest'] = {'x': self.vessel.Position[1,0], 'y': self.vessel.Position[0,0], 'status': 'active'}
 
            # sleep for control rate
            import time; time.sleep(self.dt)
    def stop(self):
        self.running = False