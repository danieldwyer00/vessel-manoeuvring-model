import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'new')
from manoeuvringModelLibrary import *



class VesselEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
         #Variables *********************************************************************************************************************************************************************
        self.dt = 1/20
        self.Frames = 4000
        self.i = 0

        self.MaxRudder = np.deg2rad(30)
        self.MaxDistance = 100
        self.MaxSpeed = 5
        self.MaxYawRate = np.deg2rad(30)

        Position = np.zeros([3,1])
        Velocity = np.zeros([3,1])
        Acceleration = np.zeros([3,1])

        Position[0,0] = 0 #Initial y Position m
        Position[1,0] = 0 #Initial x Position m
        Position[2,0] = np.deg2rad(0) #Initial Yaw Position rad

        Velocity[0,0] = 0 #Initial Surge Velocity m/s
        Velocity[1,0] = 0 #Initial Sway Velocity m/s
        Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
        Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
        Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2

        self.Density = 1025  #kg/m3

        Length = 10 #m
        Beam = 3 #m
        Draft = 1.5 #m
        CB = 0.6

        lcg = 0 #m Fwd of MS
        kz = Length/4 #m


        if self.render_mode == "human":
            # Set up plot
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [], 'b--', label='Vessel Path')
            self.CourseLine, = self.ax.plot([], [], 'b--', label='Course')
            self.point, = self.ax.plot([], [], 'r',linewidth=5)  # current position
            self.marker, = self.ax.plot([], [], 'yo', markersize=8)
            self.ax.set_xlim(-50, 50)
            self.ax.set_ylim(-50, 50)
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_title("Vessel Motion")
            self.ax.grid(True)
            self.ax.xaxis.set_major_locator(MultipleLocator(5))
            self.ax.yaxis.set_major_locator(MultipleLocator(5))
            self.ax.legend()
        else:
           self.ax = None

        self.vessel = vessel_class(self.ax,"Vessel1",0,0,Position,Velocity,Acceleration,self.Density,Length,Beam,Draft,CB,lcg,kz,self.Frames)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(10,), dtype=np.float32)

    

    def step(self, action):
        self.Previous_Rudder_Angle = self.vessel.Rudder_Angle
        self.Previous_Throttle = self.vessel.Throttle

        self.vessel.Throttle = float(np.clip(action[0], -1.0, 1.0))
        self.vessel.Rudder_Angle = float(np.clip(action[1], -1.0, 1.0)) * self.MaxRudder

        self.Delta_Rudder_Angle = self.vessel.Rudder_Angle - self.Previous_Rudder_Angle
        self.Delta_Throttle = self.vessel.Throttle - self.Previous_Throttle

        self.vessel.Position,self.vessel.Velocity,self.vessel.Acceleration,self.vessel.GlobalVelocity,self.vessel.GlobalAcceleration = Model(self.vessel.Position,self.vessel.Velocity,self.vessel.Acceleration,self.dt,self.Density,self.vessel.Length,self.vessel.lcg,self.vessel.mass,self.vessel.Iz,self.vessel.YvPrime, self.vessel.YrPrime, self.vessel.NvPrime, self.vessel.NrPrime, self.vessel.XuPrimeFwd, self.vessel.XuPrimeReverse, 0.8, 4, self.vessel.Rudder_Angle,np.pi/2, self.vessel.Throttle)  

        #waypoint with desired heading

        #distance to wp
        self.DistanceToWaypoint = self.MyWaypoint.Distance(self.vessel)
        #velocity to wp
        self.VelocityToWaypoint = self.MyWaypoint.VelocityToWaypoint(self.vessel)
        #heading to wp error
        self.HeadingToWaypointError = self.MyWaypoint.HeadingError(self.vessel)
        #desired heading error
        self.DesiredHeadingError = self.MyWaypoint.DesiredHeadingError(self.vessel)

        vel = np.asarray(self.vessel.Velocity).flatten()  # ensures shape (3,)
        vx = float(vel[0]) if len(vel) > 0 else 0.0
        vy = float(vel[1]) if len(vel) > 1 else 0.0
        r  = float(vel[2]) if len(vel) > 2 else 0.0

        observation = np.array([
            float(np.clip(self.DistanceToWaypoint / float(self.MaxDistance), -1, 1)),
            float(self.VelocityToWaypoint / float(self.MaxSpeed)),
            float(np.sin(self.HeadingToWaypointError)),
            float(np.cos(self.HeadingToWaypointError)),
            float(np.sin(self.DesiredHeadingError)),
            float(np.cos(self.DesiredHeadingError)),
            r / float(self.MaxYawRate),
            vy / float(self.MaxSpeed),
            self.vessel.Throttle,
            self.vessel.Rudder_Angle
        ], dtype=np.float32)
        
        ####rewards
        #distance reward
        distanceReward = 0.4*np.e**(-0.002 * self.DistanceToWaypoint ** 2) + 0.6*np.e**(-1 * self.DistanceToWaypoint ** 2)
        #velocity reward at far distance
        velocityReward = 0.1 * self.VelocityToWaypoint * (1 - (0.4 * np.e ** (-0.002 * self.DistanceToWaypoint ** 2) + (0.6 * np.e ** (-0.000002 * self.DistanceToWaypoint ** 6))))
        #heading Reward
        headingReward = (np.e ** (-10 * self.DesiredHeadingError ** 2)) * (np.e**(-0.5 * self.DistanceToWaypoint ** 2))
        #overspeed penalty
        overspeed_penalty = (max(0, abs(self.vessel.Velocity[0]) - self.MaxSpeed) / self.MaxSpeed)

        rudder_change_penalty = self.Delta_Rudder_Angle / (2 * self.MaxRudder * self.dt)
        throttle_change_penalty = self.Delta_Throttle / (2 * self.dt)

        #add rewards
        reward = float(
            0.6 * distanceReward +
            0.2 * velocityReward +
            0.6 * headingReward -
            1 * overspeed_penalty - 
            3 * rudder_change_penalty -
            3 * throttle_change_penalty
        )

        if(self.DistanceToWaypoint > self.MaxDistance):
           reward += -100
           self.truncated = True
        

        #if(self.DistanceToWaypoint < 1 and abs(self.DesiredHeadingError) < np.deg2rad(5)):
         #  reward += 10

        if self.i >= self.Frames:
            self.truncated = True
            return observation, reward, self.terminated, self.truncated, {}
        
        self.vessel.BowPos[:,self.i:self.i+1], self.vessel.SternPos[:,self.i:self.i+1] = ShipPos(self.vessel.Position,self.vessel.Length,self.vessel.lcg)
        self.vessel.PosLog[:,self.i] = self.vessel.Position[:,0]   

        if self.render_mode == "human":
            self.vessel.line.set_data(self.vessel.Position[1,:self.i], self.vessel.Position[0,:self.i])     # update path
            self.vessel.point.set_data((self.vessel.PosLog[1,self.i],self.vessel.BowPos[1,self.i],self.vessel.SternPos[1,self.i]), (self.vessel.PosLog[0,self.i],self.vessel.BowPos[0,self.i],self.vessel.SternPos[0,self.i]))  # update current point
            self.vessel.point.set_linewidth(10)

            plotwaypointx = [self.MyWaypoint.Position[0]]
            plotwaypointy = [self.MyWaypoint.Position[1]]
            self.MyWaypoint.point.set_data((plotwaypointx,plotwaypointx + self.vessel.Length/2 * np.sin(self.MyWaypoint.DesiredHeading), plotwaypointx - self.vessel.Length/2 * np.sin(self.MyWaypoint.DesiredHeading)), (plotwaypointy, plotwaypointy + self.vessel.Length/2 * np.cos(self.MyWaypoint.DesiredHeading), plotwaypointy - self.vessel.Length/2 * np.cos(self.MyWaypoint.DesiredHeading)))
          
            #print("Heading Error: " + str(self.HeadingToWaypointError) + "Distance : " + str(self.DistanceToWaypoint))
            print("Throttle: " + str(self.vessel.Throttle) + " Rudder angle: " + str(np.rad2deg(self.vessel.Rudder_Angle)) + " Speed: " + str(self.vessel.Velocity[0]) + " Heading Error (degrees): " + str(np.rad2deg(self.DesiredHeadingError)) + " Distance (m): " + str(self.DistanceToWaypoint))
            #print("Heading Reward: " + str(headingReward) + " Desired Heading: " + str(np.rad2deg(self.MyWaypoint.DesiredHeading)) + " Heading Error: " + str(np.rad2deg(self.DesiredHeadingError)))
            #print("Heading Reward: " + str(headingReward) + " Distance Reward: " + str(distanceReward) + " Velocity Reward" + str(velocityReward))
            plt.pause(self.dt)

        info = {}
        self.i += 1
        self.t += self.dt

        return observation, reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.i = 0
        self.t = 0
        
        self.vessel.Position[0,0] = 0 #Initial y Position m
        self.vessel.Position[1,0] = 0 #Initial x Position m
        self.vessel.Position[2,0] = 0#np.deg2rad(np.random.uniform(-180,180)) #Initial Yaw Position rad

        self.vessel.Velocity[0,0] = 0 #Initial Surge Velocity m/s
        self.vessel.Velocity[1,0] = 0 #Initial Sway Velocity m/s
        self.vessel.Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        self.vessel.Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
        self.vessel.Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
        self.vessel.Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2

        self.MaxSpeed = np.random.uniform(1,6)
        self.MaxYawRate = 0.1

        #generate random waypoint
        self.MyWaypoint = waypoint(self.ax,[np.random.uniform(-50,50),np.random.uniform(-50,50)],np.random.uniform(-np.pi,np.pi))
        #distance to wp
        self.DistanceToWaypoint = self.MyWaypoint.Distance(self.vessel)
        #velocity to wp
        self.VelocityToWaypoint = self.MyWaypoint.VelocityToWaypoint(self.vessel)
        #heading to wp error
        self.HeadingToWaypointError = self.MyWaypoint.HeadingError(self.vessel)
        #desired heading error
        self.DesiredHeadingError = self.MyWaypoint.DesiredHeadingError(self.vessel)

        self.vessel.Throttle = 0
        self.vessel.Rudder_Angle = 0
        
        vel = np.asarray(self.vessel.Velocity).flatten()  # ensures shape (3,)
        vx = float(vel[0]) if len(vel) > 0 else 0.0
        vy = float(vel[1]) if len(vel) > 1 else 0.0
        r  = float(vel[2]) if len(vel) > 2 else 0.0

        observation = np.array([
            float(np.clip(self.DistanceToWaypoint / float(self.MaxDistance), -1, 1)),
            float(self.VelocityToWaypoint / float(self.MaxSpeed)),
            float(np.sin(self.HeadingToWaypointError)),
            float(np.cos(self.HeadingToWaypointError)),
            float(np.sin(self.DesiredHeadingError)),
            float(np.cos(self.DesiredHeadingError)),
            r / float(self.MaxYawRate),
            vy / float(self.MaxSpeed),
            self.vessel.Throttle,
            self.vessel.Rudder_Angle
        ], dtype=np.float32)


        return observation, {}

    def render(self):
        ...

    def close(self):
        ...