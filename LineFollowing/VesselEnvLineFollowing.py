import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from manoeuvringModelLibrary import *



class VesselEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
         #Variables *********************************************************************************************************************************************************************
        self.dt = 0.1
        self.Frames = 4000
        self.i = 0
        
        Position = np.zeros([3,1])
        Velocity = np.zeros([3,1])
        Acceleration = np.zeros([3,1])

        Position[0,0] = 0 #Initial y Position m
        Position[1,0] = 0 #Initial x Position m
        Position[2,0] = np.deg2rad(0) #Initial Yaw Position rad

        Velocity[0,0] = 10 #Initial Surge Velocity m/s
        Velocity[1,0] = 0 #Initial Sway Velocity m/s
        Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
        Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
        Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2

        self.Density = 1025  #kg/m3

        Length = 140 #m
        Beam = 25 #m
        Draft = 10 #m
        CB = 0.75

        lcg = 20 #m Fwd of MS
        kz = 35 #m

        mass = Length*Beam*Draft*CB*self.Density #Kg
        Iz = mass*kz**2

        YvPrime, YrPrime, NvPrime, NrPrime = ClarkeDerivatives(Length,Beam,Draft,CB)
        XuPrime = -0.04

        

        if self.render_mode == "human":
            # Set up plot
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [], 'b--', label='Vessel Path')
            self.CourseLine, = self.ax.plot([], [], 'b--', label='Course')
            self.point, = self.ax.plot([], [], 'r',linewidth=10)  # current position
            self.marker, = self.ax.plot([], [], 'yo', markersize=8)
            self.ax.set_xlim(-400, 400)
            self.ax.set_ylim(0, 2000)
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_title("Vessel Motion")
            self.ax.grid(True)
            self.ax.xaxis.set_major_locator(MultipleLocator(100))
            self.ax.yaxis.set_major_locator(MultipleLocator(100))
            self.ax.legend()
        else:
           self.ax = None

        self.vessel = vessel_class(self.ax,"Vessel1",0,0,Position,Velocity,Acceleration,self.Density,Length,Beam,Draft,CB,lcg,kz,self.Frames)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-5000, high=5000, shape=(5,), dtype=np.float32)

    

    def step(self, action):
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        def heading_error(robot_x, robot_y, robot_theta, waypoint_x, waypoint_y):
            # Calculate desired heading to waypoint
            desired_theta = np.arctan2(waypoint_y - robot_y, waypoint_x - robot_x)
            # Calculate error (positive = right, negative = left)
            error = normalize_angle(desired_theta - robot_theta)
            return error
        

        
        self.vessel.Throttle = np.clip(action[0], -0.0, 1.0)
        self.vessel.Rudder_Angle = np.clip(action[1], -1.0, 1.0) * np.deg2rad(45)

        self.vessel.Position,self.vessel.Velocity,self.vessel.Acceleration,_,_ = Model(self.vessel.Position,self.vessel.Velocity,self.vessel.Acceleration,self.dt,self.Density,self.vessel.Length,self.vessel.lcg,self.vessel.mass,self.vessel.Iz,self.vessel.YvPrime, self.vessel.YrPrime, self.vessel.NvPrime, self.vessel.NrPrime, self.vessel.XuPrime,0.8,5,self.vessel.Rudder_Angle,np.pi/2, self.vessel.Throttle)  

        previouscrossTrackError = self.vessel.crossTrackError
        previousDownRangeVelocity = self.vessel.DownRangePosition

        self.vessel.crossTrackError, self.vessel.DownRangePosition, self.vessel.Heading_Error, _, _ = cross_track_and_heading_error(self.track, self.vessel)
        #print("Cross Track Error (m): " + str(self.vessel.crossTrackError))
        #print("Down Range (m): " + str(self.vessel.DownRangePosition))
        #print("Heading Error (rad): " + str(self.vessel.Heading_Error))

        crossTrackVelocity =  (self.vessel.crossTrackError - previouscrossTrackError)/self.dt
        downRangeVelocity = (self.vessel.DownRangePosition - previousDownRangeVelocity)/self.dt
        
        
        observation = np.array([downRangeVelocity, self.vessel.Heading_Error, self.vessel.Velocity[2,0], self.vessel.crossTrackError, crossTrackVelocity],dtype=np.float32)

        

        speed_encouragement = downRangeVelocity / ((self.vessel.Velocity[0,0]**2 + self.vessel.Velocity[1,0]**2)**0.5)

        cross_track_error_reward = 0.6*np.e**(-0.000005*self.vessel.crossTrackError**2) + 0.4*np.e**(-0.001*self.vessel.crossTrackError**2)
        
        heading_error_reward = np.e**(-10*abs(self.vessel.Heading_Error)) * np.e**(-0.0001*self.vessel.crossTrackError**2)

        if self.vessel.crossTrackError < 0:
            cross_track_velocity_reward = (crossTrackVelocity / ((self.vessel.Velocity[0,0]**2 + self.vessel.Velocity[1,0]**2)**0.5)) * (1 - np.e**(-0.000005*self.vessel.crossTrackError**2))
        else:
            cross_track_velocity_reward = -(crossTrackVelocity / ((self.vessel.Velocity[0,0]**2 + self.vessel.Velocity[1,0]**2)**0.5)) * (1 - np.e**(-0.000005*self.vessel.crossTrackError**2))

        reward = (
            0.05 * speed_encouragement + 
            1.5 * cross_track_error_reward +
            0.5 * heading_error_reward +
            1.0 * cross_track_velocity_reward
        )

        if(abs(self.vessel.crossTrackError) > 1500):
           reward += -100
           self.truncated = True

        if self.i >= self.Frames:
            self.truncated = True
            return observation, reward, self.terminated, self.truncated, {}
        
        self.vessel.BowPos[:,self.i:self.i+1], self.vessel.SternPos[:,self.i:self.i+1] = ShipPos(self.vessel.Position,self.vessel.Length,self.vessel.lcg)
        self.vessel.PosLog[:,self.i] = self.vessel.Position[:,0]   

        if self.render_mode == "human":
          self.vessel.line.set_data(self.vessel.Position[1,:self.i], self.vessel.Position[0,:self.i])     # update path
          self.vessel.point.set_data((self.vessel.PosLog[1,self.i],self.vessel.BowPos[1,self.i],self.vessel.SternPos[1,self.i]), (self.vessel.PosLog[0,self.i],self.vessel.BowPos[0,self.i],self.vessel.SternPos[0,self.i]))  # update current point
          self.CourseLine.set_data([self.track.p1[0], self.track.p2[0]],[self.track.p1[1], self.track.p2[1]])

          plt.pause(0.001)  # small pause to update the plot

        #print(self.vessel.crossTrackError)
        #print(cross_track_velocity_reward)
        #print(cross_track_error_reward)
        #print(heading_error_reward)
        info = {}
        self.i += 1
        self.t += self.dt

        return observation, reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        def heading_error(robot_x, robot_y, robot_theta, waypoint_x, waypoint_y):
            # Calculate desired heading to waypoint
            desired_theta = np.arctan2(waypoint_y - robot_y, waypoint_x - robot_x)
            # Calculate error (positive = right, negative = left)
            error = normalize_angle(desired_theta - robot_theta)
            return error
        
        self.terminated = False
        self.truncated = False
        self.i = 0
        self.t = 0
        
        self.vessel.Position[0,0] = 0 #Initial y Position m
        self.vessel.Position[1,0] = np.random.uniform(-500,500) #Initial x Position m
        self.vessel.Position[2,0] = 0#np.deg2rad(np.random.uniform(-180,180)) #Initial Yaw Position rad

        self.vessel.Velocity[0,0] = np.random.uniform(0,15) #Initial Surge Velocity m/s
        self.vessel.Velocity[1,0] = 0 #Initial Sway Velocity m/s
        self.vessel.Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        self.vessel.Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
        self.vessel.Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
        self.vessel.Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2

        normalized_heading = normalize_angle(self.vessel.Position[2,0])

        #Set line to follow
                        #p1x p1y, p2x, p2y
        self.track = line(0,0,0,1000)

        self.vessel.crossTrackError, self.vessel.DownRangePosition, self.vessel.Heading_Error, _, _ = cross_track_and_heading_error(self.track, self.vessel)
       
        

        
        observation = np.array([0, self.vessel.Heading_Error, self.vessel.Velocity[2,0], self.vessel.crossTrackError, 0],dtype=np.float32)
        return observation, {}

    def render(self):
        ...

    def close(self):
        ...