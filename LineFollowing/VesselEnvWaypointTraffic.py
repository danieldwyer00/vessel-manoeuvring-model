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
        
        self.detection_radius = 1000

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

        self.ax = None

        self.vessel = vessel_class(self.ax,"Vessel1",0,0,Position,Velocity,Acceleration,self.Density,Length,Beam,Draft,CB,lcg,kz,self.Frames)

        #Create Traffic
        self.numberOfTrafficVessels = 1
        self.Traffic = []
        
        for i in range(self.numberOfTrafficVessels):
            TrafficPosition = np.zeros([3,1])
            TrafficVelocity = np.zeros([3,1])
            TrafficAcceleration = np.zeros([3,1])
            TrafficPosition[0,0] = 0 #Initial y Position m
            TrafficPosition[1,0] = 0 #Initial x Position m
            TrafficPosition[2,0] = np.deg2rad(0) #Initial Yaw Position rad

            Length = 140
            Beam = 25
            Draft = 10
            CB = 0.75
            lcg = 0
            kz = Length/4

            self.Traffic.append(vessel_class(self.ax,"Vessel2",1,0,TrafficPosition,TrafficVelocity,TrafficAcceleration,self.Density,Length,Beam,Draft,CB,lcg,kz,self.Frames))

        if self.render_mode == "human":
            # Set up plot
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()

            # vessel line and point
            self.vessel.line, = self.ax.plot([], [], 'b--', label='Vessel Path')
            self.vessel.point, = self.ax.plot([], [], 'r',linewidth=self.vessel.Beam)  # current position

            #traffic line and point
            for i in range(self.numberOfTrafficVessels):
                self.Traffic[i].line, = self.ax.plot([], [], 'k--', label='Traffic Path')
                self.Traffic[i].point, = self.ax.plot([], [], 'g',linewidth=self.Traffic[i].Beam)  # current position

            #waypoint
            self.marker, = self.ax.plot([], [], 'yo', markersize=8)

            self.ax.set_xlim(-2000, 2000)
            self.ax.set_ylim(-2000, 2000)
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_title("Vessel Motion")
            self.ax.grid(True)
            self.ax.xaxis.set_major_locator(MultipleLocator(100))
            self.ax.yaxis.set_major_locator(MultipleLocator(100))
            self.ax.legend()
           

        

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-5000, high=5000, shape=(6,), dtype=np.float32)

    

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

        self.target_delta_x = self.target_x-self.vessel.Position[1,0]
        self.target_delta_y = self.target_y-self.vessel.Position[0,0]

        self.target_distance = (self.target_delta_x**2 + self.target_delta_y**2)**0.5

        #Angle to waypoint
        self.heading_error = heading_error(self.vessel.Position[1,0], self.vessel.Position[0,0], np.pi/2-self.vessel.Position[2,0], self.target_x, self.target_y)
        
        # update traffic
        for i in range(self.numberOfTrafficVessels):
            self.Traffic[i].Position,self.Traffic[i].Velocity,self.vessel.Acceleration,_,_ = Model(self.Traffic[i].Position,self.Traffic[i].Velocity,self.Traffic[i].Acceleration,self.dt,self.Density,self.Traffic[i].Length,self.Traffic[i].lcg,self.Traffic[i].mass,self.Traffic[i].Iz,self.Traffic[i].YvPrime, self.Traffic[i].YrPrime, self.Traffic[i].NvPrime, self.Traffic[i].NrPrime, self.Traffic[i].XuPrime,0.8,5,self.Traffic[i].Rudder_Angle,np.pi/2, self.Traffic[i].Throttle)
            
            #Distance to Traffic and Relative Heading to Traffic
            Traffic_delta_x = self.Traffic[i].Position[1,0]-self.vessel.Position[1,0]
            Traffic_delta_y = self.Traffic[i].Position[0,0]-self.vessel.Position[0,0]

            temp_distance = (Traffic_delta_x**2 + Traffic_delta_y**2)**0.5
            if temp_distance < self.detection_radius:
                self.Traffic[i].Distance = temp_distance
                self.Traffic[i].HeadingError = heading_error(self.vessel.Position[1,0], self.vessel.Position[0,0], np.pi/2-self.vessel.Position[2,0], self.Traffic[i].Position[1,0], self.Traffic[i].Position[0,0])
            else:
                self.Traffic[i].Distance = 9999999
                self.Traffic[i].HeadingError = 0

            
            
            

            
        
        observation = np.array([self.vessel.Velocity[0,0], self.vessel.Velocity[2,0], self.target_distance, self.heading_error, self.Traffic[0].Distance, self.Traffic[0].HeadingError],dtype=np.float32)

        if self.i >= self.Frames:
            self.truncated = True
            self.done = True
            reward = 0
            return observation, reward, self.done, self.truncated, {}
        
        progress_reward = ((self.prev_dist - self.target_distance)/self.dt)/10
        
        heading_reward = - abs(self.heading_error)/np.pi

        speed_encouragement = abs(self.vessel.Velocity[0,0]) * 0.1

        #Traffic Reward Fuction
        nearestTraffic = 9999999
        for i in range(self.numberOfTrafficVessels):
            if self.Traffic[i].Distance < nearestTraffic:
                nearestTraffic = self.Traffic[i].Distance

        if nearestTraffic < self.detection_radius:
            traffic_penalty = - (self.detection_radius - nearestTraffic) / self.detection_radius
        else:
            traffic_penalty = 0

        reward = (
            1.0 * progress_reward +
            0.1 * speed_encouragement
            + 1.5 * heading_reward
            + 0.1 * traffic_penalty
        )

        if(self.target_distance < 100):
           reward += 100
           self.done = True
        if(self.target_distance > 5000):
           reward += -100
           self.truncated = True
           self.done = True

        self.prev_dist = self.target_distance
        
        self.vessel.BowPos[:,self.i:self.i+1], self.vessel.SternPos[:,self.i:self.i+1] = ShipPos(self.vessel.Position,self.vessel.Length,self.vessel.lcg)
        self.vessel.PosLog[:,self.i] = self.vessel.Position[:,0]

        for i in range(self.numberOfTrafficVessels):
            self.Traffic[i].BowPos[:,self.i:self.i+1], self.Traffic[i].SternPos[:,self.i:self.i+1] = ShipPos(self.Traffic[i].Position,self.Traffic[i].Length,self.Traffic[i].lcg)
            self.Traffic[i].PosLog[:,self.i] = self.Traffic[i].Position[:,0] 
            if self.render_mode == "human":
                self.Traffic[i].line.set_data(self.Traffic[i].Position[1,:self.i], self.Traffic[i].Position[i,:self.i])     # update path
                self.Traffic[i].point.set_data((self.Traffic[i].PosLog[1,self.i],self.Traffic[i].BowPos[1,self.i],self.Traffic[i].SternPos[1,self.i]), (self.Traffic[i].PosLog[0,self.i],self.Traffic[i].BowPos[0,self.i],self.Traffic[i].SternPos[0,self.i]))  # update current point

        if self.render_mode == "human":
            self.vessel.line.set_data(self.vessel.Position[1,:self.i], self.vessel.Position[0,:self.i])     # update path
            self.vessel.point.set_data((self.vessel.PosLog[1,self.i],self.vessel.BowPos[1,self.i],self.vessel.SternPos[1,self.i]), (self.vessel.PosLog[0,self.i],self.vessel.BowPos[0,self.i],self.vessel.SternPos[0,self.i]))  # update current point

          

            self.marker.set_data([self.target_x],[self.target_y])
            print(str(self.i) + " Reward: " + str(reward) + "Traffic Distance: " + str(self.Traffic[0].Distance) + "Traffic Heading Error: " + str(self.Traffic[0].HeadingError))
            plt.pause(0.001)  # small pause to update the plot

        info = {}
        self.i += 1
        self.t += self.dt

        return observation, reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        def heading_error(robot_x, robot_y, robot_theta, waypoint_x, waypoint_y):
            # Calculate desired heading to waypoint
            desired_theta = np.arctan2(waypoint_y - robot_y, waypoint_x - robot_x)
            # Calculate error (positive = right, negative = left)
            error = normalize_angle(desired_theta - robot_theta)
            return error
        
        self.done = False
        self.truncated = False
        self.i = 0
        self.t = 0
        
        self.vessel.Position[0,0] = 0 #Initial y Position m
        self.vessel.Position[1,0] = 0 #Initial x Position m
        self.vessel.Position[2,0] = np.deg2rad(0) #Initial Yaw Position rad

        self.vessel.Velocity[0,0] = 10 #Initial Surge Velocity m/s
        self.vessel.Velocity[1,0] = 0 #Initial Sway Velocity m/s
        self.vessel.Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        self.vessel.Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
        self.vessel.Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
        self.vessel.Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2

        normalized_heading = normalize_angle(self.vessel.Position[2,0])

        #add traffic position
        for i in range(self.numberOfTrafficVessels):
            self.Traffic[i].Position[0,0] = random.uniform(-2000,2000) #Initial y Position m
            self.Traffic[i].Position[1,0] = random.uniform(-2000,2000) #Initial x Position m
            self.Traffic[i].Position[2,0] = random.uniform(-np.pi,np.pi) #Initial Yaw Position rad

            self.Traffic[i].Velocity[0,0] = 10 #Initial Surge Velocity m/s
            self.Traffic[i].Velocity[1,0] = 0 #Initial Sway Velocity m/s
            self.Traffic[i].Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

            #Distance to Traffic
            Traffic_delta_x = self.Traffic[i].Position[1,0]-self.vessel.Position[1,0]
            Traffic_delta_y = self.Traffic[i].Position[0,0]-self.vessel.Position[0,0]
            self.Traffic[i].Distance = (Traffic_delta_x**2 + Traffic_delta_y**2)**0.5
            
            #Relative Heading to Traffic
            self.Traffic[i].HeadingError = heading_error(self.vessel.Position[1,0], self.vessel.Position[0,0], np.pi/2-self.vessel.Position[2,0], self.Traffic[i].Position[1,0], self.Traffic[i].Position[0,0])


        

        #Set Target Position
        self.target_x = random.randint(-2000, 2000)
        self.target_y = random.randint(-2000, 2000)

        self.target_delta_x = self.target_x-self.vessel.Position[1,0]
        self.target_delta_y = self.target_y-self.vessel.Position[0,0]

        self.target_distance = (self.target_delta_x**2 + self.target_delta_y**2)**0.5
        self.prev_dist = self.target_distance
        #Angle to waypoint
        self.heading_error = heading_error(self.vessel.Position[1,0], self.vessel.Position[0,0], np.pi/2-self.vessel.Position[2,0], self.target_x, self.target_y)
        
        


        observation = np.array([self.vessel.Velocity[0,0], self.vessel.Velocity[2,0], self.target_distance, self.heading_error, self.Traffic[0].Distance, self.Traffic[0].HeadingError],dtype=np.float32)
        return observation, {}

    def render(self):
        ...

    def close(self):
        ...