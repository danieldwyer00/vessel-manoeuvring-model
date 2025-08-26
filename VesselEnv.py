import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import numpy as np
import matplotlib.pyplot as plt

def angle_difference(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi  # result in [-π, π]

class VesselEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}
    


    def __init__(self, render_mode="none"):
        super(VesselEnv,self).__init__()
        self.render_mode = render_mode
        #Simulation Properties
        self.totalTime = 1000 #Seconds
        self.dt = 0.1 #Seconds
        self.Frames = int(self.totalTime/self.dt)

        self.time = 0

        self.Position = np.zeros((3,self.Frames))
        self.Velocity = np.zeros((3,self.Frames))
        self.Acceleration = np.zeros((3,self.Frames))

        #Variables
        self.Density = 1025  #kg/m3

        self.Length = 140 #m
        self.Beam = 25 #m
        self.Draft = 6 #m
        self.CB = 0.75

        self.LCG = 0 #m Fwd of MS
        self.kz = 35 #m

        self.mass = self.Length*self.Beam*self.Draft*self.CB*self.Density #Kg
        self.Iz = self.mass*self.kz**2

        self.BowPos = np.zeros((2,self.Frames))
        self.SternPos = np.zeros((2,self.Frames))

        #Non Dimensional Derivatives Using Clarke
        self.YvPrime = -(1+0.4*self.CB*(self.Beam/self.Draft))*np.pi*(self.Draft/self.Length)**2*10
        self.YrPrime = -(-0.5+2.2*(self.Beam/self.Length)-0.08*(self.Beam/self.Draft))*np.pi*(self.Draft/self.Length)**2
        self.NvPrime = -(0.5+2.4*(self.Draft/self.Length))*np.pi*(self.Draft/self.Length)**2
        self.NrPrime = -(0.5+2.4*(self.Draft/self.Length))*np.pi*(self.Draft/self.Length)**2
        self.XuPrime = -0.002

        self.XudotPrime = 0
        self.YvdotPrime = 0
        self.YrdotPrime = 0
        self.NvdotPrime = 0
        self.NrdotPrime = 0

        self.YRudderPrime =  0.001
        self.NRudderPrime = -0.01


        if self.render_mode == "human":
            # Set up plot
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [], 'b--', label='Vessel Path')
            self.point, = self.ax.plot([], [], 'r',linewidth=10)  # current position
            self.marker, = self.ax.plot([], [], 'yo', markersize=8)
            self.ax.set_xlim(-800, 800)
            self.ax.set_ylim(-800, 800)
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_title("Vessel Motion")
            self.ax.grid(True)
            self.ax.legend()


        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-5000, high=5000,
                                            shape=(6,), dtype=np.float32)
        if self.render_mode == "human":
          plt.show(block=False)
    
        self.distance = 0

    def step(self, action):
        self.i += 1

        done = False
        truncated = False
        if self.i >= self.Frames:
          truncated = True
          done = True
          reward = 0

          return self.observation, reward, done, truncated, {}
        
        prev_distance = self.target_distance
        #Scale Non Dimensional Derivatives
        self.NetVelocity = max(0.1,np.sqrt(self.Velocity[0,self.i-1]**2+self.Velocity[1,self.i-1]**2))

        self.Yv = self.YvPrime*0.5*self.Density*self.NetVelocity*self.Length**2
        self.Yr = self.YrPrime*0.5*self.Density*self.NetVelocity*self.Length**3
        self.Nv = self.NvPrime*0.5*self.Density*self.NetVelocity*self.Length**3
        self.Nr = self.NrPrime*0.5*self.Density*self.NetVelocity*self.Length**4
        self.Xu = self.XuPrime*0.5*self.Density*self.NetVelocity*self.Length**2

        self.Xudot = 0
        self.Yvdot = 0
        self.Yrdot = 0
        self.Nvdot = 0
        self.Nrdot = 0
        
        
        self.NRudder = self.NRudderPrime*0.5*self.Density*self.NetVelocity**2*self.Length**3
        self.YRudder = self.YRudderPrime*0.5*self.Density*self.NetVelocity**2*self.Length**2

        self.throttle = np.clip(action[0], -1.0, 1.0)
        self.rudderAngle = np.clip(action[1], -1.0, 1.0) * np.deg2rad(45)

        self.YawMoment = self.NRudder*self.rudderAngle
        self.RudderSway = self.YRudder*self.rudderAngle

        self.Thrust = 700000*self.throttle

        self.RigidBodyMassMatrix = np.array([[self.mass,0,0],
                                            [0,self.mass,self.mass*self.LCG],
                                            [0,self.mass*self.LCG,self.Iz]])

        self.AddedMassMatrix = -np.array([[self.Xudot,0,0],
                                          [0,self.Yvdot,self.Yrdot],
                                          [0,self.Nvdot,self.Nrdot]])

        self.DampingMatrix = -np.array( [[self.Xu,0,0],
                                        [0,self.Yv,self.Yr],
                                        [0,self.Nv,self.Nr]])

        self.tau = np.array([[self.Thrust],
                            [self.RudderSway],
                            [self.YawMoment]])

        self.RigidBodyCoriolisMatrix = np.array( [[0,0,-self.mass*(self.LCG*self.Velocity[2,self.i-1]+self.Velocity[1,self.i-1])],
                                                  [0,0,self.mass*self.Velocity[0,self.i-1]],
                                                  [self.mass*(self.LCG*self.Velocity[2,self.i-1]+self.Velocity[1,self.i-1]),-self.mass*self.Velocity[0,self.i-1],0]])

        self.AddedMassCoriolisMatrix = np.array( [[0,0,-self.Yvdot*self.Velocity[1,self.i-1]-self.Yrdot*self.Velocity[2,self.i-1]],
                                                  [0,0,-self.Xudot*self.Velocity[0,self.i-1]],
                                                  [-(-self.Yvdot*self.Velocity[1,self.i-1]-self.Yrdot*self.Velocity[2,self.i-1]),self.Xudot*self.Velocity[0,self.i-1],0]])


        #Calculate Acceleration
        self.Acceleration[:,[self.i]] = np.linalg.inv(self.RigidBodyMassMatrix+self.AddedMassMatrix) @ (self.tau - ((self.RigidBodyCoriolisMatrix+self.AddedMassCoriolisMatrix+self.DampingMatrix)@self.Velocity[:,[self.i-1]]))
        
        #Integrate for velocities
        self.Velocity[:,[self.i]] = self.Velocity[:,[self.i-1]] + self.Acceleration[:,[self.i]]*self.dt
        
        #Rotation Matrix
        self.R = np.array(  [[np.cos(self.Position[2,self.i-1]),-np.sin(self.Position[2,self.i-1]),0],
                        [np.sin(self.Position[2,self.i-1]),np.cos(self.Position[2,self.i-1]),0],
                        [0,0,1]])
        #Calculate Global Position
        self.Position[:,[self.i]] = self.Position[:,[self.i-1]] + (self.R @ self.Velocity[:,[self.i]])*self.dt

        #Calculate Bow Position
        #Surge
        self.BowPos[0,self.i] = self.Position[0,[self.i]]+(self.Length/2-self.LCG)*np.cos(self.Position[2,[self.i]])
        #Sway
        self.BowPos[1,self.i] = self.Position[1,[self.i]]+(self.Length/2-self.LCG)*np.sin(self.Position[2,[self.i]])
        #Calculate Stern Positions
        #Surge
        self.SternPos[0,self.i] = self.Position[0,[self.i]]-(self.Length/2+self.LCG)*np.cos(self.Position[2,[self.i]])
        #Sway
        self.SternPos[1,self.i] = self.Position[1,[self.i]]-(self.Length/2+self.LCG)*np.sin(self.Position[2,[self.i]])

        if self.render_mode == "human":
          self.line.set_data(self.Position[1,:self.i], self.Position[0,:self.i])     # update path
          self.point.set_data((self.Position[1,self.i:self.i+1],self.BowPos[1,self.i:self.i+1],self.SternPos[1,self.i:self.i+1]), (self.Position[0,self.i:self.i+1],self.BowPos[0,self.i:self.i+1],self.SternPos[0,self.i:self.i+1]))  # update current point
          self.marker.set_data([self.target_x],[self.target_y])
          plt.pause(0.1)  # small pause to update the plot
        
        self.target_delta_x = self.target_x-self.Position[1,self.i]
        self.target_delta_y = self.target_y-self.Position[0,self.i]
        
        self.target_distance = (self.target_delta_x**2 + self.target_delta_y**2)**0.5


        #Angle to waypoint
        self.target_Heading = np.arctan2(self.target_delta_y,self.target_delta_x)
        
        # Heading error [-π, π]
        heading_error = angle_difference(self.Position[2,self.i],self.target_Heading)

        progress_reward = (prev_distance-self.target_distance)/self.dt 

        alignment_reward = np.exp(-abs(heading_error))  

        forward_reward = self.Velocity[0, self.i] * np.cos(heading_error)

        self.time_reward += -self.dt * 0.001

        reward = (
            3.0 * alignment_reward +
            0.0 * forward_reward +
            2.0 * progress_reward + 
            1.0 * self.time_reward
        )

        if(self.target_distance < 100):
           reward += 10
           done = True
        if(self.target_distance > 5000):
           reward += -10
           truncated = True
           done = True

        self.observation = np.array([self.Velocity[0,self.i], self.Velocity[1,self.i],self.Velocity[2,self.i], self.target_distance, self.target_Heading, heading_error],dtype=np.float32)
        
        if done or truncated:
           info = {
              "episode": {
                 "r": self.total_reward,
                 "l": self.steps
                 }
            }
        else:
            info = {}

        self.time += self.dt
        return self.observation, reward, done,truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)  # Optional: If you want to seed numpy randomness
            
        done = False
        self.i = 0
        self.time = 0

        self.total_reward = 0
        self.steps = 0
        self.time_reward = 0
        #Initial Conditions
        self.Position[0,0] = 0 #Initial Surge Position m
        self.Position[1,0] = 0 #Initial Sway Position m
        self.Position[2,0] = np.deg2rad(0) #Initial Yaw Position rad

        self.Velocity[0,0] = 6 #Initial Surge Velocity m/s
        self.Velocity[1,0] = 0 #Initial Sway Velocity m/s
        self.Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

        self.Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
        self.Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
        self.Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2
        
        #Set Target Position
        self.target_x = random.randint(-800, 800)
        self.target_y = random.randint(-800, 800)

        self.target_delta_x = self.target_x-self.Position[1,self.i]
        self.target_delta_y = self.target_y-self.Position[0,self.i]

        self.target_distance = (self.target_delta_x**2 + self.target_delta_y**2)**0.5

        #Angle to waypoint
        self.target_Heading = np.arctan2(self.target_delta_y,self.target_delta_x)
        
        # Heading error [-π, π]
        heading_error = angle_difference(self.Position[2,self.i],self.target_Heading)
        

        self.observation = np.array([self.Velocity[0,0], self.Velocity[1,0],self.Velocity[2,0], self.target_distance, self.target_Heading, heading_error],dtype=np.float32)

        return self.observation, {}
