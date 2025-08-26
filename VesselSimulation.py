import matplotlib.pyplot as plt
import numpy as np

#Simulation Properties
totalTime = 1000 #Seconds
dt = 1 #Seconds
Frames = int(totalTime/dt)

time = 0

Position = np.zeros((3,Frames))
Velocity = np.zeros((3,Frames))
Acceleration = np.zeros((3,Frames))

#Initial Conditions
Position[0,0] = 0 #Initial Surge Position m
Position[1,0] = 0 #Initial Sway Position m
Position[2,0] = np.deg2rad(0) #Initial Yaw Position rad

Velocity[0,0] = 0 #Initial Surge Velocity m/s
Velocity[1,0] = 0 #Initial Sway Velocity m/s
Velocity[2,0] = 0 #Initial Yaw Velocity rad/s

Acceleration[0,0] = 0 #Initial Surge Acceleration m/s2
Acceleration[1,0] = 0 #Initial Sway Acceleration m/s2
Acceleration[2,0] = 0 #Initial Yaw Acceleration rad/s2



#Variables
Density = 1025  #kg/m3

Length = 140 #m
Beam = 25 #m
Draft = 6 #m
CB = 0.75

LCG = 0 #m Fwd of MS
kz = 35 #m

mass = Length*Beam*Draft*CB*Density #Kg
Iz = mass*kz**2
#
BowPos = np.zeros((2,Frames))
SternPos = np.zeros((2,Frames))

#Non Dimensional Derivatives Using Clarke
YvPrime = -(1+0.4*CB*(Beam/Draft))*np.pi*(Draft/Length)**2*10
YrPrime = -(-0.5+2.2*(Beam/Length)-0.08*(Beam/Draft))*np.pi*(Draft/Length)**2
NvPrime = -(0.5+2.4*(Draft/Length))*np.pi*(Draft/Length)**2
NrPrime = -(0.5+2.4*(Draft/Length))*np.pi*(Draft/Length)**2
XuPrime = -0.002

XudotPrime = 0
YvdotPrime = 0
YrdotPrime = 0
NvdotPrime = 0
NrdotPrime = 0

YRudderPrime =  0.001
NRudderPrime = -0.005

# Set up plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b--', label='Vessel Path')
point, = ax.plot([], [], 'r',linewidth=10)  # current position
ax.set_xlim(-800, 0)
ax.set_ylim(0, 1100)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Vessel Motion")
ax.grid(True)
ax.legend()

deltaPrime = mass/(0.5*Density*Length**3)
xgprime = LCG/Length
if((NrPrime-deltaPrime*xgprime)/(YrPrime-deltaPrime)>NvPrime/YvPrime):
    print("Dynamically Stable")
else:
    print("Dynamically Unstable")
    

#Simulate
for i in range(1,Frames):
    time += dt
    #Scale Non Dimensional Derivatives
    NetVelocity = max(0.1,np.sqrt(Velocity[0,i-1]**2+Velocity[1,i-1]**2))
    if(time<100):
        rudderAngle = 0
    else:
        rudderAngle = np.deg2rad(30)

    Yv = YvPrime*0.5*Density*NetVelocity*Length**2
    Yr = YrPrime*0.5*Density*NetVelocity*Length**3
    Nv = NvPrime*0.5*Density*NetVelocity*Length**3
    Nr = NrPrime*0.5*Density*NetVelocity*Length**4
    Xu = XuPrime*0.5*Density*NetVelocity*Length**2

    Xudot = 0
    Yvdot = 0
    Yrdot = 0
    Nvdot = 0
    Nrdot = 0
    
    
    NRudder = NRudderPrime*0.5*Density*NetVelocity**2*Length**3
    YRudder = YRudderPrime*0.5*Density*NetVelocity**2*Length**2

    YawMoment = NRudder*rudderAngle
    RudderSway = YRudder*rudderAngle

    Thrust = 700000

    RigidBodyMassMatrix = np.array([[mass,0,0],
                               [0,mass,mass*LCG],
                               [0,mass*LCG,Iz]])

    AddedMassMatrix = -np.array([[Xudot,0,0],
                                [0,Yvdot,Yrdot],
                                [0,Nvdot,Nrdot]])

    DampingMatrix = -np.array( [[Xu,0,0],
                                [0,Yv,Yr],
                                [0,Nv,Nr]])

    tau = np.array([[Thrust],
                   [RudderSway],
                   [YawMoment]])
   
    RigidBodyCoriolisMatrix = np.array( [[0,0,-mass*(LCG*Velocity[2,i-1]+Velocity[1,i-1])],
                                        [0,0,mass*Velocity[0,i-1]],
                                        [mass*(LCG*Velocity[2,i-1]+Velocity[1,i-1]),-mass*Velocity[0,i-1],0]])
    
    AddedMassCoriolisMatrix = np.array( [[0,0,-Yvdot*Velocity[1,i-1]-Yrdot*Velocity[2,i-1]],
                                        [0,0,-Xudot*Velocity[0,i-1]],
                                        [-(-Yvdot*Velocity[1,i-1]-Yrdot*Velocity[2,i-1]),Xudot*Velocity[0,i-1],0]])


    #Calculate Acceleration
    Acceleration[:,[i]] = np.linalg.inv(RigidBodyMassMatrix+AddedMassMatrix) @ (tau - ((RigidBodyCoriolisMatrix+AddedMassCoriolisMatrix+DampingMatrix)@Velocity[:,[i-1]]))
    
    #Integrate for Body Fixed Velocities
    Velocity[:,[i]] = Velocity[:,[i-1]] + Acceleration[:,[i]]*dt
    
    #Rotation Matrix
    R = np.array(  [[np.cos(Position[2,i-1]),-np.sin(Position[2,i-1]),0],
                    [np.sin(Position[2,i-1]),np.cos(Position[2,i-1]),0],
                    [0,0,1]])
    #Calculate Global Position
    Position[:,[i]] = Position[:,[i-1]] + (R @ Velocity[:,[i]])*dt

    #Calculate Bow Position
    #Surge
    BowPos[0,i] = Position[0,[i]]+(Length/2-LCG)*np.cos(Position[2,[i]])
    #Sway
    BowPos[1,i] = Position[1,[i]]+(Length/2-LCG)*np.sin(Position[2,[i]])
    #Calculate Stern Positions
    #Surge
    SternPos[0,i] = Position[0,[i]]-(Length/2+LCG)*np.cos(Position[2,[i]])
    #Sway
    SternPos[1,i] = Position[1,[i]]-(Length/2+LCG)*np.sin(Position[2,[i]])

    #print("Velocity (m/s) :" + str(Velocity[:,i]))

    line.set_data(Position[1,:i], Position[0,:i])     # update path
    point.set_data((Position[1,i:i+1],BowPos[1,i:i+1],SternPos[1,i:i+1]), (Position[0,i:i+1],BowPos[0,i:i+1],SternPos[0,i:i+1]))  # update current point
    plt.pause(0.001)  # small pause to update the plot


