import numpy as np
from haversine import haversine, Unit

class vessel_class:
    def __init__(self, ax, Name, Throttle, Rudder_Angle, Position, Velocity, Acceleration, Density, Length, Beam, Draft, CB, LCG, kz, Frames):
        self.Name = Name
        
        
        self.Position = Position #[surge, sway, yaw]
        self.Velocity = Velocity
        self.Acceleration = Acceleration

        self.Length = Length
        self.Beam = Beam
        self.Draft = Draft
        self.CB = CB
        self.lcg = LCG
        self.kz = kz
        self.mass = self.Length*self.Beam*self.Draft*self.CB*Density
        self.Iz = self.mass*self.kz**2

        self.YvPrime, self.YrPrime, self.NvPrime, self.NrPrime = ClarkeDerivatives(self.Length,self.Beam,self.Draft,self.CB)
        self.XuPrimeFwd = -0.003
        self.XuPrimeReverse = -0.01

        self.PosLog = np.zeros([3,Frames])
        self.BowPos = np.zeros([2,Frames])
        self.SternPos = np.zeros([2,Frames])

        self.Throttle = Throttle
        self.Rudder_Angle = Rudder_Angle

        if ax is not None:
            self.line, = ax.plot([], [], 'b--', label='Vessel Path')
            self.point, = ax.plot([], [], 'r',linewidth=Beam)  # current position

class line:
    def __init__(self, p1x, p1y, p2x, p2y):
        #Assign Position Vectors 
        self.p1 = np.array([p1x, p1y])
        self.p2 = np.array([p2x, p2y])

        self.line_vector = self.p2 - self.p1

        #Calculate Length
        self.Length = np.linalg.norm(self.p2 - self.p1)

class waypoint:
    def __init__(self, ax, Position, DesiredHeading):
        self.Position = Position
        self.DesiredHeading = DesiredHeading

        if ax is not None:
            self.point, = ax.plot([], [], 'bo', label='Waypoint')

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def Distance(self, Vessel):
        return ((self.Position[0] - Vessel.Position[1])**2 + (self.Position[1] - Vessel.Position[0])**2)**0.5
    
    def VelocityToWaypoint(self, Vessel):
        #components of surge and sway in the direction of the wp
        heading_error = self.HeadingError(Vessel)   # not self.HeadingError(self, Vessel)
        Velocity = Vessel.Velocity[0] * np.cos(heading_error)
        return Velocity
    
    def HeadingError(self, Vessel):
        # Calculate desired heading to waypoint
        desired_theta = np.arctan2(self.Position[0] - Vessel.Position[1], self.Position[1] - Vessel.Position[0])
        # Calculate error (positive = right, negative = left)
        error = self.normalize_angle(desired_theta - Vessel.Position[2])
        return error
    
    def DesiredHeadingError(self, Vessel):
        error = self.normalize_angle(self.DesiredHeading - Vessel.Position[2])
        return error

    
    


def Model(Position,Velocity,Acceleration,dt,Density,Length,LCG,mass,Iz,YvPrime,YrPrime,NvPrime,NrPrime, XuPrimeFwd, XuPrimeReverse, CL, Ar, delta_attack, delta_stall, Throttle):
    #velocity magnitude
    NetVelocity = max(0.1,np.sqrt(Velocity[0,0]**2+Velocity[1,0]**2))

    if Velocity[0] >= 0:
        XuPrime = XuPrimeFwd
    else:
        XuPrime = XuPrimeReverse

    #Convert Non-dimensional to dimensional
    Yv = YvPrime * 0.5 * Density * NetVelocity * Length ** 2
    Yr = YrPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nv = NvPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nr = NrPrime * 0.5 * Density * NetVelocity * Length ** 4
    Xu = XuPrime * 0.5 * Density * NetVelocity * Length ** 2

    Xudot = 0
    Yvdot = 0
    Yrdot = 0
    Nvdot = 0
    Nrdot = 0

    RigidBodyMassMatrix = np.array([[mass,0,0],
                               [0,mass,mass*LCG],
                               [0,mass*LCG,Iz]])

    AddedMassMatrix = -np.array([[Xudot,0,0],
                                [0,Yvdot,Yrdot],
                                [0,Nvdot,Nrdot]])

    DampingMatrix = -np.array( [[Xu,0,0],
                                [0,Yv,Yr],
                                [0,Nv,Nr]])

    CP = np.array([-5.0, 0.0, 0.5])       # rudder 10 m aft, 1 m below CG
    CG = np.array([0.0, 0.0, 0.0])

    #Propulsion
    MaxPropRPM = 500
    PropRPS = np.sign(Throttle) * max(abs((MaxPropRPM/60) * Throttle), 0.001)

    J_data = [0,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,	0.66,	0.67,	0.68,	0.72,	0.73,	0.74,	0.75,	0.76,	0.77,	0.78,	0.79,	0.80]
    KT_data = [0.2418, 0.2418,	0.2368,	0.2317,	0.2266,	0.2214,	0.2162,	0.2110,	0.2057,	0.2004,	0.1951,	0.1897,	0.1844,	0.1789,	0.1735,	0.1680,	0.1624,	0.1569,	0.1513,	0.1457,	0.1400,	0.1343,	0.1286,	0.1053,	0.0994,	0.0934,	0.0874,	0.0814,	0.0754,	0.0693,	0.0632,	0.0570]
    KQ_data = [0.0318, 0.0318,	0.0313,	0.0308,	0.0303,	0.0298,	0.0292,	0.0287,	0.0281,	0.0276,	0.0270,	0.0264,	0.0258,	0.0252,	0.0246,	0.0240,	0.0234,	0.0227,	0.0221,	0.0214,	0.0207,	0.0201,	0.0194,	0.0165,	0.0158,	0.0150,	0.0143,	0.0135,	0.0127,	0.0119,	0.0111,	0.0103]

    wakeFraction = 0.1
    thrustDeductionFactor = 0.3
    prop_diameter = 1

    AdvanceVelocity = Velocity[0]*(1-wakeFraction)
    J = abs(AdvanceVelocity)/(abs(PropRPS) * prop_diameter)

    KT = np.interp(J, J_data, KT_data)
    KQ = np.interp(J, J_data, KQ_data)
    #print(KT)
    ThrustProduced = Density * PropRPS * abs(PropRPS) * prop_diameter ** 4 * KT
    Thrust = ThrustProduced * (1 - thrustDeductionFactor)


    #rudder flow speed using momentum theory
    a = 2 * Density * np.pi * (prop_diameter / 2) ** 2
    b = a * AdvanceVelocity
    vi = np.sign(ThrustProduced) * (-b + (b ** 2 + 4 * a * abs(ThrustProduced)) ** 0.5) / (2 * a)
    RudderFlowVelocity = AdvanceVelocity + vi
    #print("Rudder Flow (m/s): " + str(RudderFlowVelocity))
    

    rudder_forces = np.zeros([1,3])
    rudder_moments = np.zeros([1,3])

    if abs(delta_attack) < delta_stall:
        F_rudder = 0.5 * Density * CL * Ar * RudderFlowVelocity * abs(RudderFlowVelocity) * np.sin(np.pi/2 * delta_attack/ delta_stall)
    else:
        F_rudder = 0.5 * Density * CL * Ar * RudderFlowVelocity * abs(RudderFlowVelocity) * np.sin(np.pi/2 * delta_attack)
    
    #print("Rudder Force (KN): " + str(F_rudder/1000))
    #Rudder Surge
    rudder_forces[0,0] = -F_rudder * np.sin(delta_attack)
    #Rudder Sway
    rudder_forces[0,1] = F_rudder * np.cos(delta_attack)
    #Rudder Yaw
    rudder_moments = np.cross(CP-CG,rudder_forces)


    #print("Thrust (KN)" + str(Thrust/1000) + "Rudder Yaw Moment (KNm)" + str(rudder_moments[0,2]/1000))
    tau = np.array([float(rudder_forces[0,0] + Thrust),
                   rudder_forces[0,1],
                   rudder_moments[0,2]]).reshape(3,1)
   
    RigidBodyCoriolisMatrix = np.array( [[0,0,-mass*(LCG*Velocity[2,0]+Velocity[1,0])],
                                        [0,0,mass*Velocity[0,0]],
                                        [mass*(LCG*Velocity[2,0]+Velocity[1,0]),-mass*Velocity[0,0],0]])
    
    AddedMassCoriolisMatrix = np.array( [[0,0,-Yvdot*Velocity[1,0]-Yrdot*Velocity[2,0]],
                                        [0,0,-Xudot*Velocity[0,0]],
                                        [-(-Yvdot*Velocity[1,0]-Yrdot*Velocity[2,0]),Xudot*Velocity[0,0],0]])
    
    #Rotation Matrix
    R = np.array(  [[np.cos(Position[2,0]),-np.sin(Position[2,0]),0],
                    [np.sin(Position[2,0]),np.cos(Position[2,0]),0],
                    [0,0,1]])
    #Calculate Local Acceleration and Velocity
    NewLocalAcceleration = np.linalg.inv(RigidBodyMassMatrix+AddedMassMatrix) @ (tau - ((RigidBodyCoriolisMatrix+AddedMassCoriolisMatrix+DampingMatrix)@Velocity))
    NewLocalVelocity = Velocity + NewLocalAcceleration*dt

    NewGlobalPosition = Position + (R @ NewLocalVelocity)*dt
    NewGlobalAcceleration = R @ NewLocalAcceleration
    NewGlobalVelocity = R @ NewLocalVelocity



    return  NewGlobalPosition, NewLocalVelocity, NewLocalAcceleration, NewGlobalVelocity, NewGlobalAcceleration

def ClarkeDerivatives(Length,Beam,Draft,CB):
    #Non Dimensional Derivatives Using Clarke
    YvPrime = -(1+0.4*CB*(Beam/Draft))*np.pi*(Draft/Length)**2*10
    YrPrime = -(-0.5+2.2*(Beam/Length)-0.08*(Beam/Draft))*np.pi*(Draft/Length)**2
    NvPrime = -(0.5+2.4*(Draft/Length))*np.pi*(Draft/Length)**2
    NrPrime = -(0.5+2.4*(Draft/Length))*np.pi*(Draft/Length)**2
    return YvPrime, YrPrime, NvPrime, NrPrime

def ShipPos(Position,Length,LCG):
    BowPos = np.zeros([2,1])
    SternPos = np.zeros([2,1])
    BowPos[0,0] = Position[0,0]+(Length/2-LCG)*np.cos(Position[2,0])
    #Sway
    BowPos[1,0] = Position[1,0]+(Length/2-LCG)*np.sin(Position[2,0])
    #Calculate Stern Positions
    #Surge
    SternPos[0,0] = Position[0,0]-(Length/2+LCG)*np.cos(Position[2,0])
    #Sway
    SternPos[1,0] = Position[1,0]-(Length/2+LCG)*np.sin(Position[2,0])
    return BowPos, SternPos


def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def path_frame(line):
    """
    Returns d_hat (unit +y_path), n_hat (unit +x_path), and path heading phi
    in 'y-forward, clockwise-positive' convention.
    """
    dx, dy = line.line_vector.astype(float)
    L = np.hypot(dx, dy)
    if L == 0:
        raise ValueError("Line points are identical; undefined direction.")
    d_hat = np.array([dx, dy]) / L                    # +y_path (downrange)
    n_hat = np.array([ d_hat[1], -d_hat[0] ])         # +x_path (right of path)
    # Heading 0 along +y, positive clockwise -> phi = atan2(dx, dy)
    phi = np.arctan2(dx, dy)
    return d_hat, n_hat, phi

def cross_track_and_heading_error(line, vessel, unwrap_state=None):
    d_hat, n_hat, phi = path_frame(line)

    # Your storage (as per your attempt): Position[0,0]=y, Position[1,0]=x, Position[2,0]=psi_y
    y = float(vessel.Position[0, 0])
    x = float(vessel.Position[1, 0])
    psi_y = float(vessel.Position[2, 0])  # already 0 along +y, + clockwise

    # Position rel. to path origin
    r = np.array([x, y], dtype=float) - line.p1.astype(float)

    x_cross = n_hat @ r
    y_along = d_hat @ r

    # Heading error in same convention
    hdg_err_wrapped = wrap_to_pi(psi_y - phi)

    if unwrap_state is not None:
        hdg_err_unwrapped = unwrap_state.step(hdg_err_wrapped)
    else:
        hdg_err_unwrapped = None

    return x_cross, y_along, hdg_err_wrapped, hdg_err_unwrapped, phi

