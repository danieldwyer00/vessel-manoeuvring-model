import numpy as np
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

class vessel_class:
    def __init__(self, Name, Throttle, Rudder_Angle, Position, Velocity, Acceleration, Density, Length, Beam, Draft, CB, Displacement, LCG, kz):
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
        self.mass = Displacement
        self.Iz = self.mass*self.kz**2

        self.YvPrime, self.YrPrime, self.NvPrime, self.NrPrime = ClarkeDerivatives(self.Length,self.Beam,self.Draft,self.CB)

        self.PortThrottleCommand = 0 
        self.StbdThrottleCommand = 0
        self.PortRudderCommand = 0
        self.StbdRudderCommand = 0

        #if ax is not None:
        #    self.line, = ax.plot([], [], 'b--', label='Vessel Path')
         #   self.outline, = ax.plot([], [], 'r',linewidth=Beam)  # current position
    
    def Update(self, dt, Density):
        self.Position,self.Velocity,self.Acceleration,self.GlobalVelocity,self.GlobalAcceleration = WAMVModel(self.Position,self.Velocity,self.Acceleration,dt,Density,self.Length,self.lcg,self.mass,self.Iz, self.PortThrottleCommand, self.StbdThrottleCommand, self.PortRudderCommand, self.StbdRudderCommand)
        
        self.BowPos, self.SternPos = ShipPos(self.Position,self.Length,self.lcg)
        #self.outline.set_data((self.Position[1],self.BowPos[1],self.SternPos[1]), (self.Position[0],self.BowPos[0],self.SternPos[0]))


# Templates for reading YAML files
class Geometry(BaseModel):
    LOA: float
    LWL: float
    B: float
    T: float

class MassProps(BaseModel):
    displacement: float
    CG: List[float]
    kz: float

class VesselRoot(BaseModel):
    name: str
    geometry: Geometry
    mass: MassProps
    files: dict

class LinearDamping(BaseModel):
    Yv: float
    Yr: float
    Nv: float
    Nr: float
    Xu: float

class AddedMass(BaseModel):
    Yvdot: float
    Yrdot: float
    Nvdot: float
    Nrdot: float
    Xudot: float

class Hydrodynamics(BaseModel):
    linear_damping: LinearDamping
    added_mass : AddedMass

class Hydrostatics(BaseModel):
    test: float

class Propeller(BaseModel):
    id: int
    name: Optional[str]
    D: float
    position: list[float]
    rotation: Literal["left","right"] = "right"
    max_rpm: float
    wake_fraction: float
    thrust_deduction: float
    kt_table: Dict[str, list]
    kq_table: Dict[str, list] 
    pivoting: bool
    max_angle: float

class Propulsion(BaseModel):
    propellers: List[Propeller] = []



class vessel_from_file:
    def __init__(self, dir, initialPos, initialVel):
        self.dir = dir
        self.load_vessel_data()

        self.Position = initialPos
        self.Velocity = initialVel
        self.Acceleration = np.array([0, 0, 0])

        self.BowPos, self.SternPos = ShipPos(self.Position, self.vessel_data.geometry.LOA, self.vessel_data.mass.CG[0])

    def load_yaml(self, path: Path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_vessel_data(self):
        #read basic vessel data and other directories
        root = self.load_yaml(Path(self.dir + "/Vessel.yml"))
        self.vessel_data = VesselRoot(**root["vessel"])

        #read technical data
        self.hydrostatics_data = Hydrostatics(**self.load_yaml(Path(self.dir + self.vessel_data.files["hydrostatics"]))["hydrostatics"])
        self.hydrodynamics_data = Hydrodynamics(**self.load_yaml(Path(self.dir + self.vessel_data.files["hydrodynamics"]))["hydrodynamics"])
        self.propulsion_data = Propulsion(**self.load_yaml(Path(self.dir + self.vessel_data.files["propulsion"]))["propulsion"])

    def UniversalManouvringModel(self, ThrottleCommand, RudderCommand, dt, Density):
        #mass properties
        mass = self.vessel_data.mass.displacement
        LCG = self.vessel_data.mass.CG[0]
        Iz = self.vessel_data.mass.kz ** 2 * mass
        #damping
        Yv = self.hydrodynamics_data.linear_damping.Yv
        Yr = self.hydrodynamics_data.linear_damping.Yr
        Nv = self.hydrodynamics_data.linear_damping.Nv
        Nr = self.hydrodynamics_data.linear_damping.Nr
        Xu = self.hydrodynamics_data.linear_damping.Xu
        #added mass
        Yvdot = self.hydrodynamics_data.added_mass.Yvdot
        Yrdot = self.hydrodynamics_data.added_mass.Yrdot
        Nvdot = self.hydrodynamics_data.added_mass.Nvdot
        Nrdot = self.hydrodynamics_data.added_mass.Nrdot
        Xudot = self.hydrodynamics_data.added_mass.Xudot


        RigidBodyMassMatrix = np.array([[mass,0,0],
                               [0,mass,mass*LCG],
                               [0,mass*LCG,Iz]])

        AddedMassMatrix = -np.array([[Xudot,0,0],
                                    [0,Yvdot,Yrdot],
                                    [0,Nvdot,Nrdot]])

        DampingMatrix = -np.array( [[Xu,0,0],
                                    [0,Yv,Yr],
                                    [0,Nv,Nr]])

        #propulsion
        TotalThrust = np.zeros(3)
        TotalMoment = np.zeros(3)

        for propulsor in self.propulsion_data.propellers:
            Throttle = ThrottleCommand[propulsor.id]
            RudderAngle = propulsor.max_angle * RudderCommand[propulsor.id]

            n = propulsor.max_rpm / 60 * Throttle
            eps = 1e-6

            AdvanceVelocity = self.Velocity[0]*(1-propulsor.wake_fraction)
            if abs(n) > eps:
                J = abs(AdvanceVelocity)/(abs(n) * propulsor.D)
            else:
                J = 0

            KT = np.interp(J, propulsor.kt_table["J"], propulsor.kt_table["Kt"])
            KQ = np.interp(J, propulsor.kt_table["J"], propulsor.kq_table["Kq"])

            ThrustProduced = Density * n * abs(n) * propulsor.D ** 4 * KT
            Thrust = np.array([ float(ThrustProduced * (1 - propulsor.thrust_deduction) * np.cos(np.deg2rad(RudderAngle))) , float(ThrustProduced * (1 - propulsor.thrust_deduction) * np.sin(np.deg2rad(RudderAngle))), 0.0])
            Moment = np.cross(np.asarray(propulsor.position) - np.asarray(self.vessel_data.mass.CG), Thrust)
            
            TotalThrust += Thrust
            TotalMoment += Moment
        
        #physics
        tau = np.array([TotalThrust[0],
                   TotalThrust[1],
                   TotalMoment[2]]).reshape(3,1)
   
        RigidBodyCoriolisMatrix = np.array( [[0,0,-mass*(LCG*self.Velocity[2,0]+self.Velocity[1,0])],
                                            [0,0,mass*self.Velocity[0,0]],
                                            [mass*(LCG*self.Velocity[2,0]+self.Velocity[1,0]),-mass*self.Velocity[0,0],0]])
        
        AddedMassCoriolisMatrix = np.array( [[0,0,-Yvdot*self.Velocity[1,0]-Yrdot*self.Velocity[2,0]],
                                            [0,0,-Xudot*self.Velocity[0,0]],
                                            [-(-Yvdot*self.Velocity[1,0]-Yrdot*self.Velocity[2,0]),Xudot*self.Velocity[0,0],0]])
        
        #Rotation Matrix
        R = np.array(  [[np.cos(self.Position[2,0]),-np.sin(self.Position[2,0]),0],
                        [np.sin(self.Position[2,0]),np.cos(self.Position[2,0]),0],
                        [0,0,1]])
        #Calculate Local Acceleration and Velocity
        NewLocalAcceleration = np.linalg.inv(RigidBodyMassMatrix+AddedMassMatrix) @ (tau - ((RigidBodyCoriolisMatrix+AddedMassCoriolisMatrix+DampingMatrix)@self.Velocity))
        NewLocalVelocity = self.Velocity + NewLocalAcceleration*dt

        NewGlobalPosition = self.Position + (R @ NewLocalVelocity)*dt
        NewGlobalAcceleration = R @ NewLocalAcceleration
        NewGlobalVelocity = R @ NewLocalVelocity

        self.Position = NewGlobalPosition
        self.Velocity = NewLocalVelocity
        self.Acceleration = NewLocalAcceleration

        #return  NewGlobalPosition, NewLocalVelocity, NewLocalAcceleration, NewGlobalVelocity, NewGlobalAcceleration



    def Update(self, ThrottleCommands, RudderCommands, dt, Density):
        self.UniversalManouvringModel(ThrottleCommands, RudderCommands, dt, Density)
        
        self.BowPos, self.SternPos = ShipPos(self.Position, self.vessel_data.geometry.LOA, self.vessel_data.mass.CG[0])
        #self.outline.set_data((self.Position[1],self.BowPos[1],self.SternPos[1]), (self.Position[0],self.BowPos[0],self.SternPos[0]))


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

def WAMVModel(Position,Velocity,Acceleration,dt,Density,Length,LCG,mass,Iz, PortThrottleCommand, StbdThrottleCommand, PortRudderCommand, StbdRudderCommand):
    #velocity magnitude
    NetVelocity = max(0.1,np.sqrt(Velocity[0,0]**2+Velocity[1,0]**2))


    #Convert Non-dimensional to dimensional
    Yv = -40#YvPrime * 0.5 * Density * NetVelocity * Length ** 2
    Yr = 0#YrPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nv = 0#NvPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nr = -400#NrPrime * 0.5 * Density * NetVelocity * Length ** 4
    Xu = -51.3#XuPrime * 0.5 * Density * NetVelocity * Length ** 2

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

    CPPort = np.array([-2.5, -1.25, 0.1])       # rudder 2.5 m aft, 0.1 m below CG
    CPStbd = np.array([-2.5, 1.25, 0.1])       # rudder 2.5 m aft, 0.1 m below CG
    CG = np.array([0.0, 0.0, 0.0])  #CG

    #Propulsion

    MaxiumThrusterAngle = 30
    PortAngle = PortRudderCommand * np.deg2rad(MaxiumThrusterAngle)
    StbdAngle = StbdRudderCommand * np.deg2rad(MaxiumThrusterAngle)

    MaximumThrust = 150 #N
    PortThrust = np.array([PortThrottleCommand * MaximumThrust * np.cos(PortAngle), PortThrottleCommand * MaximumThrust * np.sin(PortAngle), 0])
    StbdThrust = np.array([StbdThrottleCommand * MaximumThrust * np.cos(StbdAngle), StbdThrottleCommand * MaximumThrust * np.sin(StbdAngle), 0])
    TotalThrust = PortThrust + StbdThrust

    PortThrusterMoment = np.cross(CPPort-CG,PortThrust)
    StbdThrusterMoment = np.cross(CPStbd-CG,StbdThrust)
    TotalThrustMoment = PortThrusterMoment + StbdThrusterMoment

    #print("Thrust (KN)" + str(Thrust/1000) + "Rudder Yaw Moment (KNm)" + str(rudder_moments[0,2]/1000))
    tau = np.array([TotalThrust[0],
                   TotalThrust[1],
                   TotalThrustMoment[2]]).reshape(3,1)
   
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
    


def Model(Position,Velocity,Acceleration,dt,Density,Length,LCG,mass,Iz,YvPrime,YrPrime,NvPrime,NrPrime, XuPrimeFwd, XuPrimeReverse, CL, Ar, delta_attack, delta_stall, Throttle):
    #velocity magnitude
    NetVelocity = max(0.1,np.sqrt(Velocity[0,0]**2+Velocity[1,0]**2))

    if Velocity[0] >= 0:
        XuPrime = XuPrimeFwd
    else:
        XuPrime = XuPrimeReverse

    #Convert Non-dimensional to dimensional
    Yv = -40#YvPrime * 0.5 * Density * NetVelocity * Length ** 2
    Yr = 0#YrPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nv = 0#NvPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nr = -400#NrPrime * 0.5 * Density * NetVelocity * Length ** 4
    Xu = -51.3#XuPrime * 0.5 * Density * NetVelocity * Length ** 2

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

    CP = np.array([-2.5, 0.0, 0.1])       # rudder 10 m aft, 1 m below CG
    CG = np.array([0.0, 0.0, 0.0])

    #Propulsion
    MaxPropRPM = 1000
    n = MaxPropRPM/60 * Throttle
    eps = 1e-6

    J_data = [0,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,	0.66,	0.67,	0.68,	0.72,	0.73,	0.74,	0.75,	0.76,	0.77,	0.78,	0.79,	0.80]
    KT_data = [0.2418, 0.2418,	0.2368,	0.2317,	0.2266,	0.2214,	0.2162,	0.2110,	0.2057,	0.2004,	0.1951,	0.1897,	0.1844,	0.1789,	0.1735,	0.1680,	0.1624,	0.1569,	0.1513,	0.1457,	0.1400,	0.1343,	0.1286,	0.1053,	0.0994,	0.0934,	0.0874,	0.0814,	0.0754,	0.0693,	0.0632,	0.0570]
    KQ_data = [0.0318, 0.0318,	0.0313,	0.0308,	0.0303,	0.0298,	0.0292,	0.0287,	0.0281,	0.0276,	0.0270,	0.0264,	0.0258,	0.0252,	0.0246,	0.0240,	0.0234,	0.0227,	0.0221,	0.0214,	0.0207,	0.0201,	0.0194,	0.0165,	0.0158,	0.0150,	0.0143,	0.0135,	0.0127,	0.0119,	0.0111,	0.0103]

    wakeFraction = 0.1
    thrustDeductionFactor = 0.3
    prop_diameter = 0.3

    AdvanceVelocity = Velocity[0]*(1-wakeFraction)
    if abs(n) > eps:
        J = abs(AdvanceVelocity)/(abs(n) * prop_diameter)
    else:
        J = 0
    
    #clip J to available data range
    J = np.clip(J, J_data[0], J_data[-1])

    KT = np.interp(J, J_data, KT_data)
    KQ = np.interp(J, J_data, KQ_data)

    ThrustProduced = Density * n * abs(n) * prop_diameter ** 4 * KT
    Thrust = ThrustProduced * (1 - thrustDeductionFactor)

    #print(Thrust)

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

def LegacyModel(Position,Velocity,Acceleration,dt,Density,Length,LCG,mass,Iz,YvPrime,YrPrime,NvPrime,NrPrime, XuPrimeFwd, XuPrimeReverse, CL, Ar, delta_attack, delta_stall, Throttle):
    #velocity magnitude
    NetVelocity = max(0.1,np.sqrt(Velocity[0,0]**2+Velocity[1,0]**2))

    if Velocity[0] >= 0:
        XuPrime = XuPrimeFwd
    else:
        XuPrime = XuPrimeReverse

    #Convert Non-dimensional to dimensional
    Yv = -40#YvPrime * 0.5 * Density * NetVelocity * Length ** 2
    Yr = 0#YrPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nv = 0#NvPrime * 0.5 * Density * NetVelocity * Length ** 3
    Nr = -400#NrPrime * 0.5 * Density * NetVelocity * Length ** 4
    Xu = -51.3#XuPrime * 0.5 * Density * NetVelocity * Length ** 2

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

    CP = np.array([-2.5, 0.0, 0.1])       # rudder 10 m aft, 1 m below CG
    CG = np.array([0.0, 0.0, 0.0])

    #Propulsion
    MaxPropRPM = 1000
    n = MaxPropRPM/60 * Throttle
    eps = 1e-6

    J_data = [0,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,	0.66,	0.67,	0.68,	0.72,	0.73,	0.74,	0.75,	0.76,	0.77,	0.78,	0.79,	0.80]
    KT_data = [0.2418, 0.2418,	0.2368,	0.2317,	0.2266,	0.2214,	0.2162,	0.2110,	0.2057,	0.2004,	0.1951,	0.1897,	0.1844,	0.1789,	0.1735,	0.1680,	0.1624,	0.1569,	0.1513,	0.1457,	0.1400,	0.1343,	0.1286,	0.1053,	0.0994,	0.0934,	0.0874,	0.0814,	0.0754,	0.0693,	0.0632,	0.0570]
    KQ_data = [0.0318, 0.0318,	0.0313,	0.0308,	0.0303,	0.0298,	0.0292,	0.0287,	0.0281,	0.0276,	0.0270,	0.0264,	0.0258,	0.0252,	0.0246,	0.0240,	0.0234,	0.0227,	0.0221,	0.0214,	0.0207,	0.0201,	0.0194,	0.0165,	0.0158,	0.0150,	0.0143,	0.0135,	0.0127,	0.0119,	0.0111,	0.0103]

    wakeFraction = 0.1
    thrustDeductionFactor = 0.3
    prop_diameter = 0.3

    AdvanceVelocity = Velocity[0]*(1-wakeFraction)
    if abs(n) > eps:
        J = abs(AdvanceVelocity)/(abs(n) * prop_diameter)
    else:
        J = 0
    
    #clip J to available data range
    J = np.clip(J, J_data[0], J_data[-1])

    KT = np.interp(J, J_data, KT_data)
    KQ = np.interp(J, J_data, KQ_data)

    ThrustProduced = Density * n * abs(n) * prop_diameter ** 4 * KT
    Thrust = ThrustProduced * (1 - thrustDeductionFactor)

    #print(Thrust)

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

### observations
def DPObservations(self):
    observation = np.array([
            float(np.clip(self.DistanceToWaypoint / float(self.MaxDistance), -1, 1)),
            float(self.VelocityToWaypoint / float(self.MaxSpeed)),
            float(np.sin(self.HeadingToWaypointError)),
            float(np.cos(self.HeadingToWaypointError)),
            float(np.sin(self.DesiredHeadingError)),
            float(np.cos(self.DesiredHeadingError)),
            self.vessel.Throttle,
            self.vessel.Rudder_Angle
        ], dtype=np.float32)
    return observation

