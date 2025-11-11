from manoeuvringModelLibrary import *
import matplotlib.pyplot as plt

#plot vessel as manoeuvring model and later as a real vessel
#take user inputs for actions
#deploy RL models to undertake actions

dt = 1/20
time = 0
#Environmental Conditions
Density = 1025

# create a plot
fig, ax = plt.subplots()
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

#Define Vessels
Position = np.array([0,0,0])
Velocity = np.array([0,0,0])
Acceleration = np.array([0,0,0])

Length = 10
Beam = 3
Draft = 1
CB = 0.75
lcg = 0
kz = Length / 4
XuPrimeFwd = -0.003
XuPrimeReverse = 0.01

vessels = np.array([vessel_class(ax, "Vessel 1", 0, 0, Position, Velocity, Acceleration, Density, Length, Beam, Draft, CB, lcg, kz, XuPrimeFwd, XuPrimeReverse)])
    

#update
while True:
    for vessel in range(vessels):
        vessel.Update(dt, Density)
    time += dt
    plt.pause(dt)
