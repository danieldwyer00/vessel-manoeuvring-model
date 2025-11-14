import sys, math
import numpy as np

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from manoeuvringModelLibrary import *

class VesselSim(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vessel Manoeuvring Demo")

        # State
        self.dt = 1/60
        time = 0
        #Environmental Conditions
        self.Density = 1025

        #Define Vessels
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

        Length = 10
        Beam = 3
        Draft = 1
        CB = 0.75
        lcg = 0
        kz = Length / 4
        XuPrimeFwd = -0.003
        XuPrimeReverse = 0.01

        Throttle = 100
 
        self.vessels = np.array([vessel_class("Vessel 1", 0, Throttle, Position, Velocity, Acceleration, self.Density, Length, Beam, Draft, CB, lcg, kz, XuPrimeFwd, XuPrimeReverse)])
            

        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # Controls (sliders)
        controls = QtWidgets.QVBoxLayout()
        layout.addLayout(controls, 0, 0)

        self.throttle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.throttle_slider.setRange(-100, 100)
        self.throttle_slider.setValue(0)
        self.throttle_slider.setTickInterval(10)
        self.throttle_slider.valueChanged.connect(self.on_throttle)
        controls.addWidget(QtWidgets.QLabel("Throttle"))
        controls.addWidget(self.throttle_slider,alignment=Qt.AlignmentFlag.AlignHCenter)

        self.rudder_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rudder_slider.setRange(-100, 100)
        self.rudder_slider.setValue(0)
        self.rudder_slider.setTickInterval(10)
        self.rudder_slider.valueChanged.connect(self.on_rudder)
        self.rudder_slider.setMaximumWidth(300)
        controls.addWidget(QtWidgets.QLabel("Rudder (deg)"))
        controls.addWidget(self.rudder_slider)

        # Matplotlib canvas
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 0, 1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-50, 50)
        self.ax.grid(True)

        # Vessel geometry (simple rectangle + bow/stern markers)
        x = np.array([ -self.vessels[0].Length/2,  self.vessels[0].Length/2,  self.vessels[0].Length/2, -self.vessels[0].Length/2, -self.vessels[0].Length/2 ])
        y = np.array([ -self.vessels[0].Beam/2, -self.vessels[0].Beam/2,  self.vessels[0].Beam/2,  self.vessels[0].Beam/2, -self.vessels[0].Beam/2 ])
        self.hull_xy = np.vstack((x, y))
        (self.hull_line,) = self.ax.plot([], [], "b-")
        (self.bow_pt,) = self.ax.plot([], [], "ro")
        (self.stern_pt,) = self.ax.plot([], [], "ko")

        # Timer for updates
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(int(self.dt * 1000))

    def on_throttle(self, val):
        self.vessels[0].Throttle = val / 100

    def on_rudder(self, val):
        self.vessels[0].Rudder_Angle = -float(val) / 100

    def step(self):
        for vessel in self.vessels:
            vessel.Update(self.dt, self.Density)
        
        self.update_plot()

    def update_plot(self):
        R = np.array([[np.cos(-self.vessels[0].Position[2,0] + np.pi/2), -np.sin(-self.vessels[0].Position[2,0] + np.pi/2)],
                  [np.sin(-self.vessels[0].Position[2,0] + np.pi/2),  np.cos(-self.vessels[0].Position[2,0] + np.pi/2)]])
        
        xy = R @ self.hull_xy + np.array([self.vessels[0].Position[1,0],self.vessels[0].Position[0,0]])[:,None]

        self.hull_line.set_data(np.asarray(xy[0]).ravel(), np.asarray(xy[1]).ravel())

        self.vessels[0].BowPos, self.vessels[0].SternPos = ShipPos(self.vessels[0].Position,self.vessels[0].Length,self.vessels[0].lcg)

        self.bow_pt.set_data([self.vessels[0].BowPos[1]],   [self.vessels[0].BowPos[0]])    
        self.stern_pt.set_data([self.vessels[0].SternPos[1]],   [self.vessels[0].SternPos[0]])
    
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = VesselSim()
    w.resize(1920, 1080)
    w.show()
    sys.exit(app.exec())