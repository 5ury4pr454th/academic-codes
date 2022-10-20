import specs as p
import numpy as np
import matplotlib.pyplot as plt
from f7plot import responsePlotter
from data_plotter import dataPlotter
from signal_gen import SignalGenerator
from vtolAnimation import vtolAnimation
from system_dynamics import systemDynamics
from lateral_control import lateralController
from altitude_control import AltitudeController

# initial state
interim_state = [[0, p._w_b/2, 0], [0, 0, 0]]

# creating signal objects according to F8
reference_signal = SignalGenerator(amplitude=3, frequency=0.08)
reference_signal_2 = SignalGenerator(amplitude=10)
disturbance = SignalGenerator(amplitude=0.25)
disturbance_2 = SignalGenerator(amplitude=0.00)
fl = fr = 0

# creating dynamics and altitude and lateral controllers for question F8
dynamics = systemDynamics(interim_state, p.t_step)
alt_control = AltitudeController(f7 = False)
lat_control = lateralController()

# drawing animation and plotting variables
plot_engine = dataPlotter(only_0 = True)
anim_engine = vtolAnimation()

# setting start time and getting initial state
t = p.t_start
init_state = dynamics.h()
print(f"Initial coordinates: {init_state}")
plt.pause(10)

while t < p.t_end:

    # setting next plotting time
    t_next_plot = t + p.t_plot

    # running dynamics between plotting times
    while t < t_next_plot:

        # defining signal functions
        r = reference_signal.square(t)
        r2 = reference_signal_2.step(t)
        d = disturbance.sin(t)
        d2 = disturbance_2.random(t)
        r += d
        r2 += d2
        
        # get vertical and horizontal coordinates
        z_state = np.array([dynamics.state[0][0], dynamics.state[1][0]])
        h_state = np.array([dynamics.state[0][1], dynamics.state[1][1]])
        
        # update both the controllers controller
        u = lat_control.update(r, z_state)
        u2 = alt_control.update(r2, h_state)

        # defining left and right motor forces from u and u2
        fl = (u2 - u)/2
        fr = (u2 + u)/2
        # should it be u/d??

        # saturate fr, fl to not exceed values
        fl = p.saturate_motors(fl)
        fr = p.saturate_motors(fr)

        # updating the dynamics of the system, using controller outputs
        y = dynamics.update(p._u_c(dynamics.state[0][2])*np.array([fr + fl, fr + fl, fr - fl]))

        # going for next time step
        t = t + p.t_step

    # update animation and data plots
    anim_engine.update(dynamics.state[0][0], dynamics.state[0][1], dynamics.state[0][2])
    plot_engine.update(t, dynamics.state, np.array([fr, fl]))

    # to pause in the middle for better visibility
    plt.pause(0.0005)
    
final_state = dynamics.h()
print(f"Final coordinates: {final_state}")

plt.pause(2)