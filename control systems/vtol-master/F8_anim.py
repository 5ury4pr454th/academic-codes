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
reference_signal = SignalGenerator(amplitude=5, frequency=0.1, y_offset=3)
reference_signal_2 = SignalGenerator(amplitude=3, frequency=0.08)
disturbance = SignalGenerator(amplitude = 0.00)
disturbance_2 = SignalGenerator(amplitude=0.25)

# creating dynamics and altitude and lateral controllers for question F8
dynamics = systemDynamics(interim_state, p.t_step)
alt_control = AltitudeController()
lat_control = lateralController()

# drawing animation and plotting reference, input and output
plot_engine = responsePlotter()
anim_engine = vtolAnimation()

# setting start time and getting initial state
t = p.t_start
init_state = dynamics.h()
print(f"Initial coordinates: {init_state}")

while t < p.t_end:

    # setting next plotting time
    t_next_plot = t + p.t_plot

    # running dynamics between plotting times
    while t < t_next_plot:

        # defining signal functions
        r = reference_signal.step(t)
        r2 = reference_signal_2.square(t)
        d = disturbance.sin(t)
        d2 = disturbance_2.sin(t)
        r += d
        r2 += d2
        
        # get vertical and horizontal coordinates
        z_state = np.array([dynamics.state[0][0], dynamics.state[1][0]])
        theta_state = np.array([dynamics.state[0][2], dynamics.state[1][2]])
        h_state = np.array([dynamics.state[0][1], dynamics.state[1][1]])
        
        # update both the controllers controller
        u_lat = lat_control.update(r2, z_state, theta_state)
        u_alt = alt_control.update(r, h_state)

        # for converting to fl and fr
        fl = (u_alt - u_lat/p._d_br)/2
        fr = (u_alt + u_lat/p._d_br)/2

        # saturating fl and fr
        fl = 10.00 if fl > 10.00 else fl
        fr = 10.00 if fr > 10.00 else fr
        fl = 0.00 if fl < 0.00 else fl
        fr = 0.00 if fr < 0.00 else fr

        # Note that in the beggining there is atleast one instance of overshoot

        # updating the dynamics of the system, using controller outputs
        y = dynamics.update(p._u_c(dynamics.state[0][2])*np.array([fl + fr, fl + fr, fr-fl]))

        # going for next time step
        t = t + p.t_step

    # update animation and data plots for lateral displacement
    anim_engine.update(dynamics.state[0][0], dynamics.state[0][1], dynamics.state[0][2])
    plot_engine.update(t, dynamics.state[0][0], r2, u_lat)
    
    # to pause in the middle for better visibility
    plt.pause(0.0005)

final_state = dynamics.h()
print(f"Final coordinates: {final_state}")

plt.pause(2)