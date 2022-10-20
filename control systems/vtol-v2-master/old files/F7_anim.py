import specs as p
import numpy as np
import matplotlib.pyplot as plt
from f7plot import responsePlotter
from data_plotter import dataPlotter
from signal_gen import SignalGenerator
from vtolAnimation import vtolAnimation
from system_dynamics import systemDynamics
from altitude_control import AltitudeController

# initial state
interim_state = [[0, p._w_b/2, 0], [0, 0, 0]]

# creating signal objects
reference_signal = SignalGenerator(amplitude=1)
disturbance = SignalGenerator(amplitude=0.0)

# creating dynamics and altitude controller for question F7
dynamics = systemDynamics(interim_state, p.t_step)
alt_control = AltitudeController(f7 = True)

# drawing animation and plotting reference, input and output signals
plot_engine = responsePlotter()
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
        r = reference_signal.step(t)
        d = disturbance.random(t)
        r += d
        
        # get height state
        h_state = np.array([dynamics.state[0][1], dynamics.state[1][1]])
        
        # updating controller
        u = alt_control.update(r, h_state)

        # updating the dynamics of the system, using controller output
        y = dynamics.update(p._u_c(dynamics.state[0][2])*np.array([u, u, 0]))

        # going for next time step
        t = t + p.t_step

    # update animation and data plots
    anim_engine.update(dynamics.state[0][0], dynamics.state[0][1], dynamics.state[0][2])
    plot_engine.update(t, dynamics.state[0][1], r, u)
    
    # to pause in the middle for better visibility
    plt.pause(0.0005)

final_state = dynamics.h()
print(f"Final coordinates: {final_state}")

plt.pause(2)