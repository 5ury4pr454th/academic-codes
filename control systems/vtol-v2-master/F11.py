import specs as p
import numpy as np
import matplotlib.pyplot as plt
from f7plot import responsePlotter
from data_plotter import dataPlotter
from signal_gen import SignalGenerator
from system_dynamics import systemDynamics
from st_fb_control import lateralController, altitudeController
import state_specs as sp
import control_lib as ctrlb
from vtolAnimation import vtolAnimation

# after tuning, this was the rise time, that makes sure that there is no input saturation
t_rise_alt = 8.693
t_rise_lat = 8.693

K_alt, kr_alt = ctrlb.gains_from_time(t_rise_alt)
K_lat, kr_lat = ctrlb.gains_from_time(t_rise_lat, alt = False)

# initial state
interim_state = [[0, p._w_b/2, 0], [0, 0, 0]]

# creating signal objects according to F8
reference_signal = SignalGenerator(amplitude=5, frequency=0.1, y_offset=0)
reference_signal_2 = SignalGenerator(amplitude=5, frequency=0.08)
disturbance = SignalGenerator(amplitude = 0.00)
disturbance_2 = SignalGenerator(amplitude=0.00)

# creating dynamics and altitude and lateral controllers for question F8
dynamics = systemDynamics(interim_state, p.t_step)
alt_control = altitudeController(K_alt, kr_alt)
lat_control = lateralController(K_lat, kr_lat)

# drawing animation and plotting reference, input and output
plot_engine = responsePlotter()
anim_engine = vtolAnimation()

# setting start time and getting initial state
t = p.t_start
init_state = dynamics.h()
print(f"Initial coordinates: {init_state}")

F_wind = 0.00
while t < p.t_end:

    # setting next plotting time
    t_next_plot = t + p.t_plot

    # running dynamics between plotting times
    while t < t_next_plot:

        # defining signal functions
        r = reference_signal.step(t)
        r2 = reference_signal_2.step(t)
        d = disturbance.sin(t)
        d2 = disturbance_2.sin(t)
        r += d
        r2 += d2
        
        # get vertical and horizontal coordinates
        z_state = np.array([dynamics.state[0][0], dynamics.state[1][0]])
        theta_state = np.array([dynamics.state[0][2], dynamics.state[1][2]])
        h_state = np.array([dynamics.state[0][1], dynamics.state[1][1]])
        
        # update both the controllers controller
        u_alt = alt_control.update(r, h_state)
        u_lat = lat_control.update(r2, z_state, theta_state)

        # for converting to fl and fr
        fl = (u_alt - u_lat/p._d_br)/2
        fr = (u_alt + u_lat/p._d_br)/2

        # saturating fl and fr

        # assert fl <= 10,"Left Force exceeded max"
        
        # assert fl >= 0,"Left Force exceeded min"
        
        # assert fr <= 10,"Right Force exceeded max"
        
        # assert fr >= 0,"Right Force exceeded max"

        # print(fl, fr)

        # fl = 10.00 if fl > 10.00 else fl
        # fr = 10.00 if fr > 10.00 else fr
        # fl = 0.00 if fl < 0.00 else fl
        # fr = 0.00 if fr < 0.00 else fr

        # updating the dynamics of the system, using controller outputs
        y = dynamics.update(p._u_c(dynamics.state[0][2])*np.array([fl + fr, fl + fr, fr-fl]) + [0, 0, F_wind])

        # going for next time step
        t = t + p.t_step

    # update animation and data plots for altitude displacement
    plot_engine.update(t, dynamics.state[0][0], r2, u_lat)
    anim_engine.update(dynamics.state[0][0], dynamics.state[0][1], dynamics.state[0][2])

    # to pause in the middle for better visibility
    plt.pause(0.0005)

final_state = dynamics.h()
print(f"Final coordinates: {final_state}")

plt.pause(2)