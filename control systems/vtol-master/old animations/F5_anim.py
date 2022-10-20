# the main script to execute while writing the code

# experiment
from system_dynamics import systemDynamics
import numpy as np
import matplotlib.pyplot as plt
import specs as p
from signal_gen import SignalGenerator
from vtolAnimation import vtolAnimation
from data_plotter import dataPlotter, subplotWindow
from time import time

plt.ion()

# test animation
anim = vtolAnimation()
#anim.draw_drone(p.dr_mat_i) # init position, h above centre


t = p.t_start

fr = SignalGenerator(amplitude = 15.153, frequency=0.1)
fl = SignalGenerator(amplitude = 15.653, frequency=0.1)
interim_state = [[0, p._w_b/2, 0], [0, 0, 0]]
drone_sys = systemDynamics(interim_state, p.t_step)
plotter = dataPlotter(only_0=True)
plt.pause(10)
while t < p.t_end:
    
    if t < 0.2:
        final_u = p._u_c(drone_sys.state[0][2])*np.array([fl.square(t) + fr.square(t) , fl.square(t) + fr.square(t), fr.square(t) - fl.square(t)])
    elif t < 0.5: 
        fl.amplitude = 15.153
        fr.amplitude = 15.653
        final_u = p._u_c(drone_sys.state[0][2])*np.array([fl.square(t) + fr.square(t), fl.square(t) + fr.square(t), fr.square(t) - fl.square(t)])
    elif (interim_state[2] > -0.02) and (interim_state[2] < -0.01):
        fl.amplitude = 0
        fr.amplitude = 0
        final_u = p._u_c(drone_sys.state[0][2])*np.array([fl.square(t) + fr.square(t), fl.square(t) + fr.square(t), fr.square(t) - fl.square(t)])
    
    
    interim_state = drone_sys.update(final_u) 
    anim.update(interim_state[0], interim_state[1], interim_state[2])
    plotter.update(t, drone_sys.state, [fr.square(t), fl.square(t)])
    

    anim.fig.canvas.draw()
    anim.fig.canvas.update()
    anim.fig.canvas.flush_events()
    
    t += p.t_step

plt.pause(1)
