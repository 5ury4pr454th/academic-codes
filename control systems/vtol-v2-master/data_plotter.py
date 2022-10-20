import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.ion()

class dataPlotter:
    def __init__(self, only_0 = False) -> None:
        
        # plot initializing
        self.less = only_0
        self.num_rows = 2                   ############
        self.num_columns = 4                ############

        if only_0 == True:
            self.num_columns = 5            ############
            self.num_rows = 1               ############

        self.fig, self.ax = plt.subplots(nrows=self.num_rows, ncols=self.num_columns, sharex=True, figsize = (10,10))
        
        # recording history

        self.time_history = []
        self.z_history = []
        self.h_history = []
        self.t_history = []
        self.fl_history = []
        self.fr_history = []
        self.handle = []
        
        
        if only_0 == False:
            self.z_1_history = []
            self.h_1_history = []
            self.t_1_history = []

        self.handle.append(subplotWindow(self.ax[0], ylabel='z', title='Displacement'))
        self.handle.append(subplotWindow(self.ax[1], ylabel='h', title='Height'))
        self.handle.append(subplotWindow(self.ax[2], ylabel='t', title='Angle'))
        self.handle.append(subplotWindow(self.ax[3], ylabel='fr', title='Right Motor Force'))
        self.handle.append(subplotWindow(self.ax[4], ylabel='fl', title='Left Motor Force'))
        
        if only_0 == False:

            self.handle.append(subplotWindow(self.ax[0, 1], ylabel='z\'', title='Horizontal velocity'))
            self.handle.append(subplotWindow(self.ax[0, 3], ylabel='h\'', title='Vertical velocity'))
            self.handle.append(subplotWindow(self.ax[1, 1], ylabel='t\'', title='Angular velocity'))

    def update(self, time, state, control):
        
        self.time_history.append(time)
        self.z_history.append(state[0][0])
        self.h_history.append(state[0][1])
        self.t_history.append(state[0][2])
        self.fr_history.append(control[0])
        self.fl_history.append(control[1])

        if self.less == False:
            self.z_1_history.append(state[1][0])
            self.h_1_history.append(state[1][1])
            self.t_1_history.append(state[1][2])
        
        self.handle[0].update(self.time_history, [self.z_history])
        self.handle[1].update(self.time_history, [self.h_history])
        self.handle[2].update(self.time_history, [self.t_history])
        self.handle[3].update(self.time_history, [self.fr_history])
        self.handle[4].update(self.time_history, [self.fl_history])

        
        if self.less == False:
            
            self.handle[1].update(self.time_history, [self.z_1_history])
            self.handle[3].update(self.time_history, [self.h_1_history])
            self.handle[5].update(self.time_history, [self.t_1_history])
            
class subplotWindow:
    
    def __init__(self, ax, xlabel='', ylabel='', title='', legend = None):
        self.legend = legend
        self.ax = ax
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        self.line_styles = ['-', '-', '--', '-.', ':']
        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        self.init = True 

    def update(self, time, data):

        if self.init == True:  
            for i in range(len(data)):
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            if self.legend != None:
                plt.legend(handles=self.line)
        else: 
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()