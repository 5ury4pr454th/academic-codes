import matplotlib.pyplot as plt
from data_plotter import subplotWindow

class responsePlotter:
    def __init__(self) -> None:
        """For plotting reference, input and output signals"""
        self.num_rows = 3
        self.num_columns = 1
        self.fig, self.ax = plt.subplots(nrows=self.num_rows, ncols=self.num_columns,
                                         sharex=True, figsize = (10,30))

        # recording history
        self.time_history = []
        self.h_history = []
        self.h_r_history = []
        self.F_history = []
        self.handle = []

        # creating subplots in plot window
        self.handle.append(subplotWindow(self.ax[0], ylabel='H_r', title='Reference Signal'))
        self.handle.append(subplotWindow(self.ax[1], ylabel='H', title='Output Signal'))
        self.handle.append(subplotWindow(self.ax[2], ylabel='F', title='Input Signal'))

    def update(self, time, state, ref, control):
        """updates the plots"""
        self.time_history.append(time)
        self.h_history.append(state)
        self.h_r_history.append(ref)
        self.F_history.append(control)
       
        self.handle[0].update(self.time_history, [self.h_r_history])
        self.handle[1].update(self.time_history, [self.h_history])
        self.handle[2].update(self.time_history, [self.F_history])