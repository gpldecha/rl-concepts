import numpy as np
import matplotlib.pyplot as plt

 def plot_policy_1D(ax,Q):



class PlotValueFunction1D:

    def __init__(self,states,values):
        self._states = states

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.xlabel(r'\textbf{state} (x)')
        plt.ylabel(r'V(s)',fontsize=16)

        print 'states: ', states
        print 'values: ', values

        line, = self.ax.plot(states, values, 'r-')

        self.line = line

        plt.show()

    def update(self,values):
        self.line.set_ydata(values)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
