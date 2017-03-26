import numpy as np
import matplotlib.pyplot as plt

def plot_policy_1D(ax,Q):
    """

    """
    pass


class PlotQFunction1D:

    def __init__(self,states,Q):
        self._states = states


        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.xlabel(r'\textbf{state} (x)')
        plt.ylabel(r'V(s)',fontsize=16)

        V = np.max(Q,1)

        line_v,  = self.ax.plot(states, V, 'k-')
        line_a1, = self.ax.plot(states, Q[:,0], 'r-')
        line_a2, = self.ax.plot(states, Q[:,1], 'b-')

        self.line_v = line_v
        self.line_a1 = line_a1
        self.line_a2 = line_a2

        plt.show()

    def update(self,Q):

        V = np.max(Q,1)

        self.line_v.set_ydata(V)
        self.line_a1.set_ydata(Q[:,0])
        self.line_a2.set_ydata(Q[:,1])

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
