"""Epsilon greedy policy"""

import numpy as np


class EpsilonGreedy:

    def __init__(self,Q):
        """
            Args:
                Q (numpy.ndarray) : 2D-array (states x actions)
        """

        # check that shape is 2D
        assert len(Q.shape) == 2

        self.Q = Q
        self.k = 1.0
        self.min_epsilon = 0.1
        self.epsilon = 1.0 / self.k# start with random exploration
        self.num_actions = Q.shape[1]



    def action(self,state):
        """ Corresponding to state with maximum value
            Args:
                state (int) : State index (discrete states)
            Returns:
                (int)       : action to be applied
        """
        if np.random.rand(1)[0] < self.epsilon:
            return np.random.randint(self.num_actions,size=1)[0]
        else:
            return np.argmax(self.Q[state,:])

    def decay_learning_rate(self):
        """ Simple exploration strategy: gradually decay epsilon until
            at a sepcific threashold is reached.
        """
        self.epsilon = self.epsilon * self.k
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
