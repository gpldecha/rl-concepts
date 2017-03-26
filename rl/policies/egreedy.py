import numpy as np

class Egreedy:

    def __init__(self,Q,epsilon=0.5):
        """ epsilon-greedy action-selection policy
            Args:
                Q (numpy.ndarray) : Q-value table (states,actions)
                temp (int) : temperature
        """
        if len(Q.shape) == 2:
            self._num_states  = Q.shape[0]
            self._num_actions = Q.shape[1]
        else:
            raise ValueError('Q is not a 2-dimensional numpy.ndarray')
        self._Q = Q
        self.epsilon = epsilon
        self._action_idx = range(0,self._num_actions)


    def action(self,state):
        """ Returns max action if rand > epsilon
            Args:
                state (int) : state index (row of Q).

            Return:
                    (int) : action index (column of Q).
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self._num_actions,)
        else:
            return np.argmax(self._Q[state,:])
