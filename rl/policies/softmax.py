import numpy as np

def exp_softmax(x):
    """ Exponential softmax (numerically stable)
        Args:
            x (numpy.ndarray) : 1D vector.
        Returns:
            (numpy.ndarry) : softmax values of elements of x.
    """
    rebase_x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def log_softmax(q):
    """Logaritmic softmax (numerically stable)
        Args:
            q (numpy.ndarray) : 1D vector
        Returns:
            logarithmic softmax
    """
    max_q = max(0.0, np.max(q))
    rebased_q = q - max_q
    return rebased_q - np.logaddexp(-max_q, np.logaddexp.reduce(rebased_q))

class SoftMax:

    def __init__(self,Q,temp=1000):
        """ Softmax action-selection according to a Boltzmann/Gibbs distribution.
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
        self.temp = temp
        self._P = np.ones(self._num_actions) / self._num_actions
        self._action_idx = range(0,self._num_actions)


    def action(self,state):
        """ Returns an action sampled from a Gibbs probability distribution
            Args:
                state (int) : state index (row of Q).

            Return:
                    (int) : action index (column of Q).
        """
        return np.random.choice(self._action_idx,1,p=list(exp_softmax(self._Q[state,:]/self.temp)))[0]
