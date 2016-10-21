import numpy as np


class DiscretiseState:
    """ Discretises multivariate continuous state array to an index. This is
        necessary when runing discrete RL methods on continuous state
        environments.
    """

    def __init__(self,state_bins,state_mins,state_maxs):
        """
            Args:
                state_bins (numpy.ndarray)  : Number of bins for each dimension of the continuous state space.
                state_mins (list)           : Number of minimum values for each dimension of the coninuous state space.
                state_maxs (list)           : Number of maximum values for each dimension of the coninuous state space.
        """
        assert state_bins.size == len(state_mins) == len(state_maxs)
        self._num_dim  = len(state_bins)
        self._N        = state_bins
        self.bins      = []

        for i in range(0,self._num_dim):
            _num = state_bins[i]
            _min = state_mins[i]
            _max = state_maxs[i]
            #print 'dim(',i,'): _num(',_num,')  _min(',_min,')  _max(',_max,')'

            self.bins.append(np.linspace(_min,_max,_num))

            print self.bins[0]



    def toint(self,state):
        """ Converts a continuous state to a discrete state
            Args:
                state (numpy.ndarray) : Current state of the environment.
                                        numpy.ndarray((num_points,dims))

            Returns:
                    (int) : Discretised state index.
        """

        shape       = state.shape
        dims        = None
        num_points  = None
        idx         = []

        if len(shape) == 2:
            dims        = shape[1]
            num_points  = shape[0]
        else:
            return idx

        #print 'dims: ', dims, ' self._num_dim: ', self._num_dim

        assert self._num_dim == dims

        print 'state.ndim == 2'
        print state

        for i in range(0,num_points):
            idx_i = 0
            for k in range(0,dims-1):
                print 'state[', i , ',' , k ,'] '
                    idx_i = idx_i + np.prod(self._N[k+1:-1]) * np.digitize(state[i,k], self.bins[k])
            idx_i = idx_i + np.digitize(state[i,-1], self.bins[-1])
            idx.append(idx_i)

        return idx
