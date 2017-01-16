import numpy as np
import warnings


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
        self._N        = state_bins+1
        self.bins      = []

        for i in range(0,self._num_dim):
            _num = state_bins[i]
            _min = state_mins[i]
            _max = state_maxs[i]
            self.bins.append(np.linspace(_min,_max,_num))


    def twoDtoint(self,state):
        """ Converts a continous 2D state to a discrete sate
            Args:
                state (numpy.ndarray) : Current state of the environment.
                                        numpy.ndarray((num_points,dims=2))
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

        assert len(self.bins) == 2

        """ idx = i_row  * num_cols * i_col """

        N_cols = self._N[1]
        for i in range(0,num_points):
            idx_i = np.digitize(state[i,0], self.bins[0]) * N_cols + np.digitize(state[i,1], self.bins[1])
            idx.append(idx_i)
        return idx


    def toint(self,state):
        """ Converts a continuous state to a discrete state
            Args:
                state (numpy.ndarray) : Current state of the environment.
                                        numpy.ndarray((num_points,dims))

            Returns:
                    (int) : Discretised state index.

            Comments:
                    Uses numpy.digitize (double) -> (int):  bin index i of x will
                    satisfy bins[i-1] <= x < bins[i]

                    If state is multivariate each dimension is bined individually
                    an a single index is computed via generic row-major formula.
        """

        if not isinstance(state,np.ndarray):
            warnings.warn("type(state) == " + type(state) + " is not of type numpy.ndarray", Warning)



        shape       = state.shape
        dims        = None
        num_points  = None
        idx         = []

        if len(shape) == 2:
            dims        = shape[1]
            num_points  = shape[0]
        else:
            return idx

        assert self._num_dim == dims

        for i in range(0,num_points): # i is index of points
            idx_i = 0
            for j in range(0,dims-1): # j is index of dimensions
                #print '(', i, ',',  j , ')   N[', j+1, ']'
                #print 'dim(',j,') -> ',  np.digitize(state[i,j], self.bins[j])
                idx_i = idx_i + np.prod(self._N[j+1:]) * np.digitize(state[i,j], self.bins[j])
            #print 'dim(1)   -> ', np.digitize(state[i,-1], self.bins[-1])
            idx_i = idx_i + np.digitize(state[i,-1], self.bins[-1])
            idx.append(idx_i)

        return idx
