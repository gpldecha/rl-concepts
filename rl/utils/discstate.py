import numpy as np
import warnings
import numbers
import types

class Discretise:
    """ Discretises multivariate continuous state array to an index. This is
        necessary when runing discrete RL methods on continuous state
        environments.
    """
    def __init__(self,bin_sizes,state_mins,state_maxs):
        """
            Args:
                bin_sizes  (list or int)           : Number of bins for each dimension of the continuous state space.
                state_mins (list or int)           : Number of minimum values for each dimension of the coninuous state space.
                state_maxs (list or int)           : Number of maximum values for each dimension of the coninuous state space.
        """
        if isinstance(bin_sizes,types.IntType):
            self._num_dim = 1
        elif isinstance(bin_sizes,list):
            assert len(bin_sizes) == len(state_mins) == len(state_maxs)
            self._num_dim  = len(bin_sizes)
        else:
            warnings.warn("all inputs should be of either int or np.ndarray your input is " + type(bin_sizes), Warning)

        self.bin_sizes = bin_sizes
        self.state_mins = state_mins
        self.state_maxs = state_maxs

    def bound_value(self,value,vmin,vmax):
        if value < vmin:
            return vmin
        elif value > vmax:
            return vmax
        else:
            return value

    def numtoint(self,state):
        """ Converts a continuous 1D number to a discrete state
            Args:
                state (Number) : Current state value
            Returns:
                (int) : Discretised state index.
            Comments:
        """
        assert isinstance(state,numbers.Number)
        state = self.bound_value(float(state),self.state_mins,self.state_maxs)
        return int(round((state - self.state_mins)/(self.state_maxs - self.state_mins) * self.bin_sizes, 0))

    def num1Dlist2int(self,states):
        """ Converts a continuous list of 1D numbers to a list of discrete states
                Args:
                    states list(Number) : list of continuous tate values
                Returns:
                    (int) : Discretised state index.
                Comments:
        """
        assert isinstance(states,list)
        return [ self.numtoint(s) for s in states ]

    def vecNDtoint(self, state):
        """ Converts a continuous state vector to a discrete state integer
                Args:
                    state (numpy.ndarray) : Current state of the environment.
                                            numpy.ndarray((dims,))
                Returns:
                        (int) : Discretised state index.
                Comments:
        """
        assert isinstance(state, np.ndarray)

        idx = 0
        state_j = 0
        for j in range(0,dims-1): # j is index of dimensions
            state_j = self.bound_value(state[j],self.state_mins[j],self.state_maxs[j])
            idx   = idx + np.prod(self._N[j+1:]) *  round( (state_j - self.state_mins[j])/(self.state_maxs[j] - self.state_mins[j]) * self.bin_sizes[j] , 0 )
        state_j = self.bound_value(state[-1],self.state_mins[-1],self.state_maxs[-1])
        idx = idx +  round( (state_j - self.state_mins[-1])/(self.state_maxs[-1] - self.state_mins[-1]) * self.bin_sizes[-1], 0)
        return idx


    def vecNDtoint(self,states):
        """ Converts a continuous list of vector states to a discrete state
                Args:
                    states list(numpy.ndarray) : Current state of the environment.
                                                 numpy.ndarray((num_points,dims))

                Returns:
                        (int) : Discretised state index.

                Comments:
                        rescales each dimension to the interval [0,]

        """
        assert isinstance(state,np.ndarray)
        return [ self.vecNDtoint(s) for s in states ]

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
                idx_i = idx_i + np.prod(self._N[j+1:]) * np.digitize(state[i,j], self.bins[j])
            idx_i = idx_i + np.digitize(state[i,-1], self.bins[-1])
            idx.append(idx_i)

        return idx
