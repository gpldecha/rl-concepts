import unittest

import numpy as np
from rl.utils.discstate import DiscretiseState


def testmethod():
    pass


class TestDiscretiseStateMethods(unittest.TestCase):

    def test1DdValues2Int(self):
        """ Test bininig of discrete values """
        # create a discretisation object
        bins        = np.array([11])
        mins        = [0]
        maxs        = [100]
        # bins will be:   [0.   10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]
        # bin ids        0    1     2     3    4     5     6     7     8     9     10    11
        discState   = DiscretiseState(bins,mins,maxs)

        # bins[i-1] <= x < bins[i]
        val     = np.zeros((8,1))
        val[0]  = -1  # 0 (outside min range)
        val[1]  = 9   # 1 (first bin)
        val[2]  = 20  # 3 (third bin, upper boundary is not part of the bin)
        val[3]  = 45  # 5
        val[4]  = 89  # 9
        val[5]  = 99  # 10 (just before boundary of last bin)
        val[6]  = 100 # 11 (boundary max outer bin )
        val[7]  = 110 # 11 (outside max range)
        idxs    = discState.toint(val)
        self.assertItemsEqual(idxs,[0, 1, 3, 5, 9,10,11,11])

    def test1DcValues2Int(self):
        """ Test discretizing 1D continuous values """
        bins        = np.array([11])
        mins        = [0]
        maxs        = [1]
        # bins will be:   [0.0  0.1   0.2   0.3  0.4   0.5   0.6   0.7   0.8   0.9   1.0]
        # bin ids        0    1     2     3    4     5     6     7     8     9     10    11
        discState   = DiscretiseState(bins,mins,maxs)
        val     = np.zeros((8,1))
        val[0]  = -0.01  # 0 (outside min range)
        val[1]  = 0.09   # 1 (first bin)
        val[2]  = 0.2    # 3 (third bin, upper boundary is not part of the bin)
        val[3]  = 0.45   # 5
        val[4]  = 0.89   # 9
        val[5]  = 0.99   # 10 (just before boundary of last bin)
        val[6]  = 1.0    # 11 (boundary max outer bin )
        val[7]  = 1.10   # 11 (outside max range)
        idxs    = discState.toint(val)
        self.assertItemsEqual(idxs,[0, 1, 3, 5, 9,10,11,11])


    def test2DdValues2Int(self):
        """ Tests the conversion of a 2-dimensional continuous variable to a
            bin index.
        """

        bins        = np.array([3,3])
        mins        = [0,0]
        maxs        = [100,100]
        discState   = DiscretiseState(bins,mins,maxs)

        val         = np.zeros((6,2))
        val[0,:]    = np.array([-1,-1]) # outside both dimensions so idx = 0 + 0
        val[1,:]    = np.array([-1,10])  # outside both dimensions so idx = 0 + 1

        val[2,:]    = np.array([0,0])    # 2
        val[3,:]    = np.array([80,0])   # 3
        val[4,:]    = np.array([25,80])  # 4
        val[5,:]    = np.array([80,80])  # 5

        idxs        = discState.toint(val)

        print 'idxs: ', idxs



if __name__ == "__main__":
    unittest.main()
