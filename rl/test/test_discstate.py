import unittest
import numpy as np
from rl.utils.discstate import DiscretiseState

class TestDiscretiseStateMethods(unittest.TestCase):

    def setUp(self):
            #  Test discrestisation of 2D continuous data
            #             ____________________
            #          3 | 12 | 13 | 14 | 15 |
            #    y       |____|____|____|____|
            #    |     2 |  8 |  9 | 10 | 11 |
            #    a       |____|____|____|____|
            #    x     1 |  4 |  5 |  6 |  7 |
            #    i       |____|____|____|____|
            #    s     0 |  0 |  1 |  2 |  3 |
            #            |____|____|____|____|
            #               0    1    2    3
            #                   x-axis
            #
            #   x-axis : 1st dimension of val
            #   y-axis : 2nd dimension of val

            val         = np.zeros((16,2))
            val[0,:]    = np.array([-1,-1])     # (0,0) =  0
            val[1,:]    = np.array([-1,25])     # (0,1) =  1
            val[2,:]    = np.array([-1,75])     # (0,2) =  2
            val[3,:]    = np.array([-1,101])    # (0,3) =  3

            val[4,:]    = np.array([10,-1])     # (1,0) =  5
            val[5,:]    = np.array([10,25])     # (1,1) =  5
            val[6,:]    = np.array([10,75])     # (1,2) =  6
            val[7,:]    = np.array([10,101])    # (1,3) =  7

            val[8,:]    = np.array([75,-1])    # (2,0) =  8
            val[9,:]    = np.array([75,25])    # (2,1) =  9
            val[10,:]   = np.array([75,75])    # (2,2) = 10
            val[11,:]   = np.array([75,101])   # (2,3) = 11

            val[12,:]    = np.array([101,-1])    # (2,0) =  12
            val[13,:]    = np.array([101,25])    # (2,1) =  13
            val[14,:]    = np.array([101,75])    # (2,2) =  14
            val[15,:]    = np.array([101,101])   # (2,3) =  15

            self.twoDvals = val
            self.twoDidx  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

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

        bins        = np.array([3,3]) # 1st dimension has four bins [0,1,2,3,4].
        mins        = [0,0]           # 2nd dimension same as 1st.
        maxs        = [100,100]
        discState   = DiscretiseState(bins,mins,maxs)
        idxs        = discState.twoDtoint(self.twoDvals)
        self.assertItemsEqual(idxs,self.twoDidx)

    def testNDdValues2Int(self):
        """ Tests the conversion of a 2-dimensional continuous variable to a
            bin index whilst using the N-dimensional conversion function.
        """
        bins        = np.array([3,3]) # 1st dimension has four bins [0,1,2,3,4].
        mins        = [0,0]           # 2nd dimension same as 1st.
        maxs        = [100,100]
        discState   = DiscretiseState(bins,mins,maxs)
        idxs        = discState.toint(self.twoDvals)
        self.assertItemsEqual(idxs,self.twoDidx)

if __name__ == '__main__':
    unittest.main()
