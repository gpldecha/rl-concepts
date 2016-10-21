import unittest

import numpy as np
from rl.utils.discstate import DiscretiseState


def testmethod():
    pass


class TestDiscretiseStateMethods(unittest.TestCase):

    def test1DcValues2Int(self):

        # create a discretisation object
        bins        = np.array([10])
        mins        = [0]
        maxs        = [100]
        discState   = DiscretiseState(bins,mins,maxs)

        val     = np.zeros((5,1))
        val[0]  = 0
        val[1]  = 9
        val[2]  = 20
        val[3]  = 45
        val[4]  = 89
        idxs    = discState.toint(val)
        self.assertItemsEqual(idxs,[1, 1, 2, 5, 9])



if __name__ == "__main__":
    unittest.main()
