import unittest
import numpy as np
import rl.policies.softmax as sf
from rl.policies.softmax import SoftMax

class TestPolicies(unittest.TestCase):

    def testSoftMax(self):

        Q = np.zeros((10,2))
        softmax = SoftMax(Q)
        softmax.temp = 100000
        Q[5,:] = np.array([1,10])

        print '== Testing softmax action selection =='
        print ' temperature: ', softmax.temp

        counts = np.array([0,0])
        num_a_selec = 1000
        for i in range(0,num_a_selec):
            a = softmax.action(5)
            counts[a] = counts[a] + 1

        #counts = counts / num_a_selec
        print 'p(actions): ', (counts + 0.0001) / num_a_selec
