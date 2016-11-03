""" Test discrete Q-learning behaviour """

import unittest
import numpy as np
from rl.utils.discstate import DiscretiseState
from rl.methods.qlearning import Qlearning
from rl.policies.egreedy import EpsilonGreedy
import gym_leftright
import gym
import time


class TestQlearning(unittest.TestCase):

    def testQ(self):
        """ Tests weather the Q-table is updated correctly """

        qlearning   = Qlearning(10,2)

        state  = 1
        statep = 0
        action = 0
        reward = 50

        qlearning.update(state,action,reward,statep)

        print 'Q: ', qlearning.Q

    def test1DWorld(self):

        print '== Testing Q-learning =='
        return True
        bins        = np.array([11])
        mins        = [0]
        maxs        = [10]
        discState   = DiscretiseState(bins,mins,maxs)
        val         = np.zeros((1,1))

        qlearning   = Qlearning(12,2)
        e_policy    = EpsilonGreedy(qlearning.Q)

        env = gym.make('leftright-v0')

        for i_episode in range(20):

            print 'episode(', i_episode, ')'

            val[0][0] = env.reset()
            state = discState.toint(val)[0]

            for t in range(200):
                env.render()

                action = e_policy.action(state)

                observation, reward, done, info = env.step(action)

                val[0][0] = observation
                statep = discState.toint(val)[0]

                print 'state: ', state, ' action: ', action, ' rewad: ', reward, ' statep: ', statep

                qlearning.update(state,action,reward,statep)

                state = statep
                time.sleep(1.0)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break






        #num_episodes = 1
        #for i in range(0,num_episodes):

if __name__ == '__main__':
    unittest.main()
