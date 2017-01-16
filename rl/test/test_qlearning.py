""" Test discrete Q-learning behaviour """

import unittest
import numpy as np
from rl.utils.discstate import DiscretiseState
from rl.methods.qlearning import Qlearning
from rl.policies.egreedy import Egreedy
import rl.utils.plot as pl
import time
import gym
import gym_leftright


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
        print "== Test Q-learning =="

        # Setup of environment and discretisation
        env = gym.make('leftright-v0')
        tmp                     = np.zeros((1,1))
        bins                    = np.array([9])
        mins                    = [env.left_boundary]
        maxs                    = [env.right_boundary]
        discState               = DiscretiseState(bins,mins,maxs)


        qlearning = Qlearning(num_states=10,num_actions=2)
        egreedy   = Egreedy(Q=qlearning.Q,epsilon=0.5)

        num_episodes = 100

        plot_value_func = pl.PlotValueFunction1D(np.arange(0,10),np.max(qlearning.Q,1))

        #raw_input("Press Enter to continue...")

        for eps in range(num_episodes):

            print "episode: ", eps

            tmp[0]                  = env.reset()
            previous_state_index    = discState.toint(tmp)
            env.render()


            for _ in range(50):
                env.render()

                action = egreedy.action(previous_state_index)

                state_value, reward, done, info = env.step(action)
                tmp[0] = state_value
                current_state_index = discState.toint(tmp)

                qlearning.update(state=previous_state_index,action=action,reward=reward,statep=current_state_index)

                previous_state_index = current_state_index

                plot_value_func.update(np.max(qlearning.Q,1))

                if done:
                    print 'goal reached!'
                    print '  '
                    print 'state:                   ', state_value
                    print 'previous_state_index:    ', previous_state_index
                    print 'action:                  ', action
                    print 'reward:                  ', reward
                    print 'current_state_index:     ', current_state_index
                    print '  '
                    #raw_input("Finished Press Enter to continue...")
                    break


        #raw_input("Finished Press Enter to continue...")
        print 'Q:       ', qlearning.Q

if __name__ == '__main__':
    unittest.main()
