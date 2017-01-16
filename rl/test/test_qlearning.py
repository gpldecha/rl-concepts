""" Test discrete Q-learning behaviour """

import unittest
import numpy as np
from rl.utils.discstate import DiscretiseState
from rl.methods.qlearning import Qlearning
<<<<<<< HEAD
from rl.policies.egreedy import Egreedy
import rl.utils.plot as pl
import time
import gym
import gym_leftright
=======
from rl.policies.egreedy import EpsilonGreedy
import gym_leftright
import gym
import time
>>>>>>> 30ef5d6fa4f56284eb86f7f3bf3764939cc9bcbf


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

<<<<<<< HEAD
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
=======
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
>>>>>>> 30ef5d6fa4f56284eb86f7f3bf3764939cc9bcbf

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
