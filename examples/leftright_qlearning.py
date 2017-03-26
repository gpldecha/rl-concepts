""" Test discrete Q-learning behaviour """

import numpy as np

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print 'dir_path: ', dir_path
import sys
sys.path.insert(0, dir_path + "/..")
print(sys.path)


from rl.utils.discstate import Discretise
from rl.methods.qlearning import Qlearning
from rl.policies.egreedy import Egreedy
import rl.utils.plot as pl
import time
import gym
import gym_leftright



if __name__ == "__main__":

        env = gym.make('leftright-v0')
        env.max_speed  = 1
        discState      = Discretise(bin_sizes=10,state_mins=0,state_maxs=10)

        qlearning = Qlearning(num_states=11,num_actions=2)
        qlearning.alpha = 0.05
        qlearning.gamma = 0.5

        egreedy   = Egreedy(Q=qlearning.Q,epsilon=0.8)

        num_episodes = 300
        bPlot        = True
        bRender      = False

        if bPlot:
            plot_value_func = pl.PlotQFunction1D(np.arange(0,11),qlearning.Q)

        for eps in range(num_episodes):

            print "episode: ", eps

            state_value     = env.reset()
            state           = discState.numtoint(float(state_value))

            for _ in range(100):
                if bRender:
                    env.render()

                action = egreedy.action(state)

                state_value, reward, done, info = env.step(action)
                statep = discState.numtoint(float(state_value))

                qlearning.update(state=state,action=action,reward=reward,statep=statep,bterminal=done)

                if done:
                    break

                state = statep

                if bPlot:
                    plot_value_func.update(qlearning.Q)
                    time.sleep(0.01)


        raw_input("Finished Press Enter to continue...")
