import numpy as np

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print 'dir_path: ', dir_path
import sys
sys.path.insert(0, dir_path + "/../..")
print(sys.path)


import rl.utils.plot as pl
import time
import gym
import gym_square



if __name__ == "__main__":


    env = gym.make('square-continuous-state-v0')
    env.reset()

    env.square_world.set_agent_state([0.5,0.5])

    num_episodes    = 5
    bRender         = True

    for eps in range(num_episodes):
        print "episode: ", eps

        state     = env.reset()
        statep    = state
        for _ in range(100):
            if bRender: env.render()

            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if done: break

            state = statep
            time.sleep(0.1)
