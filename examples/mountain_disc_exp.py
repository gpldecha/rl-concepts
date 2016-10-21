"""
    Mountain car example

        method: discrete q-learning

"""


import gym
from gym.spaces import Discrete

import sys
sys.path.append('../rlmethods/')
from qlearning import Qlearning



if __name__ == "__main__":

    qlearninig = Qlearning(num_states=100,num_actions=3)
