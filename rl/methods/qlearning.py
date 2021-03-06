"""
Q-learning: implementation of Q-learning for discrete state and actions

Q(s,a)_new <- Q(s,a)_old + alpha * ( r + gamma * max_a Q(s',a) - Q(s,a)_old  )

s     : state   (discrete)
a     : action  (discrete)
gamma : discount factor
alpha : learning rate
r     : reward

"""

import numpy as np


class Qlearning:

    def __init__(self,num_states,num_actions):
        """
            Args:
                num_states  (int): Number of actions
                num_actions (int): Number of states

        """
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self._num_states = num_states
        self._num_actions = num_actions
        self.Q = np.zeros((num_states,num_actions))

    def update(self, state, action, reward, statep, bterminal):
        """Q-learning update rule: Q(s,a)_new <- Q(s,a)_old + alpha * ( r + gamma * max_a Q(s',a) - Q(s,a)_old  )
            Args:
                state (int)     : previous state   s
                action (int)    : previous action  a
                reward (double) : current reward   r
                statep (int)    : current state    s'
                bterminal (bool): True if s is terminal state
        """
        if bterminal:
            self.Q[state, action] = reward
        else:
            self.Q[state, action] += self.alpha*(reward + self.gamma*self.Q[statep,np.argmax(self.Q[statep, :])] - self.Q[state, action])
