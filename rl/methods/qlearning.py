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

    alpha = 0.1 # learning rate
    gamma = 0.9 # discount factor

    def __init__(self,num_states,num_actions):
        """
            Args:
                num_states  (int): Number of actions
                num_actions (int): Number of states

        """

        self._num_states  = num_states
        self._num_actions = num_actions
        self.Q            = np.zeros((num_states,num_actions))

    def action(self,state):
        """ orresponding to state with maximum value
            Args:
                state (int) : State index (discrete states)
            Returns:
                (int)       : action to be applied
        """
        return self.actions[np.argmax(self.Q[state,:])]

    def update(self,state,action,reward,statep):
        """Q-learning update rule: Q(s,a)_new <- Q(s,a)_old + alpha * ( r + gamma * max_a Q(s',a) - Q(s,a)_old  )
            Args:
                state (int)     : previous state   s
                action (int)    : previous action  a
                reward (double) : current reward   r
                statep (int)    : current state    s'
        """
        self.Q[state,action] = self.Q[state,action] + self.alpha * ( self.gamma  * self.Q[statep,np.argmax(self.Q[statep,:])] - self.Q[state,action]  )
