"""
    Base class : RL methods
"""


class base_rl:

    """ Base class for RL methods """

    gamma = 0.9 % discount factor
    alpha = 0.1 % learning rate

    def __init__(self,get_action,update_parameters):
        """
        Args:
                get_action : A function which retuns an action given a state
                             a = get_action(state)

                update_parameters : A function which updates the parameters of
                                    either the policy, value or q-value functions
        """

        self.get_action = get_action
        self.update_parameters = update_parameters


    def action(self,state):
        """ policy: returns action  """
        return self.update_parameters(state)

    def update(self,state):
        """
            updates the parameters of the value function or policy
        """
        self.update_parameters(state)
