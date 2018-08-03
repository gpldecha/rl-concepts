import tensorflow as tf
from rl.predictors.mlp import MLP
import numpy as np

# https://github.com/leimao/OpenAI_Gym_AI/blob/master/LunarLander-v2/REINFORCE/2017-05-24-v1/OpenAI_REINFORCE_FC_TF.py


class ReinforcePG:

    def __init__(self, session, num_actions, num_features):
        self.session = session
        self.num_actions = num_actions
        self.num_features = num_features

        with tf.name_scope('inputs'):

            self.tf_observations = tf.placeholder(tf.float32, [None, self.num_features], name = 'observations')
            self.tf_actions = tf.placeholder(tf.int32, [None,], name = 'num_actions')
            self.tf_values = tf.placeholder(tf.float32, [None,], name = 'state_values')

        self.policy = MLP(n_input=1, hidden_layers=[5, 5, 5], n_output=1)

        with tf.name_scope('loss'):
            # To maximize (log_p * V) is equal to minimize -(log_p * V)
            # Construct loss function mean(-(log_p * V)) to be minimized by tensorflow
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy.action_probs, labels=self.tf_actions)  # this equals to -log_p
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_values)

    def get_action(self, observation):
        action_probs = self.session.run([self.policy.action_probs], feed_dict={self.policy.X: observation})

        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
        return action, action_probs[action]
