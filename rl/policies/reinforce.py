import tensorflow as tf
from rl.predictors.mlp import MLP
import numpy as np
import time
# https://github.com/leimao/OpenAI_Gym_AI/blob/master/LunarLander-v2/REINFORCE/2017-05-24-v1/OpenAI_REINFORCE_FC_TF.py


class REINFORCE:

    def __init__(self, session,  s_size=4, h_size=16, a_size=2):

        self.session = session
        self.outout_size = a_size

        with tf.name_scope('policy'):

            self.tf_observations = tf.placeholder(tf.float32, [None, 4], name='observations')
            self.tf_actions = tf.placeholder(tf.int32, [None, ], name='num_actions')
            self.tf_values = tf.placeholder(tf.float32, [None, ], name='state_values')

            # FC1
            fc1 = tf.layers.dense(
                inputs=self.tf_observations,
                units=16,
                activation=tf.nn.relu,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='FC1'
            )

            # FC2
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=32,
                activation=tf.nn.tanh,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='FC2'
            )

            # FC3
            self.logits = tf.layers.dense(
                inputs=fc2,
                units=2,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='FC3'
            )

            self.probabilities = tf.nn.softmax(self.logits, name='action_probs')


        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_actions)  # this equals to -log_p
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_values)

        with tf.name_scope('train'):

            self.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

    def train(self, state, values, actions):
        _, train_loss = self.session.run([self.optimizer, self.loss], feed_dict = {
        self.tf_observations: state,
        self.tf_actions: actions,
        self.tf_values: values})

    def get_action(self, observation):
        observation = np.reshape(observation, (1, 4))
        action_prob = self.session.run([self.probabilities], feed_dict={self.tf_observations: observation})[0]
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, action_prob[0][action]
