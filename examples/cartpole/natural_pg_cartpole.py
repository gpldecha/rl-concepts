import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import tensorflow.contrib.slim as slim
import numpy as np
import random
import gym

# https://studywolf.wordpress.com/2018/06/20/natural-policy-gradient-in-tensorflow/

# https://github.com/studywolf/blog/blob/master/tensorflow_models/npg_cartpole/natural_policy_gradient.py

# check http://rail.eecs.berkeley.edu/deeprlcoursesp17/docs/lec5.pdf

# https://medium.com/autonomous-agents/how-to-tame-the-valley-hessian-free-hacks-for-optimizing-large-neuralnetworks-5044c50f4b55

# computing hessian https://gist.github.com/guillaume-chevalier/6b01c4e43a123abf8db69fa97532993f


def hessian_vector_product(ys, xs, v):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = tf.gradients(ys, xs)

    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v)
        if grad_elem is not None
    ]
    # Second backprop
    return tf.gradients(elemwise_products, xs)


class Agent:

    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        with tf.variable_scope("policy"):
            hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
            self.probability = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.probability, 1)

        # Training procedure setup
        self.value_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.probability)[0]) * tf.shape(self.probability)[1] + self.action_holder
        # gets the probability which were associated with each action
        self.responsible_outputs = tf.gather(tf.reshape(self.probability, [-1]), self.indexes)

        self.action_log_prob = tf.log(self.responsible_outputs)

        # # calculate the gradient of the log probability at each point in time.
        self.t_parameters = tf.trainable_variables("policy")

        self.loss = -tf.reduce_mean(self.action_log_prob*self.value_holder)

        self.gradients = tf.gradients(self.loss, self.t_parameters)

        self.vector_holder = []
        for parameter in self.t_parameters:
            self.vector_holder.append(tf.Variable(tf.random_normal(parameter.get_shape())))


        self.npg_grad = hessian_vector_product(self.loss, self.t_parameters, self.vector_holder)


        # self.hess = tf._hessian_vector_product(self.action_log_prob, self.t_parameters)

        # list ov variables
        # [0] tf.Variable shape=(4, 8)
        # [1] tf.Variable shape=(8, 2)

        # g_log_prob = tf.map_fn(lambda x: tf.gradients(x, self.t_parameters)[0], self.action_log_prob)
        #
        # self.gradient_holders = []
        # with tf.variable_scope("policy"):
        #     for idx, var in enumerate(self.t_parameters):
        #         placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
        #         self.gradient_holders.append(placeholder)
        #
        # # gradient
        # self.g = tf.reduce_mean(g_log_prob*self.value_holder)

        # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.t_parameters))


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def set_gradient_zero(gradients):
    for ix, grad in enumerate(gradients):
        gradients[ix] = grad*0


if __name__ == "__main__":
    tf.reset_default_graph()  # Clear the Tensorflow graph.

    # setup the environment
    env = gym.make('CartPole-v0')

    agent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)

    gamma = 0.99
    total_episodes = 5000  # Set total number of episodes to train agent on.
    max_ep = 999
    update_frequency = 5  # how many MC samples of the gradient we gather before we make an update to the policy.

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_lenght = []

        # Setting gradients of the policy to ZERO dc_dw1, dc_dw2
        #grad_policy_buffer = sess.run(agent.t_parameters)
        #set_gradient_zero(grad_policy_buffer)

        while i < total_episodes:
            s = env.reset()
            running_reward = 0
            ep_history = []
            for j in range(max_ep):
                # Probabilistically pick an action given our network outputs.
                a_dist = sess.run(agent.probability, feed_dict={agent.state_in: [s]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)

                s1, r, done, _ = env.step(a)  # Get our reward for taking an action given a bandit.
                ep_history.append([s, a, r, s1])
                s = s1
                running_reward += r

                if done:
                    # Update the network.
                    ep_history = np.array(ep_history)

                    S = np.vstack(ep_history[:, 0])

                    A = ep_history[:, 1]

                    # compute discounted rewards G
                    G = discount_rewards(ep_history[:, 2], gamma)

                    gradient = sess.run(agent.gradients, feed_dict={agent.value_holder: G, agent.action_holder: A, agent.state_in: S})
                    print('gradient[0] {}'.format(gradient[0].shape))
                    print('gradient[1] {}'.format(gradient[1].shape))


                    npg = sess.run(agent.npg_grad, feed_dict={agent.value_holder: G, agent.action_holder: A, agent.state_in: S})
                    print('npg[0] {}'.format(gradient[0].shape))
                    print('npg[1] {}'.format(gradient[1].shape))

                    # update control policy
                    #_ = sess.run(agent.gra, feed_dict=dict(zip(agent.gradient_holders, agent.gradients)))

                    total_reward.append(running_reward)
                    total_lenght.append(j)
                    break

                    # Update our running tally of scores.
            if i % 100 == 0:
                print(np.mean(total_reward[-100:]))
            i += 1