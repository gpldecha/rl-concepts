import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym


class Agent:

    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

        with tf.variable_scope("policy"):
            hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
            self.probability = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.probability, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.probability)[0]) * tf.shape(self.probability)[1] + self.action_holder
        # gets the probability which were associated with each action
        self.responsible_outputs = tf.gather(tf.reshape(self.probability, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables("policy")
        self.gradient_holders = []
        with tf.variable_scope("policy"):
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


class BaseLine:

    def __init__(self, lr, s_size, h_size):
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

        with tf.variable_scope("value"):
            hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
            self.baseline = slim.fully_connected(hidden, 1, activation_fn=tf.nn.relu, biases_initializer=None)

        # The next lines establish the training procedure
        self.delta = tf.placeholder(shape=[None], dtype=tf.float32)

        self.loss = -tf.reduce_mean(self.baseline*self.delta)

        tvars = tf.trainable_variables("value")
        self.gradient_holders = []
        with tf.variable_scope("value"):
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_baseline_holder')
                self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def delta_value(G, states, value):
    b = sess.run(value.baseline, feed_dict={value.state_in: states})
    return G - b.flatten()


def set_gradient_zero(gradients):
    for ix, grad in enumerate(gradients):
        gradients[ix] = grad*0


if __name__ == "__main__":
    tf.reset_default_graph()  # Clear the Tensorflow graph.

    # setup the environment
    env = gym.make('CartPole-v0')

    agent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)
    value = BaseLine(lr=1e-2, s_size=4, h_size=8)

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
        grad_policy_buffer = sess.run(tf.trainable_variables("policy"))
        grad_value_buffer = sess.run(tf.trainable_variables("value"))

        set_gradient_zero(grad_policy_buffer)
        set_gradient_zero(grad_value_buffer)

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

                    # compute delta
                    delta = delta_value(G, S, value)

                    # derivative of the value/baseline function with respect to its parameters
                    value_grads = sess.run(value.gradients, feed_dict={value.delta: delta, value.state_in: S})
                    # add gradient to the statistics
                    for idx, v_grad in enumerate(value_grads):
                        grad_value_buffer[idx] += v_grad

                    # derivative of the policy function with respect to its parameters
                    policy_grads = sess.run(agent.gradients, feed_dict={agent.reward_holder: delta, agent.action_holder: A, agent.state_in: S})
                    # add gradient to the statistics
                    for idx, p_grad in enumerate(policy_grads):
                        grad_policy_buffer[idx] += p_grad

                    # actually update the policy and value at specific intervals
                    # after sufficient gradient statistics have been computed.
                    if i % update_frequency == 0 and i != 0:
                        # update value
                        _ = sess.run(value.update_batch, feed_dict=dict(zip(value.gradient_holders, grad_value_buffer)))
                        #  update policy
                        _ = sess.run(agent.update_batch, feed_dict=dict(zip(agent.gradient_holders, grad_policy_buffer)))

                        set_gradient_zero(grad_policy_buffer)
                        set_gradient_zero(grad_value_buffer)

                    total_reward.append(running_reward)
                    total_lenght.append(j)
                    break

                    # Update our running tally of scores.
            if i % 100 == 0:
                print(np.mean(total_reward[-100:]))
            i += 1