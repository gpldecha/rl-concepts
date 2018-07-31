import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')


tf.reset_default_graph()

inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W       = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout    = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

nextQ       = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss        = tf.reduce_sum(tf.square(nextQ - Qout))
trainer     = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
max_episode_length = 100

env.reset()
#create lists to contain total rewards and steps per episode
jList = []
rList = []

for i in range(num_episodes):
    if i % 100 == 0:
        print 'episode: ', i

    s = env.reset()

    rAll = 0
    d = False
    j = 0

    for _ in range(max_episode_length):
        j = j + 1
        # generate an action
        a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]}) # (1 x 16) hot encoding

        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a[0])

        # Obtain the Q' values by feeding the new state through our network
        Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})

        # Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)

        targetQ = allQ
        targetQ[0, a[0]] = r + y * maxQ1

        _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
        rAll += r
        s = s1
        if d == True:
            # Reduce chance of random action as we train the model.
            e = 1. / ((i / 50) + 10)
            break

    jList.append(j)
    rList.append(rAll)

plt.figure()
plt.plot(jList)
plt.show()




