import gym
import tensorflow as tf
import numpy as np
from rl.utils.experience_buffer import ExperienceBuffer
from rl.utils.math import uniform_sample
from rl.methods.qmlp import QMLP
import matplotlib.pyplot as plt
import time

env = gym.make('MountainCar-v0')

# check https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/agent.py
# check https://github.com/vikasjiitk/Deep-RL-Mountain-Car/blob/master/MCqlearn.py
# https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/
max_episode_length = 200
visualise = True

experience_buffer = ExperienceBuffer(state_size=2, action_size=1, num_samples=4*max_episode_length)

s_ = experience_buffer.EntryType.s
r_ = experience_buffer.EntryType.r
sp_ = experience_buffer.EntryType.sp
done_ = experience_buffer.EntryType.done

num_episodes = 100
discount = 0.99
epsilon = 0.9
epsilon_start = 1.0
epsilon_end = 0.01
annealing_steps = 100*max_episode_length
epsilon_delta = (epsilon_end - epsilon_start)/annealing_steps
C = 4*max_episode_length
minibatch_size = max_episode_length*4

qmlp = QMLP(n_input=2, hidden_layers=[10, 10], n_output=3)
qmlp.is_saving = False
qmlp.display_step = 1000
target_qmlp = QMLP(n_input=2, hidden_layers=[10, 10], n_output=3)

sess = tf.Session()

reward = -1
logs_path = qmlp.logs_path
myobj = None

summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

total_steps = 0
steps = 0
for episode in range(num_episodes):
    print 'episode: ', episode
    state = env.reset()
    sum_reward = 0.0
    steps = 0
    for _ in range(max_episode_length):
        if visualise: env.render()

        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            action, q_values = qmlp.max_a(sess, np.reshape(state, (1, 2)))

        next_state, reward, done, _ = env.step(action)
        if reward != -1:
            print 'GOAL REACHED'
            done = True
        else:
            done = False
       # print 'next_state: ', next_state, ' reward: ', reward, ' done : ', done, ' step: ', steps

        experience_buffer.append(s=state, a=action, r=reward, sp=next_state, done=int(done))

        mini_batch = uniform_sample(experience_buffer, minibatch_size)

        y, s = target_qmlp.get_targets(sess=sess, discount=discount, s=experience_buffer.get_item(mini_batch, s_), r=experience_buffer.get_item(mini_batch, r_), sp=experience_buffer.get_item(mini_batch, sp_), done=experience_buffer.get_item(mini_batch, done_))

        # train
        qmlp.train(sess=sess, s=s, y=y, max_steps=10)

        if epsilon > epsilon_end: epsilon += epsilon_delta
        state = next_state

        if total_steps % C == 0:
            print 'update QMLP'
            target_qmlp.update_parameters(sess, qmlp, 1.0)

        sum_reward += reward
        total_steps += 1
        steps +=1
        if done or steps >= max_episode_length:
            break

    h1 = target_qmlp.get_parameter(sess, 'h1')

    plt.close("all")
    plt.figure()
    myobj = plt.imshow(h1, interpolation='None', cmap='seismic')
    plt.axis('off')
    plt.colorbar(myobj)
    plt.show(False)

    average_reward = float(sum_reward)/float(steps)
    print 'average reward: ', average_reward, ' steps: ', steps, '  epsilon: ', epsilon
    summary = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=average_reward)])
    summary_writer.add_summary(summary, episode)