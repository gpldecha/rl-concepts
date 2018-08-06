from __future__ import print_function

import gym
import tensorflow as tf
import numpy as np

from rl.policies.reinforce import REINFORCE
from collections import deque

env = gym.make('CartPole-v0')  # see https://github.com/openai/gym/wiki/CartPole-v0
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

sess = tf.Session()

num_actions = 2
num_observations = 4

policy = REINFORCE(sess)


def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):

    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes + 1): # generate episodes

        states = []
        actions = []
        rewards = []

        total_rewards = []
        episode_reward = 0

        observation = env.reset()
        for t in range(max_t):
            # calculate policy
            action, action_probability = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            states.append(observation)
            actionblank = np.zeros(2)
            actionblank[action] = 1
            actions.append(actionblank)
            if done:
                break
            # take the action in the environment
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards) + 1)]
        # R = sum([a * b for a, b in zip(discounts, rewards)])

        print(rewards)

        rewards = np.reshape(rewards, (-1, 1))
        policy.train(states, rewards, actions)

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores


scores = reinforce()

#!/usr/bin/env python