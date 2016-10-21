import gym
import numpy



if __name__ == "__main__":


    env = gym.make('MountainCar-v0')
    observation = env.reset()
    print(observation)
    action = env.action_space.sample()
    print 'action: ', type(action), ' state: ', type(observation)

    observation, reward, done, info = env.step(action)


    data = numpy.random.random(100)
    print data
    bins = numpy.linspace(0, 1, 10)
    digitized = numpy.digitize(data, bins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
