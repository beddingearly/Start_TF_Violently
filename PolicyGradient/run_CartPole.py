#coding=utf-8
'''
@Time    : 2018/10/25 08:33
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : run_CartPole.py
@Software: PyCharm
'''
"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import numpy as np
import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
from scipy.interpolate import spline
DISPLAY_REWARD_THRESHOLD = 1500  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print('env.action_space.n', env.action_space.n)
print('env.observation_space.shape[0]', env.observation_space.shape[0])

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=2,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)


X, Y = [], []

for i_episode in range(110):

    X.append(i_episode)

    print('i_episode: ', i_episode)
    observation = env.reset()
    #print('observation', observation)

    while True:
        #if RENDER: env.render()

        action = RL.choose_action(observation)
        #print('action', action)
        observation_, reward, done, info = env.step(action)
        # print('observation_', observation_)
        # print('reward', reward)
        #print('done', done)
        #print('info', info)
        # if done:
        #     print(done)

        RL.store_transition(observation, action, reward)

        if done:
            print('Done', done)
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt, loss = RL.learn()
            Y.append(loss)

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_

fig = plt.figure()
X = np.array(X)
Y = np.array(Y)

plt.plot(X, Y)
plt.show()