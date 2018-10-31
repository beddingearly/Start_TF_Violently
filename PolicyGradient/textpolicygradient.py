#coding=utf-8
'''
@Time    : 2018/10/25 14:50
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : textpolicygradient.py
@Software: PyCharm
'''
import numpy as np
import tensorflow as tf
import gym
from numpy.core.multiarray import ndarray


class PolicyGradient(object):
    def __init__(self):
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.tboard = False
        self.n_actions = 2
        self.n_features = 4

        self.sess = tf.Session()

        self._build()

        self.sess.run(tf.global_variables_initializer())  # finally

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
    @staticmethod
    def _layer(inputs, units, activation, w, b, n):
        return tf.layers.dense(
            inputs=inputs,
            units=units,
            activation=activation,
            kernel_initializer=w,
            bias_initializer=b,
            name=n
        )


    # 建立神经网络
    def _build(self):
        # 环境信息
        self.tf_observations = tf.placeholder(tf.float32, [None, self.n_features], name='observations')

        self.tf_actions_num = tf.placeholder(tf.int32, [None, ], name='actions_num')

        self.tf_actions_value = tf.placeholder(tf.float32, [None, ], name='actions_value')

        # 建立隐藏层
        # 4-10-2结构（4代表输入，16代表隐层神经元个数，2代表两个动作对应的Q值）

        # 输入observation环境信息
        hidden_layer = tf.layers.dense(inputs=self.tf_observations,
                                       units=10,
                                       activation=tf.nn.tanh,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name='fc1')

        # 建立输出层
        # tf.random_normal_initializer的返回结果就是产生平均值为0、标准差为0.3的一组随机数
        all_actions = tf.layers.dense(inputs=hidden_layer,
                                      units=self.n_actions,
                                      activation=None,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      name='fc2')

        # 归一化处理
        # 如果某一个动作得到reward多，那么我们就使其出现的概率增大，
        # 如果某一个动作得到的reward少，那么我们就使其出现的概率减小。
        self.all_acitons_probably = tf.nn.softmax(all_actions, name='all_acitons_probably')
        #print(self.sess.run(self.all_acitons_probably))
        # due to the tf only to minimize
        # 如何得到loss来训练优化？

        with tf.name_scope('loss'):
            # -(log_p * R)
            log_probably = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_actions, labels=self.tf_actions_num)

            # tensorflow中tf.one_hot()函数的作用是将一个值化为一个概率分布的向量，一般用于分类问题。
            log_probably = tf.reduce_sum(-tf.log(all_actions)*tf.one_hot(self.tf_actions_num, self.n_actions), axis=1)
            a = np.array([0.03073904, 0.0014500, -0.03088818, -0.03131252])  # type: ndarray
            #print(self.sess.run(log_probably,
            # feed_dict={self.tf_observations: a[np.newaxis, :],self.tf_actions_num: [2, ]}))

            loss = tf.reduce_mean(log_probably * self.tf_actions_value)

            #print tf.Session().run(loss)

        # 训练
        with tf.name_scope('train'):
            self.train_operation = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    # 选择行为
    def action(self, observation):
        probably_weights = self.sess.run(self.all_acitons_probably,
                                         feed_dict={self.tf_observations: observation[np.newaxis, :]})
        action = np.random.choice(range(probably_weights.shape[1]), p=probably_weights.ravel())
        return action

    # 学习
    def learn(self):
        discounted_and_normalized_episode_reward = self._discounted_and_normalizad_reward()
        self.sess.run(self.train_operation, feed_dict={
            self.tf_observations: np.vstack(self.episode_observations), #
            self.tf_actions_num: np.array(self.episode_actions),
            self.tf_actions_value: discounted_and_normalized_episode_reward
        })

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        return discounted_and_normalized_episode_reward

    # 存储回合
    def store_transition(self, s, a, r):
        self.episode_observations.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)

    # get action_value
    def _discounted_and_normalizad_reward(self):
        print self.episode_rewards
        discounted_episode_reward = np.zeros_like(self.episode_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.episode_rewards))):
            running_add = running_add * self.gamma + self.episode_rewards[t]
            discounted_episode_reward[t] = running_add

        # normalize episode rewards
        discounted_episode_reward = self.normalize(discounted_episode_reward)

        return discounted_episode_reward
    @staticmethod
    def normalize(x):
        x -= x
        x /= x
        return x


if __name__ == '__main__':
    p = PolicyGradient()
    # 设置observations
    ob = np.array([0.293049,  0.9, -0.03088818, -0.03131252])
    action = p.action(ob)
    ob, re, done, info =[0.2, 0.2, 0.2, 0.1], -1, False, {}
    #print action
    #print re
    p.store_transition(ob, action, re)
    discount_episode_reward_normalize = p.learn()

