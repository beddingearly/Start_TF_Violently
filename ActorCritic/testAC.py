#coding=utf-8
'''
@Time    : 2018/10/27 09:15
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : testAC.py
@Software: PyCharm
'''
import tensorflow as tf
import numpy as np

class Actor(object):
    def __init__(self):
        self.n_features = 4
        self.learning_rate = 0.0001

        self.state = tf.placeholder(tf.float32, [1, self.n_features], name='state')
        self.action = tf.placeholder(tf.float32, [None, ], name='action')
        self.TD_Error = tf.placeholder(tf.float32, [None, ], name='TD_Error')

        self.action_bound = [0, 1]

        self.sess = tf.Session()

        self._build()

        self.sess.run(tf.global_variables_initializer())

    #
    def _build(self):
        # hidden layers 1
        l1 = tf.layers.dense(
            inputs=self.state,
            units=30,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            name='l1'
        )

        # hidden layers  2
        mu = tf.layers.dense(
            inputs=l1,
            units=1,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            name='mu'
        )

        # output
        sigma = tf.layers.dense(
            inputs=mu,
            units=1,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., 0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)

        self.mu = tf.squeeze(mu * 2)
        self.sigma = tf.squeeze(sigma + 0.1)

        # 概率分布
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        # 值域，确定范围
        self.action = tf.clip_by_value(self.normal_dist.sample(1), self.action_bound[0], self.action_bound[1])

        with tf.name_scope('expected_value'):
            log_probably = self.normal_dist.log_prob(self.action)
            self.expected_value = log_probably * self.TD_Error

            self.expected_value += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_operation = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.expected_value, global_step)



    def learn(self, state, action, td_error):
        _, expected_value = self.sess.run([self.train_operation, self.expected_value], feed_dict={
            self.state: state,
            self.action: action,
            self.TD_Error: td_error
        })
        return expected_value

    def action(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.action, feed_dict={
            self.state: state
        })


class Critic(object):
    def __init__(self):
        self.sess = tf.Session()
        self.n_features = 4
        self.learning_rate = 0.01
        self.gamma = 0.9


    def _build(self):
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [1, self.n_features], name='state')
            self.value_next = tf.placeholder(tf.float32, [1, 1], name='value_next')
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.state,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + self.gamma * self.value_next - self.v)
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        value_next = self.sess.run(self.v,
                                   feed_dict={self.state: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.state: s,
                                               self.value_next: value_next,
                                               self.r: r})
        return td_error

if __name__ == '__main__':
    pass
