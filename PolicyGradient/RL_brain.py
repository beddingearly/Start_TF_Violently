#coding=utf-8
'''
@Time    : 2018/10/25 08:32
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : RL_brain.py
@Software: PyCharm
'''
"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import gym
import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability


        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # tensorflow中tf.one_hot()函数的作用是将一个值化为一个概率分布的向量，一般用于分类问题。
            # 所有的action求和
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)

            #print('neg_log_prob', self.sess.run(neg_log_prob))

            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

            # with tf.Session() as sess:
            #     print('loss', sess.run(loss,
            #                            feed_dict={self.tf_obs: self.observation}))
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # ('range(prob_weights.shape[1])', [0, 1])
        # ('prob_weights.ravel()', array([0.4713806, 0.5286194], dtype=float32))
        print('range(prob_weights.shape[1])', range(prob_weights.shape[1]))
        print('prob_weights.ravel()',prob_weights.ravel())
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob

        # with tf.Session() as sess:
        #     print('loss', sess.run(self.loss,
        #                            feed_dict={self.tf_obs: observation[np.newaxis, :],
        #                                       self.tf_acts: np.array(self.ep_as)}))

        return action

    def store_transition(self, s, a, r):

        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        print('reward', self.ep_rs)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        '''
        ('discounted_ep_rs_norm', 883, array([ 5.38614011e-01,  5.38607212e-01,  5.38600345e-01,  5.38593407e-01,
        5.38586400e-01,  5.38579322e-01,  5.38572173e-01,  5.38564951e-01,
        5.38557657e-01,  5.38550288e-01,  5.38542846e-01,  5.38535328e-01,
        5.38527734e-01,  5.38520063e-01,  5.38512315e-01,  5.38504489e-01,
        5.38496584e-01,  5.38488599e-01,  5.38480533e-01,  5.38472385e-01,
        5.38464156e-01,  5.38455843e-01,  5.38447446e-01,  5.38438965e-01,
        '''
        print('vt', discounted_ep_rs_norm)

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        # LOSS: -log(prob) * vt
        loss = ('loss', self.sess.run(self.loss, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        # ('discounted_ep_rs_norm', array([ 0.36681077,  0.36681077,  0.36681076, ..., -5.66579261,
        #        -5.726728  , -5.7882789 ]))
        print('discounted_ep_rs_norm', len(discounted_ep_rs_norm), discounted_ep_rs_norm)
        return discounted_ep_rs_norm, loss

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # R = r1 + r2 + r3 + ...
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    env = env.reset()
    a = np.array([0.03073904, 0.0014500, -0.03088818, -0.03131252])
    p = PolicyGradient(n_actions=2,
                       n_features=4,
                       learning_rate=0.02,
                       reward_decay=0.99)
    print(p.choose_action(a))