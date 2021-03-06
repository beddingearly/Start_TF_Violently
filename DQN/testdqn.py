#coding=utf-8
'''
@Time    : 2018/10/30 10:24
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : testdqn.py
@Software: PyCharm
'''
import numpy as np
import tensorflow as tf
import random
import numpy as np
import time
import sys
import collections

class DeepQNetwork():
    def __init__(self,
                 model,
                 env,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.001,
                 gamma=0.9,
                 replay_memeory_size=10000,
                 batch_size=32,
                 initial_epsilon=0.5,
                 final_epsilon=0.01,
                 decay_factor=1,
                 logdir=None,
                 save_per_step=1000,
                 test_per_epoch=100):
        self.model = model
        self.env = env
        self.num_actions = model.get_num_outputs()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_factor = decay_factor
        self.logdir = logdir
        self.test_per_epoch = test_per_epoch

        self.replay_memeory = collections.deque()
        self.replay_memeory_size = replay_memeory_size
        self.batch_size = batch_size
        self.define_q_network()
        # session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        if self.logdir is not None:
            if not self.logdir.endswith('/'): self.logdir += '/'
            self.save_per_step = save_per_step
            self.saver = tf.train.Saver()
            checkpoint_state = tf.train.get_checkpoint_state(self.logdir)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                path = checkpoint_state.model_checkpoint_path
                self.saver.restore(self.sess, path)
                print('Restore from {} successfully.'.format(path))
            else:
                print('No checkpoint.')
            self.summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                self.logdir, self.sess.graph)
            sys.stdout.flush()

    def define_q_network(self):
        self.input_states, self.q_values = self.model.definition()
        self.input_actions = tf.placeholder(tf.float32,
                                            [None, self.num_actions])
        # placeholder of target q values
        self.input_q_values = tf.placeholder(tf.float32, [None])
        # only use selected q values
        action_q_values = tf.reduce_sum(
            tf.multiply(self.q_values, self.input_actions),
            reduction_indices=1)

        self.global_step = tf.Variable(0, trainable=False)
        # define cost
        self.cost = tf.reduce_mean(
            tf.square(self.input_q_values - action_q_values))
        self.optimizer = self.optimizer(self.learning_rate).minimize(
            self.cost, global_step=self.global_step)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('reward', tf.reduce_mean(action_q_values))

    def egreedy_action(self, state):
        if random.random() <= self.epsilon:
            action_index = random.randint(0, self.num_actions - 1)
        else:
            action_index = self.action(state)
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.decay_factor
        return action_index

    def action(self, state):
        q_values = self.q_values.eval(feed_dict={self.input_states:
                                                 [state]})[0]
        return np.argmax(q_values)

    def do_train(self, epoch):
        # randomly select a batch
        mini_batches = random.sample(self.replay_memeory, self.batch_size)
        state_batch = [batch[0] for batch in mini_batches]
        action_batch = [batch[1] for batch in mini_batches]
        reward_batch = [batch[2] for batch in mini_batches]
        next_state_batch = [batch[3] for batch in mini_batches]

        # target q values
        target_q_values = self.q_values.eval(
            feed_dict={self.input_states: next_state_batch})
        input_q_values = []
        for i in range(len(mini_batches)):
            terminal = mini_batches[i][4]
            if terminal:
                input_q_values.append(reward_batch[i])
            else:
                # Discounted Future Reward
                input_q_values.append(reward_batch[i] +
                                      self.gamma * np.max(target_q_values[i]))
        feed_dict = {
            self.input_actions: action_batch,
            self.input_states: state_batch,
            self.input_q_values: input_q_values
        }
        self.optimizer.run(feed_dict=feed_dict)
        step = self.global_step.eval()
        if self.saver is not None and epoch > 0 and step % self.save_per_step == 0:
            summary = self.sess.run(self.summaries, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()
            self.saver.save(self.sess, self.logdir + 'dqn', self.global_step)

    # num_epoches: train epoches
    def train(self, num_epoches):
        for epoch in range(num_epoches):
            epoch_rewards = 0
            state = self.env.reset()
            # 9999999999: max step per epoch
            for step in range(9999999999):
                # ε-greedy exploration
                action_index = self.egreedy_action(state)
                next_state, reward, terminal, info = self.env.step(
                    action_index)
                # one-hot action
                one_hot_action = np.zeros([self.num_actions])
                one_hot_action[action_index] = 1
                # store trans in replay_memeory
                self.replay_memeory.append((state, one_hot_action, reward,
                                            next_state, terminal))
                # remove element if exceeds max size
                if len(self.replay_memeory) > self.replay_memeory_size:
                    self.replay_memeory.popleft()

                # now train the model
                if len(self.replay_memeory) > self.batch_size:
                    self.do_train(epoch)

                # state change to next state
                state = next_state
                epoch_rewards += reward
                if terminal:
                    # Game over. One epoch ended.
                    break
            # print("Epoch {} reward: {}, epsilon: {}".format(
            #     epoch, epoch_rewards, self.epsilon))
            # sys.stdout.flush()

            #evaluate model
            if epoch > 0 and epoch % self.test_per_epoch == 0:
                self.test(epoch, max_step_per_test=99999999)

    def test(self, epoch, num_testes=10, max_step_per_test=300):
        total_rewards = 0
        print('Testing...')
        sys.stdout.flush()
        for _ in range(num_testes):
            state = self.env.reset()
            for step in range(max_step_per_test):
                # self.env.render()
                action = self.action(state)
                state, reward, terminal, info = self.env.step(action)
                total_rewards += reward
                if terminal:
                    break
        average_reward = total_rewards / num_testes
        print("epoch {:5} average_reward: {}".format(epoch, average_reward))
        sys.stdout.flush()
class CartPoleEnv(Env):
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def step(self, action_index):
        s, r, t, i = self.env.step(action_index)
        return s, r, t, i

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

model = SimpleNeuralNetwork([4, 16, 2])
env = CartPoleEnv()
qnetwork = DeepQNetwork(
model=model, env=env, learning_rate=0.0001, logdir='./tmp/CartPole/')
qnetwork.train(4000)