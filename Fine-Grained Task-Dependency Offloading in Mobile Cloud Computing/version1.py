# coding=utf-8
'''
@Time    : 2019/1/8 14:41
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : version1.py
@Software: PyCharm

state: 一个工作中的任务(6)的cost
reward: -1/0/1
action: 0/1
envirment:

q_table:
-------------
| s | 0 | 1 |
-------------
| s1| 0 | 0 |
-------------
| s2| 0 | 0 |
-------------

'''
import random
import matplotlib.pyplot as plt

class QL:
    def __init__(self):
        self.ALPHA = 0.1
        self.state_num = 6
        self.EPSILON = 0.9  # greedy
        self.GAMMA = 0.9

        self.q_table = [[0.0, 0.0] for _ in range(self.state_num)]
        self.MAX_EPISODES = 10000
        self.action = {0: 'local', 1: 'cloud', 'T': 'done'}
        self.a = []
        self.threshold = 25
        self.eta = 0  #

        self.cost = 0.0

        self.cloudlet = Cloudlet()

    def reset(self):
        self.a = []
        self.cost = 0.0
        #self.q_table = [[0.0, 0.0] for _ in range(self.state_num)]
        self.cloudlet = Cloudlet()

    def get_action(self, old_state):
        r = random.random()
        if r <= self.EPSILON:
            if self.q_table[old_state][0] == self.q_table[old_state][1]:
                action = random.randint(0, 1)
            else:
                action = self.q_table[old_state].index(max(self.q_table[old_state]))  # 0/1最大值的索引
        else:
            action = random.randint(0, 1)
        self.a.append(action)
        return action

    def get_reward(self, old_state, action):
        if self.cost > self.threshold:
            return -1
        # if old_state+2 == self.state_num:
        #     return 1
        else:
            if self.get_cost(old_state + 1, action) > self.threshold:

                return -1
            else:  # < self.threshold
                if old_state + 1 == self.state_num - 1:
                    return 1
                else:
                    return 0

    def get_state(self, old_state, action):
        if self.cost > self.threshold:
            return None, None

        # if old_state == 0:
        #     self.cost += self.get_cost(old_state, action)
        self.cost += self.get_cost(old_state, action)

        if self.cost > self.threshold:
            return None, None
        else:
            return old_state + 1, self.cost

    def get_cost(self, state, action):
        return self.cloudlet.get_cost(state, action)

    def update_q_table(self, old_state, action, new_state):
        q_predict = 0
        q_target = 0

    @staticmethod
    def draw(Y):
        X = [i for i in range(len(Y))]
        plt.plot(Y)
        plt.show()

class Cloudlet(object):

    def __init__(self):
        self.state_num = 6
        self.local_energy = [0] * self.state_num
        self.cloud_energy = [0] * self.state_num
        self.local_time = [0] * self.state_num
        self.cloud_time = [0] * self.state_num
        self.test()
        self.Alpha = 0.3
        self.Beta = 0.7
        self.cost = 0.0
        # self.a = 0

    def test(self):
        self.local_energy[0] = 5
        self.local_energy[1] = 2
        self.local_energy[2] = 3
        self.local_energy[3] = 6
        self.local_energy[4] = 5
        self.local_energy[5] = 2

        self.cloud_energy[0] = 3
        self.cloud_energy[1] = 1
        self.cloud_energy[2] = 2
        self.cloud_energy[3] = 4
        self.cloud_energy[4] = 2
        self.cloud_energy[5] = 1

        self.local_time[0] = 3
        self.local_time[1] = 2
        self.local_time[2] = 1
        self.local_time[3] = 2
        self.local_time[4] = 3
        self.local_time[5] = 2

        self.cloud_time[0] = 6
        self.cloud_time[1] = 5
        self.cloud_time[2] = 2
        self.cloud_time[3] = 2
        self.cloud_time[4] = 5
        self.cloud_time[5] = 4

    def get_cost(self, i, a):  # 一次
        cost = (self.local_energy[i] * self.Alpha + self.local_time[i] * self.Beta) * (1 - a) \
                     + (self.cloud_energy[i] * self.Alpha + self.cloud_time[i] * self.Beta) * a
        return cost

    def vertify(self, a):
        cost = 0
        for i in range(len(a)):
            cost += (self.local_energy[i] * self.Alpha + self.local_time[i] * self.Beta) * (1 - a[i]) \
                         + (self.cloud_energy[i] * self.Alpha + self.cloud_time[i] * self.Beta) * a[i]
        return cost


if __name__ == '__main__':
    ql = QL()
    C = []
    for i in range(ql.MAX_EPISODES):
        s = 0
        r = 0
        c = 0.0
        ql.reset()
        while s != ql.state_num:
            a = ql.get_action(s)
            s_, c = ql.get_state(s, a)
            if s_ is None:
                #print "111111"
                break
            if s == ql.state_num-1:
                break
            r = ql.get_reward(s, a)
            q_predict = ql.q_table[s][a]
            if s_ != ql.state_num - 1 and s_ != ql.state_num:
                bigger = ql.q_table[s_][0] if ql.q_table[s_][0] > ql.q_table[s_][1] else ql.q_table[s_][1]
                q_target = r + ql.GAMMA * bigger
            else:
                q_target = r
            ql.q_table[s][a] += ql.ALPHA * (q_target - q_predict)
            ql.q_table[s][a] = round(ql.q_table[s][a], 2)
            s = s_
            #print ql.q_table
        if c:
            print i
            C.append(c)
            print(c)
            print ql.cloudlet.vertify(ql.a)
            print ql.a
            print ql.q_table
            print ''
    #ql.draw(C)
            #print(ql.q_table)
