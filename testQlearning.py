#coding=utf-8
import random

'''
Q-Learning:
q_target = r + self.gamma * self.q_table.ix[s_, :].max()
self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

ACTION
STATE
REWARD
ENVIRONMENT

action-state-action-state
'''

class QL(object):
    def __init__(self):
        self.ALPHA = 0.1
        self.N_STATE = 6
        self.EPSILON = 0.9 # greedy
        self.GAMMA = 0.9

        # Q表
        self.q_table = [[0, 0] for i in range(self.N_STATE)]# 0 left / 1 right
        self.MAX_EPISODES = 50# 学习次数
        self.reward = 0
        self.action = {0: 'left', 1: 'right', 'T': 'done'}
        self.state = 0 # len(self.q_table)


    def get_Action(self, state):# 上一个state--下一个action
        r = random.random()
        if r <= self.EPSILON:
            if self.q_table[state][0] == self.q_table[state][1]:
                action = random.randint(0, 1)
            # 找最大值
            else:
                action = self.q_table[state].index(max(self.q_table[state]))
        else:
            action = random.randint(0, 1)
        return action

    # choose action based on state
    def get_Reward(self, state, action): # 旧的state 返回新的state
        #action = self.action(state) # 0-1
        if action == 1:  # right
            if state == self.N_STATE - 2:
                reward = 1
            else:
                reward = 0
        else:  # left
            if state == 0:
                reward = 0
            else:
                reward = 0
        return reward

    def get_State(self, state, action):#  新的action old state
        if action == 1:  # right
            if state == self.N_STATE - 2:
                return 'T'
            else:
                state += 1
        else:  # left
            if state == 0:
                state = 0
            else:
                state -= 1
        return state

    # update Q-table
    def get_Environment(self, state, reward, action, old_state):# 本次state
        # state, reward = self.reward(state)
        predict = self.q_table[old_state][action]
        if state == 'T':
            target = reward
        else:
            target = reward + self.GAMMA * self.q_table[state].index(max(self.q_table[state]))
        # if self.ALPHA * (target - predict) != 0.0:
        #     print self.ALPHA * (target - predict)
        self.q_table[old_state][action] += self.ALPHA * (target - predict)
        #print self.q_table[old_state][action]


    def ql(self):
        for i in range(self.MAX_EPISODES):
            state = 0
            Terminate = True
            while Terminate:
                action = self.get_Action(state)
                reward = self.get_Reward(state, action)
                new_state = self.get_State(state, action)
                if new_state == 'T':
                    Terminate = False
                self.get_Environment(new_state, reward, action, state)
                #print self.q_table

                state = new_state
                print self.action[action]
                print self.q_table


    def move(self):
        pic = '''
        ooooo
        ooooo
        ooooo
        ooooo
        ooooo
        '''
        print pic
if __name__ == '__main__':

    q = QL()
    q.ql()
