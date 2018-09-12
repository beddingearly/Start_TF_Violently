#coding=utf-8

import numpy as np
import pandas as pd
import time

N_STATES = 6   # 1维世界的宽度

ACTIONS = ['left', 'right']     # 探索者的可用动作

# 90%的情况会按照Q表的最优值来选择行为，10%随机选择行为
EPSILON = 0.9   # 贪婪度 greedy

# 这一次有多少误差要被学习 a<1
ALPHA = 0.1     # 学习效率

GAMMA = 0.9    # 未来奖励的递减/衰减值

MAX_EPISODES = 13   # 最大回合数

FRESH_TIME = 0.3    # 移动间隔时间

# 初始化
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )
    return table

# 在某个 state 地点, 选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    print state_actions
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()    # 贪婪模式
    print action_name
    return action_name


if __name__ == '__main__':
    q_table = build_q_table(N_STATES, ACTIONS)
    choose_action(0, q_table)
