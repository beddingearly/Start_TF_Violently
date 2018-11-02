#coding=utf-8
'''
@Time    : 2018/10/29 14:30
@Author  : Zt.Wang
@Email   : 137602260@qq.com
@File    : testddpg.py
@Software: PyCharm
'''
import tensorflow as tf
import numpy as np

class DDPG(object):
    def __init__(self):
        self.gamma = 0.9
        self.lr_A = 0.001
        self.lr_C = 0.002




