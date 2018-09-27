#coding=utf-8

import tensorflow as tf
import numpy as np

### 创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

### 搭建模型
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weight * x_data + biases

### 计算误差
loss = tf.reduce_mean(tf.square(y - y_data))

### 传播误差（反向传播）
optimaizer = tf.train.GradientDescentOptimizer(0.5)# 学习效率
train = optimaizer.minimize(loss)

### 训练
init = tf.global_variables_initializer()


### 创建会话
sess = tf.Session()
sess.run(init)

for i in range(200):
    sess.run(train)
    if i % 20 == 0:
        print(i, sess.run(weight), sess.run(biases))