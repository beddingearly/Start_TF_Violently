#coding=utf-8
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
def sess_():
    m1 = tf.constant([[3, 3]])
    m2 = tf.constant([[2], [2]])
    product = tf.matmul(m1, m2)  # multiply

    ## 1
    sess = tf.Session()
    result = sess.run(product)
    sess.close()

    ### 2
    with tf.Session() as sess:
        result = sess.run(product)
        print result

def variable_(): # +1操作
    state = tf.Variable(0, name='counter')
    # print state.name
    one = tf.constant(1)
    new_value = tf.add(state, one)

    update = tf.assign(state, new_value)

    init = tf.initialize_all_variables() #

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

def placeholder_():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1:[7.], input2:[4.]}))


def add_layer(inputs, in_size, out_size, activation_function=None):
    # 增加神经层，确定上层和下层的结构，W的结构是行由输入层决定，列由输出层决定
    # 输入输出层与隐藏层区别在于是否有激励函数
    # in_size 行 out_size 列
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def build():
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)# mean std format
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

    # 隐藏层       输入 输入层1 输出层10
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

    # 输出层
    prediction = add_layer(l1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)# 学习效率
    train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
    train_step = tf.train.MomentumOptimizer(0.1, 0.9).minimize(loss)

    init = tf.global_variables_initializer()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    with tf.Session() as sess:
        sess.run(init)
        #print x_data[:5]
        for i in range(100000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 5 == 0:
                try:
                    ax.lines.pop()
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
                ax.plot(x_data, prediction_value, 'r+')
                plt.pause(0.01)
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                if sess.run(loss, feed_dict={xs: x_data, ys: y_data}) < 0.003:
                    print i
                    break
def test():
    a = tf.Variable(1)
    e = tf.Variable(4)
    b = tf.constant(2)
    d = tf.constant(3)
    init = tf.global_variables_initializer()##
    c = tf.add(a, d)
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a*e))


build()