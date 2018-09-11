#coding=utf-8
import tensorflow as tf
import numpy as np
def test_matmul():
    tf.enable_eager_execution()
    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)
    D = tf.add(A, B)
    print D
def test_Grad():
    tf.enable_eager_execution()
    x = tf.get_variable('y', shape=[1], initializer=tf.constant_initializer(3.))
    print x.numpy()
    with tf.GradientTape() as tape:
        y = tf.square(x)
    y_grad = tape.gradient(y, x)
    print y.numpy()
    print y_grad.numpy()

def test_Liner_Regression():
    X_raw = np.array([2013, 2014, 2015, 2016, 2017])
    Y_raw = np.array([12000, 14000, 15000, 16500, 17500])

    #  归一化处理，每个值减最小值 比 最大值减最小值
    X = (X_raw - X_raw.min() / X_raw.max() - X_raw.min())
    Y = (Y_raw - Y_raw.min() / Y_raw.max() - Y_raw.min())

    # 人工求导
    # a, b = 0, 0
    # num_epoch = 10000
    # learning_rate = 1e-3
    # print learning_rate
    # for e in range(num_epoch):
    #
    #     y_pred = a * X + b
    #     grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()
    #     # 更新参数
    #     a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
    # print(a, b)
    X = tf.constant(X)
    y = tf.constant(Y)
    a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    variables  = [a, b]

    num_epoch = 10000
    optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.001)
    for e in range(num_epoch):
        with tf.GradientTape() as tape:
            y_pred = a * X + b
            print y_pred
            loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))

        grads = tape.gradient(loss,  variables)

        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs, training=None, mask=None):
        output = ''
        return output
class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear).__init__()

class A(object):
    def __init__(self):
        print 'A'
class B(A):
    def __init__(self):
        super(A, self).__init__()
        print 'B'



if __name__ == '__main__':
    #test_Liner_Regression()
    b = B()