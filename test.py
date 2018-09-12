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

# 模型类的形式非常简单，主要包含 __init__() (构造函数，初始化)和 call(input) (模型调用)两个方法，
# 但也可以根据需要增加自定义的方法
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs, training=None, mask=None):
        output = ''
        return output

class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer(),
                                           bias_initializer=tf.zeros_initializer())
    def call(self, input):
        output = self.dense(input)
        return output

def test_model_layer():
    tf.enable_eager_execution()
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    model = Linear()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    for i in range(10000):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print model.variables

def test_minist():
    pass



if __name__ == '__main__':
    test_model_layer()


#class A(object):
#     def __init__(self):
#         print 'A'
# class B(A):
#     def __init__(self):
#         super(B, self).__init__()
#         print 'B'