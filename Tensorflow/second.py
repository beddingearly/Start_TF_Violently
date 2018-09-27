#coding=utf-8
import tensorflow as tf
import numpy as np

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

placeholder_()