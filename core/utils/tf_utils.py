#coding: utf-8
import tensorflow as tf

def batch_dot(t1, t2):
  with tf.name_space('batch_dot'):
    t1_ex = tf.expand_dims(t1, 1)
    t2_ex = tf.expand_dims(t2, 2)
    return tf.squeeze(tf.matmul(t1_ex, t2_ex), [1,2])
