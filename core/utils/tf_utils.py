#coding: utf-8
import tensorflow as tf

def batch_dot(t1, t2, n_unk_dims=1):
  with tf.name_scope('batch_dot'):
    t1_ex = tf.expand_dims(t1, n_unk_dims)
    t2_ex = tf.expand_dims(t2, n_unk_dims+1)
    return tf.squeeze(tf.matmul(t1_ex, t2_ex), [n_unk_dims, n_unk_dims+1])



def linear_trans_for_seq(seq_repls, output_size, activation=tf.nn.tanh, scope=None):
  """
  Args:
    seq_repls : Rank 3 Tensor of shape [batch_size, sequence_size, hidden_size].
                 the sequence_size must be known.
    output_size : An integer.
  """
  with tf.variable_scope(scope or "linear"):
    hidden_size = seq_repls.get_shape()[-1]
    w = tf.get_variable('weights', [hidden_size, output_size])
    b = tf.get_variable('biases', [output_size])
    return tf.stack([activation(tf.nn.xw_plus_b(t, w, b)) for t in tf.unstack(seq_repls, axis=1)], axis=1)
