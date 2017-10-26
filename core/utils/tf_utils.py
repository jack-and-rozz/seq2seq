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


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 2:
    current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
  else:
    current_inputs = inputs

  for i in xrange(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
    current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size])
  outputs = tf.matmul(current_inputs, output_weights) + output_bias

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
  elif len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
  return outputs

def cnn(inputs, filter_sizes, num_filters):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]
