#coding: utf-8
import shutil
import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in list(value_dict.items())])

def batch_dot(t1, t2, n_unk_dims=1):
  with tf.name_scope('batch_dot'):
    t1_ex = tf.expand_dims(t1, n_unk_dims)
    t2_ex = tf.expand_dims(t2, n_unk_dims+1)
    return tf.squeeze(tf.matmul(t1_ex, t2_ex), [n_unk_dims, n_unk_dims+1])

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def linear(inputs, output_size,
           activation=tf.nn.tanh, scope=None):
  """
  Args:
    inputs : Rank 2 or 3 Tensor of shape [batch_size, (sequence_size,) hidden_size].
    output_size : An integer.
  """
  with tf.variable_scope(scope or "linear"):
    inputs_rank = len(inputs.get_shape().as_list())
    hidden_size = shape(inputs, -1)
    w = tf.get_variable('weights', [hidden_size, output_size])
    b = tf.get_variable('biases', [output_size])
    if inputs_rank == 3:
      batch_size = shape(inputs, 0)
      max_sentence_length = shape(inputs, 1)
      inputs = tf.reshape(inputs, [batch_size * max_sentence_length, hidden_size])
      outputs = activation(tf.nn.xw_plus_b(inputs, w, b))
      outputs = tf.reshape(outputs, [batch_size, max_sentence_length, output_size])
    elif inputs_rank == 2:
      outputs = activation(tf.nn.xw_plus_b(inputs, w, b))
    else:
      ValueError("linear with rank {} not supported".format(inputs_rank))

    #if out_keep_prob is not None and out_keep_prob < 1.0:
    return outputs

def cnn(inputs, filter_sizes=[3, 4, 5], num_filters=50):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  with tf.variable_scope('CNN'):
    for i, filter_size in enumerate(filter_sizes):
    #with tf.variable_scope("conv_{}".format(i)):
      with tf.variable_scope("conv_width{}".format(filter_size)):
        w = tf.get_variable("w", [filter_size, input_size, num_filters])
        b = tf.get_variable("b", [num_filters])
      conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
      h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
      pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
      outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 2:
    current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
  else:
    current_inputs = inputs

  with tf.variable_scope('FFNN'):
    for i in range(num_hidden_layers):
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

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


