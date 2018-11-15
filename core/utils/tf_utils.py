# coding: utf-8
import math
import shutil
import tensorflow as tf
from tensorflow.python.client import device_lib
from core.utils.common import dbgprint
from collections import defaultdict

def compute_gradients(loss):
  #gradients = [tf.expand_dims(g, 0) if g is not None else g for g in tf.gradients(loss, params)]
  params = tf.contrib.framework.get_trainable_variables()
  gradients = defaultdict(lambda: None)
  for p, g in zip(params, tf.gradients(loss, params)):
    #gradients[p] = tf.expand_dims(g, 0) if g is not None else g
    gradients[p] = g
  return gradients

def sum_gradients(gradients):
  summed_gradients = defaultdict(lambda: None)
  params = tf.contrib.framework.get_trainable_variables()
  for grad in gradients:
    assert type(grad) == defaultdict

  # https://stackoverflow.com/questions/37074077/fail-to-average-indexedslices-on-porting-ptb-word-lm-py-to-multi-gpu-towers
  # Some of the values returned from tf.compute_gradients(...) can be IndexedSlice, which require a trick to compute the sum.
  for p in params:
    grads = [tf.expand_dims(grad[p], 0) for grad in gradients if grad[p] is not None]
    if not grads:
      continue
    grads = tf.concat(grads, axis=0)
    summed_gradients[p] = tf.reduce_sum(grads, axis=0)
    #summed_gradients.append(tf.reduce_sum(grads, axis=0))
  return summed_gradients

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

def batch_gather(emb, indices):
  '''
  e.g. arr = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]  # arr.shape = [2,2,2]
       indices = [[0], [1]]
       res = [[[0, 1], [6, 7]]]

       indices = [[0, 0], [0, 1]]
       res = [[[0, 1], [0, 1]], [[4, 5], [6, 7]]]
  '''

  # Indices of rank-1 tensor is regarded as batch respective specifications.
  if len(indices.get_shape()) == 1:
    indices = tf.expand_dims(indices, 1)
 
  batch_size = shape(emb, 0)
  seqlen = shape(emb, 1)
  if len(emb.get_shape()) > 2:
    emb_size = shape(emb, 2)
  else:
    emb_size = 1
  flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]

  offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]

  gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]
  #if len(emb.get_shape()) == 2:
  #  gathered = tf.squeeze(gathered, 2) # [batch_size, num_indices]
  return gathered

def linear(inputs, output_size=None,
           activation=tf.nn.tanh, scope=None):
  """
  Args:
    inputs : Rank 2 or 3 Tensor of shape [batch_size, (sequence_size,) hidden_size].
    output_size : An integer.
  """
  if activation is None:
    activation = lambda x: x

  if output_size is None:
    output_size = shape(inputs, -1)
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

def cnn(inputs, filter_widths=[3, 4, 5], filter_size=50, scope=None):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  with tf.variable_scope(scope or 'CNN'):
    for i, filter_width in enumerate(filter_widths):
      with tf.variable_scope("conv_width{}".format(filter_width)):
        w = tf.get_variable("w", [filter_width, input_size, filter_size])
        b = tf.get_variable("b", [filter_size])
      conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_width, filter_size]
      h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_width, filter_size]
      pooled = tf.reduce_max(h, 1) # [num_words, filter_size]
      outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, filter_size * len(filter_widths)]

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


def batch_loop(loop_func, *args, n_parallel=10):
  '''
  Args:
  '''
  with tf.name_scope('batch_loop'):
    batch_size = shape(args[0], 0)
    idx = tf.zeros((), dtype=tf.int32)
    res_shape = loop_func(idx, *args).get_shape()
    res = tf.zeros((0, *res_shape)) # A fake value
    cond = lambda idx, res: idx < batch_size
    body = lambda idx, res: (
      idx + 1, 
      tf.concat([res, tf.expand_dims(loop_func(idx, *args), 0)], axis=0),
    )
    loop_vars = [idx, res]
    _, res = tf.while_loop(
      cond, body, loop_vars,
      shape_invariants=[idx.get_shape(),
                        tf.TensorShape([None, *res_shape])],
      parallel_iterations=n_parallel,
    )
    return res


def initialize_embeddings(name, emb_shape, initializer=None, 
                          trainable=True):
  if not initializer:
    initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
  embeddings = tf.get_variable(name, emb_shape, trainable=trainable,
                               initializer=initializer)
  return embeddings
