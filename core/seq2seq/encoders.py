# coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import copy, math
# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.ops import rnn
#from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

class Encoder(object):
  pass

class RNNEncoder(Encoder):
  def __init__(self, cell, embedding, 
               scope=None, activation=tf.nn.tanh):
    with tf.variable_scope(scope or "rnn_encoder") as scope:
      self.cell = cell
      self.embedding = embedding
      self.activation = activation

  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size

  def __call__(self, inputs, sequence_length=None, time_major=False,
               scope=None, dtype=tf.float32):
    '''
    inputs assume to be embedded when Seq2Seq is called.
    '''
    with tf.variable_scope(scope or "rnn_encoder") as scope:
      if nest.is_sequence(inputs):
        inputs = tf.stack(inputs, axis=1)
      outputs, state = rnn.dynamic_rnn(
        self.cell, inputs, time_major=time_major,
        sequence_length=sequence_length,
        scope=scope, dtype=dtype)
    return outputs, state

class BidirectionalRNNEncoder(RNNEncoder):
  def __init__(self, cell, embedding, 
               scope=None, activation=tf.nn.tanh):
    with tf.variable_scope(scope or "bidirectional_rnn_encoder"):
      self.cell = self.cell_fw = cell
      self.cell_bw = copy.deepcopy(cell)
      self.embedding = embedding
      self.activation = activation

  def __call__(self, inputs, sequence_length=None, merge_type='avg',
               scope=None, dtype=tf.float32, time_major=False):
    """
    Args:
      merge_type: How to merge the output and state from forward and backward RNN.
                  Values can only be 'avg', 'linear', or None.
    """
    if merge_type not in ['avg', 'linear', None]:
      raise ValueError('merge_type must be \'avg\', \'linear\', or None.')
    with tf.variable_scope(scope or "bidirectional_rnn_encoder"):
      if nest.is_sequence(inputs):
        inputs = tf.stack(inputs, axis=1)
      outputs, state = rnn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, inputs,
        sequence_length=sequence_length, time_major=time_major,
        scope=scope, dtype=dtype)
      output_fw, output_bw = outputs
      state_fw, state_bw = state

      if not merge_type:
        return outputs, state
      elif merge_type == 'avg':
        #print outputs
        outputs = (output_fw + output_bw) / 2
        #if isinstance(self.cell, rnn_cell.MultiRNNCell):
        if nest.is_sequence(self.state_size):
          state = ((s_fw + s_bw /2) for s_fw, s_bw in zip(state_fw, state_bw))
          state = tuple(state)
        else:
          state = (state_fw + state_bw) / 2
        return outputs, state

      # merge outputs and state from forward RNN and backword RNN.
      def linear_merge(size, t1, t2): # Linearly transform the fw and bw.
        axis = len(t1.get_shape()) - 1 
        if not t1.get_shape() != t2.get_shape():
          raise ValueError('Two merged tensors must have a same shape.')
        w = tf.get_variable("proj_w", [size * 2, size])
        b = tf.get_variable("proj_b", [size])
        s = self.activation(
          tf.nn.xw_plus_b(tf.concat([t1, t2], axis=axis), w, b))
        return s

      merged_outputs = []
      for i, (o_fw, o_bw) in enumerate(zip(tf.unstack(output_fw, axis=1), 
                                           tf.unstack(output_bw, axis=1))):
        reuse = True if i > 0 else None
        with tf.variable_scope("outputs", reuse=reuse):
          merged_outputs.append(linear_merge(self.output_size, o_fw, o_bw))
      merged_outputs = tf.stack(merged_outputs, axis=1)

      if nest.is_sequence(self.state_size):
        merged_state = []
        for i, (size, s_fw, s_bw) in enumerate(
            zip(self.state_size, state_fw, state_bw)):
          with tf.variable_scope("state_%d" % (i)):
            merged_state.append(linear_merge(size, s_fw, s_bw))
      else:
        num_layers = len(self.cell._cells) if isinstance(self.cell, rnn_cell.MultiRNNCell) else 1
        size = self.state_size // num_layers
        merged_state = []
        state_fw = tf.split(state_fw, [size for _ in xrange(num_layers)], 1)
        state_bw = tf.split(state_bw, [size for _ in xrange(num_layers)], 1)
        for i, (s_fw, s_bw) in enumerate(zip(state_fw, state_bw)):
          with tf.variable_scope("state_%d" % (i)):
            w = tf.get_variable("proj_w", [size * 2, size])
            b = tf.get_variable("proj_b", [size])
            merged_state.append(self.activation(
              tf.nn.xw_plus_b(tf.concat([s_fw, s_bw], 1), w, b)))

        merged_state = tf.concat(merged_state, 1)
      return merged_outputs, merged_state
