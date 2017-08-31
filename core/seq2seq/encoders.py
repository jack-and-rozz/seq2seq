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
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

class Encoder(object):
  pass

class RNNEncoder(Encoder):
  def __init__(self, cell, embedding, sequence_length=None,
               scope=None, activation=tf.nn.tanh):
    with tf.variable_scope(scope or "rnn_encoder") as scope:
      self.cell = cell
      self.embedding = embedding
      self.activation = activation
      self.sequence_length = sequence_length
  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size

  def __call__(self, inputs, scope=None, dtype=tf.float32,):
    '''
    inputs assume to be embedded when Seq2Seq is called.
    '''
    with tf.variable_scope(scope or "rnn_encoder") as scope:
      outputs, state = rnn.dynamic_rnn(
        self.cell, tf.stack(inputs, axis=1),
        sequence_length=self.sequence_length,
        scope=scope, dtype=dtype)
    return outputs, state

class BidirectionalRNNEncoder(RNNEncoder):
  def __init__(self, cell, embedding, sequence_length=None,
               scope=None, activation=tf.nn.tanh):
    with tf.variable_scope(scope or "bidirectional_rnn_encoder"):
      self.cell = self.cell_fw = cell
      self.cell_bw = copy.deepcopy(cell)
      self.embedding = embedding
      self.activation = activation
      self.sequence_length=sequence_length

  def __call__(self, inputs, scope=None, dtype=tf.float32):
    with tf.variable_scope(scope or "bidirectional_rnn_encoder"):
      outputs, states = rnn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, tf.stack(inputs, axis=1),
        sequence_length=self.sequence_length, time_major=False,
        scope=scope, dtype=dtype)
      output_fw, output_bw = outputs
      state_fw, state_bw = states
      def merge(size, s_fw, s_bw): # Linearly transform the fw and bw.
        w = tf.get_variable("proj_w", [size * 2, size])
        b = tf.get_variable("proj_b", [size])
        states = self.activation(
          tf.nn.xw_plus_b(tf.concat([s_fw, s_bw], 1), w, b))
        return states

      merged_outputs = []
      for i, (o_fw, o_bw) in enumerate(zip(tf.unstack(output_fw, axis=1), 
                                           tf.unstack(output_bw, axis=1))):
        reuse = True if i > 0 else None
        with tf.variable_scope("outputs", reuse=reuse):
          merged_outputs.append(merge(self.output_size, o_fw, o_bw))
      merged_outputs = tf.stack(merged_outputs, axis=1)
      if nest.is_sequence(self.state_size):
        merged_state = []
        for i, (size, s_fw, s_bw) in enumerate(
            zip(self.state_size, state_fw, state_bw)):
          with tf.variable_scope("state_%d" % (i)):
            merged_state.append(merge(size, s_fw, s_bw))
      else:
        num_layers = len(self.cell._cells)
        size = self.state_size // num_layers
        print size
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
