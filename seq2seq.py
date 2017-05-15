# coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import copy, math
import numpy as np
# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
# from tensorflow.python.framework import dtypes
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.util import nest

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest
from tensorflow.contrib.legacy_seq2seq import sequence_loss, sequence_loss_by_example, rnn_decoder #, model_with_buckets
# TODO(ebrevdo): Remove once _linear is fully deprecated.
#linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function

def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  outputs = []
  losses = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        loss_func = sequence_loss_by_example if per_example_loss else sequence_loss
        losses.append(
          loss_func(
            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
            softmax_loss_function=softmax_loss_function))
  return outputs, losses

def projection_and_sampled_loss(target_vocab_size, hidden_size, num_samples):
  # If we use sampled softmax, we need an output projection.
  output_projection = None
  softmax_loss_function = None

  if num_samples > 0 and num_samples < target_vocab_size:
    w_t = tf.get_variable("proj_w", [target_vocab_size, hidden_size])
    w = tf.transpose(w_t)
    b = tf.get_variable("proj_b", [target_vocab_size])
    output_projection = (w, b)

    def sampled_loss(labels, logits):
      labels = tf.reshape(labels, [-1, 1])
      # We need to compute the sampled_softmax_loss using 32bit floats to
      # avoid numerical instabilities.
      dtype=tf.float32
      local_w_t = tf.cast(w_t, tf.float32)
      local_b = tf.cast(b, tf.float32)
      local_inputs = tf.cast(logits, tf.float32)
      return tf.cast(
        tf.nn.sampled_softmax_loss(
          weights=local_w_t,
          biases=local_b,
          labels=labels,
          inputs=local_inputs,
          num_sampled=num_samples,
          num_classes=target_vocab_size),
        dtype)
    softmax_loss_function = sampled_loss
  return output_projection, softmax_loss_function


class Encoder(object):
  pass


class Decoder(object):
  pass


class RNNEncoder(Encoder):
  def __init__(self, cell, embedding, sequence_length=None,
               scope=None, activation=math_ops.tanh):
    with variable_scope.variable_scope(scope or "rnn_encoder") as scope:
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
    with variable_scope.variable_scope(scope or "rnn_encoder") as scope:
      embedded = [embedding_ops.embedding_lookup(self.embedding, inp)
                  for inp in inputs]
      # outputs = []
      # batch_size = 200 #embedded[0].get_shape().with_rank_at_least(2)[0]
      # state = self.cell.zero_state(batch_size, tf.float32)
      # with tf.variable_scope(scope or "rnn"):
      #   for time_step in range(len(embedded)):
      #     if time_step > 0: tf.get_variable_scope().reuse_variables()
      #     #(cell_output, state) = cell(sources[:, time_step, :], state)
      #     (cell_output, state) = self.cell(embedded[time_step], state)
      #     outputs.append(cell_output)
      #outputsが全部同じになる。なんかstatic_rnnバグってる？
      outputs, state = core_rnn.static_rnn(
        self.cell, embedded,
        sequence_length=self.sequence_length,
        scope=scope, dtype=dtype)
    return outputs, state

class RNNDecoder(Decoder):
  def __init__(self, cell, embedding, scope=None):
    with variable_scope.variable_scope(scope or "rnn_decoder") as scope:
      self.cell = cell
      self.embedding=embedding
  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size

  def __call__(self, inputs, state, loop_function=None, scope=None):
    with variable_scope.variable_scope(scope or "rnn_decoder") as scope:
      embedded = [embedding_ops.embedding_lookup(
        self.embedding, inp) for inp in inputs]
      outputs = []
      prev = None
      for i, inp in enumerate(embedded):
        if loop_function is not None and prev is not None:
          with variable_scope.variable_scope("loop_function", reuse=True):
            inp = loop_function(prev, i)
        if i > 0:
          variable_scope.get_variable_scope().reuse_variables()
        output, state = self.cell(inp, state)
        outputs.append(output)
        if loop_function is not None:
          prev = output
    return outputs, state

class BidirectionalRNNEncoder(RNNEncoder):
  def __init__(self, cell, embedding, sequence_length=None,
               scope=None, activation=math_ops.tanh):
    with variable_scope.variable_scope(scope or "bidirectional_rnn_encoder"):
      self.cell = self.cell_fw = cell
      self.cell_bw = copy.deepcopy(cell)
      self.embedding = embedding
      self.activation = activation
      self.sequence_length=sequence_length

  def __call__(self, inputs, scope=None, dtype=np.float32):
    with variable_scope.variable_scope(scope or "bidirectional_rnn_encoder"):
      embedded = [embedding_ops.embedding_lookup(
        self.embedding, inp) for inp in inputs]
      outputs, state_fw, state_bw = core_rnn.static_bidirectional_rnn(
        self.cell_fw, self.cell_bw, embedded,
        sequence_length=self.sequence_length,
        scope=scope, dtype=dtype)

      def merge_state(size, s_fw, s_bw):
        w = tf.get_variable("proj_w", [size * 2, size])
        b = tf.get_variable("proj_b", [size])
        states = self.activation(
          tf.nn.xw_plus_b(array_ops.concat([s_fw, s_bw], 1), w, b))
        return states
      if nest.is_sequence(self.state_size):
        state = []
        for i, (size, s_fw, s_bw) in enumerate(
            zip(self.state_size, state_fw, state_bw)):
          with variable_scope.variable_scope("cell_%d" % (i)):
            state.append(merge_state(size, s_fw, s_bw))
      else:
        state = merge_state(self.state_size, state_fw, state_bw)
      return outputs, state

class BasicSeq2Seq(object):
  def __init__(self, encoder, decoder, num_samples, feed_previous=False):
    self.encoder = encoder
    self.decoder = decoder
    self.projection, self.loss = projection_and_sampled_loss(
      decoder.embedding.shape[0], decoder.cell.output_size, num_samples)
    self.loop_function = None
    if feed_previous:
      self.loop_function = _extract_argmax_and_embed(
        self.decoder.embedding, output_projection=self.projection)

  def seq2seq(self, encoder_inputs, decoder_inputs):
    with variable_scope.variable_scope("Encoder") as scope:
      encoder_outputs, encoder_state = self.encoder(
        encoder_inputs, scope=scope)
    with variable_scope.variable_scope("Decoder") as scope:
      decoder_outputs, decoder_state = self.decoder(
        decoder_inputs, encoder_state, scope=scope,
        loop_function=self.loop_function)
    return decoder_outputs, decoder_state

  def __call__(self, encoder_inputs, decoder_inputs, targets, weights,
               buckets, per_example_loss=False):
    outputs, losses = model_with_buckets(
      encoder_inputs, decoder_inputs, targets, weights, buckets,
      self.seq2seq, per_example_loss=per_example_loss, 
      softmax_loss_function=self.loss)

    
    def to_logits(outputs):
      return [tf.nn.xw_plus_b(output, self.projection[0], self.projection[1])
              for output in outputs]


    logits = [to_logits(o) for o in outputs] if self.projection is not None else outputs
    return logits, losses

