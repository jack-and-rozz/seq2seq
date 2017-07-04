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
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest
from tensorflow.contrib.legacy_seq2seq import sequence_loss, sequence_loss_by_example, rnn_decoder, attention_decoder #, model_with_buckets
from core.seq2seq.decoders import BeamSearchWrapper
from core.seq2seq.beam_search import _extract_beam_search


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

class BasicSeq2Seq(object):
  def __init__(self, encoder, decoder, num_samples, batch_size,
               feed_previous=False, beam_size=1):
    self.encoder = encoder
    self.decoder = decoder
    self.projection, self.loss = projection_and_sampled_loss(
      decoder.embedding.shape[0], decoder.cell.output_size, num_samples)
    self.loop_function = None
    self.do_beam_decode = False
    if feed_previous:
      if beam_size > 1:
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.do_beam_decode = True
        update_embedding_for_previous = False
        self.loop_function = _extract_beam_search(
          decoder.embedding, beam_size, batch_size,
          output_projection=self.projection,
          update_embedding=update_embedding_for_previous)
        self.decoder = BeamSearchWrapper(self.decoder, beam_size, self.projection)
      else:
        self.loop_function = _extract_argmax_and_embed(
          self.decoder.embedding, output_projection=self.projection)
      

  def seq2seq(self, encoder_inputs, decoder_inputs):
    with variable_scope.variable_scope("Encoder") as scope:
      encoder_outputs, encoder_state = self.encoder(
        encoder_inputs, scope=scope)
    with variable_scope.variable_scope("Decoder") as scope:
      return self.decoder(
        decoder_inputs, encoder_state, encoder_outputs,
        scope=scope, loop_function=self.loop_function)


  def __call__(self, encoder_inputs, decoder_inputs, targets, weights,
               per_example_loss=False):
    encoder_inputs = [embedding_ops.embedding_lookup(
      self.encoder.embedding, inp) for inp in encoder_inputs]
    decoder_inputs = [embedding_ops.embedding_lookup(
      self.decoder.embedding, inp) for inp in decoder_inputs]

    if self.do_beam_decode:
      beam_paths, beam_symbols, decoder_states = self.seq2seq(encoder_inputs, 
                                                              decoder_inputs)
      return beam_paths, beam_symbols, decoder_states
    else:
      outputs, decoder_states = self.seq2seq(encoder_inputs, 
                                             decoder_inputs)
      def to_logits(outputs):
        return [tf.nn.xw_plus_b(output, self.projection[0], self.projection[1])
                for output in outputs]
      logits = to_logits(outputs) if self.projection is not None else outputs

      loss_func = sequence_loss_by_example if per_example_loss else sequence_loss
      losses = loss_func(outputs, targets, weights,
                       softmax_loss_function=self.loss)
      return logits, losses, decoder_states


