# coding: utf-8



import copy, math
import numpy as np
# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import inspect
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest
from tensorflow.contrib.legacy_seq2seq import sequence_loss, sequence_loss_by_example, rnn_decoder, attention_decoder #, model_with_buckets
#from beam_search import _extract_beam_search, beam_rnn_decoder, beam_attention_decoder
from core.seq2seq import beam_search

class Decoder(object):
  pass

class RNNDecoder(Decoder):
  def __init__(self, cell, embedding, scope=None):
    with variable_scope.variable_scope(scope or "rnn_decoder") as scope:
      self.cell = cell
      self.embedding = embedding
      # Decoder keeps its decoder function as a property for extention.
      self.decoder_func = rnn_decoder 
  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size

  def __call__(self, inputs, init_state, _,
               loop_function=None, scope=None):
    with variable_scope.variable_scope(scope or "rnn_decoder") as scope:
      inputs = tf.nn.embedding_lookup(self.embedding, inputs)
      if not nest.is_sequence(inputs):
        inputs = tf.unstack(inputs, axis=1)
      return self.decoder_func(inputs, init_state, self.cell,
                              scope=scope, loop_function=loop_function)

class AttentionDecoder(RNNDecoder):
  def __init__(self, cell, embedding, num_heads=1, scope=None):
    with variable_scope.variable_scope(scope or "attention_decoder") as scope:
      self.cell = cell
      self.embedding = embedding
      self.num_heads = num_heads
      # Decoder keeps its decoder function as a property for extention.
      self.decoder_func = attention_decoder 

  def __call__(self, inputs, init_state, encoder_outputs,
               loop_function=None, scope=None):
    with variable_scope.variable_scope(scope or "attention_decoder") as scope:
      attention_states = encoder_outputs
      return self.decoder_func(inputs, init_state, attention_states, self.cell,
                               num_heads=self.num_heads,
                               scope=scope, loop_function=loop_function)


class BeamSearchWrapper(object):
  def __init__(self, decoder, beam_size, projection):
    self.decoder = decoder
    self.beam_size = beam_size
    self.embedding = decoder.embedding
    self.cell = decoder.cell
    self.projection = projection

  # Decoderクラスへの引数はdecoder関数に共通して渡されるものを優先的に
  def __call__(self, inputs, init_state, encoder_outputs, **kwargs):
    kwargs['beam_size'] = self.beam_size
    kwargs['output_projection'] = self.projection
    if self.decoder.decoder_func == rnn_decoder:
      return beam_search.beam_rnn_decoder(
        inputs, init_state, self.cell, **kwargs)
    elif self.decoder.decoder_func == attention_decoder:
      return beam_search.beam_attention_decoder(
        inputs, init_state, encoder_outputs, self.cell, **kwargs)
