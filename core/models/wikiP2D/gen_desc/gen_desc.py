# coding: utf-8 
import math, time, sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import sequence_loss, sequence_loss_by_example, rnn_decoder, attention_decoder
from core.utils import common
from core.models.base import ModelBase
from core.seq2seq import rnn
from core.seq2seq.decoders import BeamSearchWrapper, RNNDecoder

from core.seq2seq.seq2seq import projection_and_sampled_loss, _extract_argmax_and_embed, to_logits
from core.seq2seq.beam_search import _extract_beam_search


class DescriptionGeneration(ModelBase):
  def __init__(self, config, encoder, w_vocab,
               activation=tf.nn.tanh, feed_previous=False, 
               beam_size=1, num_samples=512):
    """
    Args:
    """
    self.name = "desc"
    self.dataset = 'wikiP2D'
    self.activation = activation
    self.encoder = encoder
    self.embeddings = encoder.w_embeddings
    self.w_vocab = w_vocab

    self.max_output_length = config.max_output_length.decode

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.w_sentences = tf.placeholder(
        tf.int32, name='w_sentences',
        shape=[None, None]) if self.encoder.wbase else None
      self.c_sentences = tf.placeholder(
        tf.int32, name='c_sentences',
        shape=[None, None, None]) if self.encoder.cbase else None

      # BOS + sentence_length + EOS.
      self.descriptions = desc = tf.placeholder(
        tf.int32, name='descriptions', shape=[None, self.max_output_length+2])

      self.sentence_length = tf.placeholder(tf.int32, shape=[None], name="sentence_length")

    # BOS + sentence_length
    self.decoder_inputs = tf.stack(tf.unstack(desc, axis=1)[:-1], axis=1)
    self.targets = tf.stack(tf.unstack(desc, axis=1)[1:], axis=1)
    self.weights = tf.placeholder(tf.float32, name='weights',
                                  shape=[None, self.max_output_length+1])

    ## Seq2Seq for description generation.
    with tf.variable_scope('Decoder') as scope:
      self.cell = rnn.setup_cell(
        config.cell_type, config.hidden_size,
        num_layers=config.num_layers, 
        in_keep_prob=config.in_keep_prob, 
        out_keep_prob=config.out_keep_prob,
        state_is_tuple=config.state_is_tuple)
      self.decoder = RNNDecoder(self.cell, self.encoder.w_embeddings, scope=scope)
      ########## DEBUG
      self.loss = tf.constant(1.0)
      self.outputs = tf.constant(1.0)
    return
    with tf.variable_scope('Seq2Seq') as scope:
      self.projection, loss_func = projection_and_sampled_loss(
        self.embeddings.shape[0], self.cell.output_size, num_samples)

      self.do_beam_decode = False
      if not feed_previous:
        self.loop_function = None
      else:
        if beam_size > 1:
          self.do_beam_decode = True
          # TODO
        else:
          self.loop_function = _extract_argmax_and_embed(
            self.embeddings, output_projection=self.projection)

      res = self.decoder(
        self.decoder_inputs, self.encoder.link_outputs, self.encoder.outputs,
        scope=scope, loop_function=self.loop_function)
      if self.do_beam_decode:
        beam_paths, beam_symbols, decoder_states = res
      else:
        outputs, decoder_states = res
        logits = to_logits(outputs, self.projection) if self.projection is not None else outputs
        self.losses = sequence_loss_by_example(
          outputs, 
          tf.unstack(self.targets, axis=1), 
          tf.unstack(self.weights, axis=1), 
          softmax_loss_function=loss_func)
        # Empty descriptions ([BOS, EOS]) are not used for loss. 
        mask = tf.cast(tf.greater(tf.reduce_sum(self.weights, axis=1), 1), tf.float32)
        self.losses = self.losses * mask
        self.loss = tf.reduce_sum(self.losses) / tf.reduce_sum(mask)

  def get_input_feed(self, batch):
    input_feed = {}
    #descriptions = batch['descriptions']
    descriptions = [e['desc'] for e in batch['entities']]
    descriptions, sentence_length = self.w_vocab.padding(
      batch['descriptions'], self.max_output_length)

    def to_weights(sequence_lengthes, max_length):
      # Targets don't include BOS, so we need to decrement sentence_length by 1
      return [[1.0]*(sl-1) + [0.0]*(max_length-sl+1) for sl in sequence_lengthes]
    weights = np.array(to_weights(sentence_length, 
                                  self.weights.get_shape()[1]))
    input_feed[self.weights] = weights
    input_feed[self.descriptions] = np.array(descriptions)
    return input_feed
