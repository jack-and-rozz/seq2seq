# coding: utf-8 
import math, time, sys
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from core.utils.tf_utils import shape, linear, make_summary, initialize_embeddings
from core.utils.common import dbgprint, recDotDefaultDict, flatten, flatten_recdict, flatten_batch
from core.models.base import ModelBase
from core.seq2seq.rnn import setup_cell
from core.vocabulary.base import PAD_ID

class RNNDecoder(ModelBase):
  def __init__(self, config, is_training, vocab, embeddings=None,
               activation=tf.nn.relu, shared_scope=None):
    self.is_training = is_training
    self.activation = activation
    self.shared_scope = shared_scope
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.cell_type = config.cell
    self.num_layers = config.num_layers
    self.beam_width = config.beam_width
    self.length_penalty_weight = config.length_penalty_weight
    self.max_output_len = config.max_output_len

    if embeddings:
      self.embeddings = embeddings 
    else:
      with tf.device('/cpu:0'):
        self.embeddings = initialize_embeddings(
          'word_emb', 
          vocab.word.embeddings.shape, 
          initializer=tf.constant_initializer(vocab.word.embeddings), 
          trainable=vocab.word.trainable)

  def decode_train(self, init_state, dec_input_tokens, 
                   dec_input_lengths, dec_output_lengths):
    with tf.variable_scope(self.shared_scope or "RNNDecoder", 
                           reuse=tf.AUTO_REUSE) as scope:
      state_size = shape(init_state, -1)

      self.cell = setup_cell(self.cell_type, state_size, 
                             self.num_layers,
                             keep_prob=self.keep_prob)

      with tf.variable_scope('projection') as scope:
        self.projection = tf.layers.Dense(shape(self.embeddings, 0), 
                                          use_bias=True, trainable=True)

      with tf.name_scope('Train'):
        dec_input_embs = tf.nn.embedding_lookup(self.embeddings, 
                                                dec_input_tokens)
        helper = tf.contrib.seq2seq.TrainingHelper(
          dec_input_embs, sequence_length=dec_input_lengths, time_major=False)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(
          self.cell, helper, init_state,
          output_layer=self.projection)
        train_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          train_decoder, impute_finished=True,
          maximum_iterations=tf.reduce_max(dec_output_lengths), scope=scope)
        logits = train_dec_outputs.rnn_output
    return logits


  def decode_test(self, init_state, start_token=PAD_ID, end_token=PAD_ID):
    with tf.variable_scope(self.shared_scope or "RNNDecoder", 
                           reuse=tf.AUTO_REUSE) as scope:
      with tf.name_scope('Test'):
        tiled_init_state = tf.contrib.seq2seq.tile_batch(
          init_state, multiplier=self.beam_width)
        batch_size = shape(init_state, 0)
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), 
                               [batch_size])
        test_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          self.cell, self.embeddings, start_tokens, end_token, 
          tiled_init_state,
          self.beam_width, output_layer=self.projection,
          length_penalty_weight=self.length_penalty_weight)

        test_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          test_decoder, impute_finished=False,
          maximum_iterations=self.max_output_len, scope=scope)
        predictions = test_dec_outputs.predicted_ids # [batch_size, T, beam_width]
        predictions = tf.transpose(predictions, perm = [0, 2, 1]) # [batch_size, beam_width, T]
    #return predictions
    return predictions

