# coding: utf-8 
import math, time, sys
import numpy as np
import tensorflow as tf

from core.utils.tf_utils import shape, linear, make_summary
from core.utils import common
from core.models.base import ModelBase
from core.seq2seq.rnn import setup_cell
from core.vocabulary.base import PAD_ID

START_TOKEN = PAD_ID
END_TOKEN = PAD_ID

class DescriptionGeneration(ModelBase):
  def __init__(self, config, encoder, vocab,
               activation=tf.nn.relu):
    """
    Args:
    """
    self.config = config
    self.name = config.name
    self.dataset = config.dataset
    self.activation = activation
    self.encoder = encoder
    self.w_embeddings = encoder.w_embeddings
    self.w_vocab = vocab.word

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.is_training = tf.placeholder(tf.bool, [], name='is_training')
      with tf.name_scope('keep_prob'):
        self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

      # encoder's placeholder
      self.w_sentences = tf.placeholder(
        tf.int32, name='w_sentences', shape=[None, None]) 
      self.c_sentences = tf.placeholder(
        tf.int32, name='c_sentences',
        shape=[None, None, None]) if self.encoder.cbase else None
      enc_inputs = [self.w_sentences, self.c_sentences]
      enc_input_lengths = tf.count_nonzero(self.w_sentences, 
                                           axis=1, dtype=tf.int32)

      # decoder's placeholder
      self.descriptions = desc = tf.placeholder(
        tf.int32, name='descriptions', shape=[None, None])


      # add start_token (end_token) to decoder's input (output).
      batch_size = shape(self.descriptions, 0)
      with tf.name_scope('start_tokens'):
        start_tokens = tf.tile(tf.constant([START_TOKEN], dtype=tf.int32), [batch_size])
      with tf.name_scope('end_tokens'):
        end_tokens = tf.tile(tf.constant([END_TOKEN], dtype=tf.int32), [batch_size])
      dec_input_tokens = tf.concat([tf.expand_dims(start_tokens, 1), self.descriptions], axis=1)
      dec_output_tokens = tf.concat([self.descriptions, tf.expand_dims(end_tokens, 1)], axis=1)

      # Length of description + end_token (or start_token)
      dec_input_lengths = dec_output_lengths = tf.count_nonzero(
        self.descriptions, axis=1, dtype=tf.int32) + 1

      # Encode input text
      _, enc_outputs, enc_state = self.encoder.encode(enc_inputs, 
                                                      enc_input_lengths)
      self.logits, self.predictions = self.setup_decoder(
        enc_state, dec_input_tokens, dec_input_lengths, dec_output_lengths)

      # Convert dec_output_lengths to binary masks
      dec_output_weights = tf.sequence_mask(dec_output_lengths, dtype=tf.float32)

      # Compute loss
      self.loss = tf.contrib.seq2seq.sequence_loss(
        self.logits, dec_output_tokens, dec_output_weights,
        average_across_timesteps=True, average_across_batch=True)

  def setup_decoder(self, enc_state, dec_input_tokens, 
                    dec_input_lengths, dec_output_lengths):
    config = self.config
    with tf.variable_scope('Decoder') as scope:
      dec_cell = setup_cell(config.decoder_cell, shape(enc_state, -1), 
                            config.num_layers,
                            keep_prob=self.keep_prob)
      with tf.variable_scope('projection') as scope:
        projection_layer = tf.layers.Dense(self.w_vocab.size, 
                                           use_bias=True, trainable=True)

      dec_input_embs = tf.nn.embedding_lookup(self.w_embeddings, 
                                              dec_input_tokens)

      with tf.name_scope('Train'):
        helper = tf.contrib.seq2seq.TrainingHelper(
          dec_input_embs, sequence_length=dec_input_lengths, time_major=False)
        dec_initial_state = enc_state
        train_decoder = tf.contrib.seq2seq.BasicDecoder(
          dec_cell, helper, dec_initial_state,
          output_layer=projection_layer)
        train_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          train_decoder, impute_finished=True,
          maximum_iterations=tf.reduce_max(dec_output_lengths), scope=scope)
        logits = train_dec_outputs.rnn_output

      with tf.name_scope('Test'):
        beam_width = config.beam_width
        dec_initial_state = tf.contrib.seq2seq.tile_batch(
          enc_state, multiplier=beam_width)
        batch_size = shape(enc_state, 0)
        start_tokens = tf.tile(tf.constant([START_TOKEN], dtype=tf.int32), [batch_size])
        test_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          dec_cell, self.w_embeddings, start_tokens, END_TOKEN, 
          dec_initial_state,
          beam_width, output_layer=projection_layer,
          length_penalty_weight=config.length_penalty_weight)

        test_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          test_decoder, impute_finished=False,
          maximum_iterations=config.max_output_len, scope=scope)
        predictions = test_dec_outputs.predicted_ids
    return logits, predictions

  def get_input_feed(self, batch):
    input_feed = {}
    input_feed[self.is_training] = is_training
    descriptions = [e['desc'] for e in batch['entities']]

    input_feed[self.descriptions] = np.array(descriptions)
    return input_feed
