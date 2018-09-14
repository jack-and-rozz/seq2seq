# coding: utf-8 
import math, time, sys
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from core.utils.tf_utils import shape, linear, make_summary
from core.utils.common import dbgprint, recDotDefaultDict, flatten, flatten_recdict, flatten_batch
from core.models.base import ModelBase
from core.models.wikiP2D.desc.evaluation import evaluate_and_print
from core.seq2seq.rnn import setup_cell
from core.vocabulary.base import PAD_ID

START_TOKEN = PAD_ID
END_TOKEN = PAD_ID

class DescriptionGeneration(ModelBase):
  def __init__(self, sess, config, encoder,
               activation=tf.nn.relu):
    """
    Args:
    """
    super(DescriptionGeneration, self).__init__(sess, config)
    self.config = config
    self.dataset = config.dataset
    self.activation = activation
    self.encoder = encoder
    self.vocab = encoder.vocab
    self.w_embeddings = encoder.w_embeddings
    self.is_training = encoder.is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.ph = recDotDefaultDict()
      # encoder's placeholder
      self.ph.text.word = tf.placeholder( 
        tf.int32, name='text.word', shape=[None, None, None]) # [batch_size, n_max_contexts, n_max_word]
      self.ph.text.char = tf.placeholder(
        tf.int32, name='text.char',
        shape=[None, None, None, None]) if self.encoder.cbase else None # [batch_size, n_max_contexts, n_max_word, n_max_char]

      self.ph.link = tf.placeholder( 
        tf.int32, name='link.position', shape=[None, None, 2]) # [batch_size, n_max_contexts, 2]

      enc_sentence_length = tf.count_nonzero(self.ph.text.word, 
                                             axis=-1, dtype=tf.int32)
      enc_context_length =  tf.count_nonzero(enc_sentence_length, 
                                             axis=-1, dtype=tf.float32)

      # decoder's placeholder
      self.ph.target = tf.placeholder(
        tf.int32, name='descriptions', shape=[None, None])


      # add start_token (end_token) to decoder's input (output).
      batch_size = shape(self.ph.target, 0)
      with tf.name_scope('start_tokens'):
        start_tokens = tf.tile(tf.constant([START_TOKEN], dtype=tf.int32), [batch_size])
      with tf.name_scope('end_tokens'):
        end_tokens = tf.tile(tf.constant([END_TOKEN], dtype=tf.int32), [batch_size])
      dec_input_tokens = tf.concat([tf.expand_dims(start_tokens, 1), self.ph.target], axis=1)
      dec_output_tokens = tf.concat([self.ph.target, tf.expand_dims(end_tokens, 1)], axis=1)

      # Length of description + end_token (or start_token)
      dec_input_lengths = dec_output_lengths = tf.count_nonzero(
        self.ph.target, axis=1, dtype=tf.int32) + 1
      word_repls = encoder.word_encoder.word_encode(self.ph.text.word)
      char_repls = encoder.word_encoder.char_encode(self.ph.text.char)
      enc_inputs = [word_repls, char_repls]
      dbgprint(enc_inputs)
      # Encode input text
      enc_inputs, enc_outputs, enc_state = self.encoder.encode(enc_inputs, 
                                                               enc_sentence_length)
      dbgprint(enc_state)
      dbgprint(enc_outputs)
      mention_starts, mention_ends = tf.unstack(self.ph.link, axis=-1)
      mention_repls, head_scores = encoder.get_batched_mention_emb(
        enc_inputs, enc_outputs, mention_starts, mention_ends) # [batch_size, max_n_contexts, mention_size]
      dbgprint(mention_starts)
      dbgprint(mention_ends)
      dbgprint(mention_repls)
      # Aggregate context representations.
      init_state = tf.reduce_sum(mention_repls, axis=1)
      init_state = init_state / tf.expand_dims(enc_context_length, -1)
      dbgprint(init_state)
      dbgprint(enc_context_length)

      self.logits, self.predictions = self.setup_decoder(
        init_state, dec_input_tokens, dec_input_lengths, dec_output_lengths)

      # Convert dec_output_lengths to binary masks
      dec_output_weights = tf.sequence_mask(dec_output_lengths, dtype=tf.float32)

      # Compute loss
      self.loss = tf.contrib.seq2seq.sequence_loss(
        self.logits, dec_output_tokens, dec_output_weights,
        average_across_timesteps=True, average_across_batch=True)
      self.debug_ops = [self.ph.text.word, enc_sentence_length, enc_context_length]

  def setup_decoder(self, init_state, dec_input_tokens, 
                    dec_input_lengths, dec_output_lengths):
    config = self.config
    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
      state_size = shape(init_state, -1)
      dec_cell = setup_cell(config.decoder.cell, state_size, 
                            config.decoder.num_layers,
                            keep_prob=self.keep_prob)
      with tf.variable_scope('projection') as scope:
        projection_layer = tf.layers.Dense(self.vocab.word.size, 
                                           use_bias=True, trainable=True)

      dec_input_embs = tf.nn.embedding_lookup(self.w_embeddings, 
                                              dec_input_tokens)

      with tf.name_scope('Train'):
        helper = tf.contrib.seq2seq.TrainingHelper(
          dec_input_embs, sequence_length=dec_input_lengths, time_major=False)
        dec_initial_state = init_state
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
          init_state, multiplier=beam_width)
        batch_size = shape(dec_input_tokens, 0)
        start_tokens = tf.tile(tf.constant([START_TOKEN], dtype=tf.int32), [batch_size])
        test_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          dec_cell, self.w_embeddings, start_tokens, END_TOKEN, 
          dec_initial_state,
          beam_width, output_layer=projection_layer,
          length_penalty_weight=config.length_penalty_weight)

        test_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          test_decoder, impute_finished=False,
          maximum_iterations=config.decoder.max_output_len, scope=scope)
        predictions = test_dec_outputs.predicted_ids # [batch_size, T, beam_width]
        predictions = tf.transpose(predictions, perm = [0, 2, 1]) # [batch_size, beam_width, T]
    return logits, predictions

  def test(self, batches, mode, logger, output_path):
    results = []
    used_batches = []
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      outputs = self.sess.run(self.predictions, input_feed)
      try:
        used_batches += flatten_batch(batch)
      except Exception as e:
        pprint(batch)
        print(e)
        exit(1)
      results.append(outputs[:, 0, :])
    results = np.concatenate(results, axis=0)
    sys.stdout = open(output_path, 'w') if output_path else sys.stdout
    bleu = evaluate_and_print(used_batches, results, 
                              vocab=self.encoder.vocab)
    if output_path:
      sys.stderr.write("Output the testing results to \'{}\' .\n".format(output_path))
    sys.stdout = sys.__stdout__
    summary_dict = {}
    summary_dict['category/%s/BLEU' % mode] = bleu
    summary = make_summary(summary_dict)
    return bleu, summary


  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    input_feed[self.ph.text.word] = batch.contexts.word
    input_feed[self.ph.text.char] = batch.contexts.char
    input_feed[self.ph.link] = batch.contexts.link
    input_feed[self.ph.target] = batch.desc.word
    return input_feed
