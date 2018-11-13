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
from core.models.wikiP2D.decoder import RNNDecoder

START_TOKEN = PAD_ID
END_TOKEN = PAD_ID


class DescModelBase(ModelBase):
  pass

class DescriptionGeneration(DescModelBase):
  def __init__(self, sess, config, manager, activation=tf.nn.relu):
    """
    Args:
    """
    super(DescriptionGeneration, self).__init__(sess, config)
    self.config = config
    self.activation = activation
    self.dataset = config.dataset
    self.train_shared = config.train_shared

    # Encoder
    self.vocab = manager.vocab
    print('restoring shared layers in desc')
    shared_layers = manager.restore_shared_layers()
    #shared_layers = manager.shared_layers

    self.is_training = shared_layers.is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.encoder = self.setup_encoder(shared_layers.encoder, 
                                      manager.use_local_rnn)

    # Placeholders
    self.ph = self.setup_placeholders()

    enc_sentence_length = tf.count_nonzero(self.ph.text.word, 
                                           axis=-1, dtype=tf.int32)
    enc_context_length =  tf.count_nonzero(enc_sentence_length, 
                                           axis=-1, dtype=tf.float32)

    word_repls = self.encoder.word_encoder.word_encode(self.ph.text.word)
    char_repls = self.encoder.word_encoder.char_encode(self.ph.text.char)
    enc_inputs = [word_repls, char_repls]
    # Encode input text
    enc_inputs, enc_outputs, enc_state = self.encoder.encode(
      enc_inputs, enc_sentence_length, prop_gradients=self.train_shared)

    mention_starts, mention_ends = tf.unstack(self.ph.link, axis=-1)
    mention_repls, head_scores = self.encoder.get_batched_mention_emb(
      enc_inputs, enc_outputs, mention_starts, mention_ends) # [batch_size, max_n_contexts, mention_size]
    if not self.train_shared:
      mention_repls = tf.stop_gradient(mention_repls)
      head_scores = tf.stop_gradient(head_scores)

    # Aggregate context representations.
    init_state = tf.reduce_sum(mention_repls, axis=1)
    init_state = init_state / tf.expand_dims(enc_context_length, -1)

    with tf.variable_scope('Intermediate'):
      if config.decoder.num_layers > 1:
        for i in range(config.decoder.num_layers):
          init_states = []
          with tf.variable_scope('L%d' % (i+i)):
            init_states.append(linear(init_state, config.decoder.rnn_size))
        init_state = init_states
      else:
        init_state = linear(init_state, config.decoder.rnn_size)

    # Add BOS and EOS to the decoder's inputs and outputs.
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

    with tf.variable_scope('Decoder') as scope:
      self.decoder = RNNDecoder(config.decoder, self.is_training, 
                                self.vocab.decoder, shared_scope=scope)
      self.logits = self.decoder.decode_train(init_state, dec_input_tokens, 
                                              dec_input_lengths, 
                                              dec_output_lengths)
      self.predictions = self.decoder.decode_test(init_state)

    # Convert dec_output_lengths to binary masks
    dec_output_weights = tf.sequence_mask(dec_output_lengths, dtype=tf.float32)

    # Compute loss
    self.loss = tf.contrib.seq2seq.sequence_loss(
      self.logits, dec_output_tokens, dec_output_weights,
      average_across_timesteps=True, average_across_batch=True)
    #self.debug_ops = [self.ph.text.word, enc_sentence_length, enc_context_length]

    with tf.name_scope('AdversarialInputs'):
      self.adv_inputs = tf.reshape(tf.reduce_mean(enc_outputs, axis=1), 
                                   [-1, shape(enc_outputs, -1)])

  def setup_placeholders(self):
    # Placeholders
    with tf.name_scope('Placeholder'):
      ph = recDotDefaultDict()
      # encoder's placeholder
      ph.text.word = tf.placeholder( 
        tf.int32, name='text.word', shape=[None, None, None]) # [batch_size, n_max_contexts, n_max_word]
      ph.text.char = tf.placeholder(
        tf.int32, name='text.char',
        shape=[None, None, None, None]) if self.encoder.cbase else None # [batch_size, n_max_contexts, n_max_word, n_max_char]

      ph.link = tf.placeholder( 
        tf.int32, name='link.position', shape=[None, None, 2]) # [batch_size, n_max_contexts, 2]

      # decoder's placeholder
      ph.target = tf.placeholder(
        tf.int32, name='descriptions', shape=[None, None])
    return ph

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
    results = flatten([r.tolist() for r in results])
    sys.stdout = open(output_path, 'w') if output_path else sys.stdout
    bleu = evaluate_and_print(used_batches, results, 
                              vocab=self.vocab)
    if output_path:
      sys.stderr.write("Output the testing results to \'{}\' .\n".format(output_path))
    sys.stdout = sys.__stdout__
    summary_dict = {}
    summary_dict['desc/%s/BLEU' % mode] = bleu
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
