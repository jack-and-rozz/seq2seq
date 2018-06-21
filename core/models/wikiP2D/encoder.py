# coding: utf-8 
import sys
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
import numpy as np

from core.utils import common, tf_utils
from core.utils.tf_utils import shape, cnn, linear
from core.seq2seq.rnn import setup_cell
from core.models.base import ModelBase
from core.vocabulary.base import VocabularyWithEmbedding
from tensorflow.contrib.rnn import LSTMStateTuple

def merge_state(state):
  if isinstance(state[0], LSTMStateTuple):
    new_c = tf.concat([s.c for s in state], axis=1)
    new_h = tf.concat([s.h for s in state], axis=1)
    state = LSTMStateTuple(c=new_c, h=new_h)
  else:
    state = tf.concat(state, 1)
  return state

class WordEncoder(ModelBase):
  def __init__(self, config, is_training, vocab,
               activation=tf.nn.relu, shared_scope=None):
    self.wbase = True if config.vocab_size.word else False
    self.cbase = True if config.vocab_size.char else False
    self.vocab = vocab
    self.is_training = is_training
    self.activation = activation
    self.shared_scope = shared_scope # to reuse variables
    self.reuse = None
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.lexical_dropout_rate

    w_trainable = config.trainable_emb
    sys.stderr.write("Initialize word embeddings with pretrained ones.\n")
    w_initializer = tf.constant_initializer(vocab.word.embeddings)
    w_emb_shape = vocab.word.embeddings.shape

    with tf.device('/cpu:0'):
      self.w_embeddings = self.initialize_embeddings('word_emb', w_emb_shape, initializer=w_initializer, trainable=w_trainable)

    if self.cbase:
      c_emb_shape = [vocab.char.size, config.c_embedding_size] 
      with tf.device('/cpu:0'):
        self.c_embeddings = self.initialize_embeddings(
          'char_emb', c_emb_shape, trainable=True)

  def encode(self, wc_inputs):
    # inputs: the list of [None, max_sentence_length] or [None, max_sentence_length, max_word_length]
    if not isinstance(wc_inputs, list):
      wc_inputs = [wc_inputs]

    outputs = []
    with tf.variable_scope(self.shared_scope or "WordEncoder", reuse=self.reuse):
      for inputs in wc_inputs:
        if inputs is None:
          continue
        if len(inputs.get_shape()) == 3: # char-based
          char_repls = tf.nn.embedding_lookup(self.c_embeddings, inputs)
          batch_size = shape(char_repls, 0)
          max_sentence_length = shape(char_repls, 1)
          flattened_char_repls = tf.reshape(char_repls, [batch_size * max_sentence_length, shape(char_repls, 2), shape(char_repls, 3)])
          flattened_aggregated_char_repls = cnn(flattened_char_repls)
          word_repls = tf.reshape(flattened_aggregated_char_repls, [batch_size, max_sentence_length, shape(flattened_aggregated_char_repls, 1)]) # [num_sentences, max_sentence_length, emb_size]
        else: # word-based
          word_repls = tf.nn.embedding_lookup(self.w_embeddings, inputs)
        outputs.append(word_repls)
      outputs = tf.concat(outputs, axis=-1)
      if self.shared_scope:
        self.reuse = True
    return tf.nn.dropout(outputs, self.keep_prob) # [None, max_sentence_length, emb_size]

  def get_input_feed(self, batch):
    input_feed = {}
    return input_feed

class SentenceEncoder(ModelBase):
  def __init__(self, config, is_training, word_encoder, activation=tf.nn.relu, 
               shared_scope=None):
    self.wbase = True if config.vocab_size.word else False
    self.cbase = True if config.vocab_size.char else False
    self.rnn_size = config.rnn_size
    self.is_training = is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.word_encoder = word_encoder
    self.vocab = word_encoder.vocab
    self.w_embeddings = word_encoder.w_embeddings
    self.c_embeddings = word_encoder.c_embeddings
    self.activation = activation
    self.shared_scope = shared_scope
    self.reuse = None # to reuse variables defined in encode()

    # For 'initial_state' of CustomLSTMCell, different scopes are required in these initializations.
    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(config.encoder_cell, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

    with tf.variable_scope('bw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_bw = setup_cell(config.encoder_cell, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

  def encode(self, wc_sentences, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse) as scope:
      word_repls = self.word_encoder.encode(wc_sentences)

      batch_size = shape(word_repls, 0)
      initial_state_fw = self.cell_fw.initial_state(batch_size) if hasattr(self.cell_fw, 'initial_state') else None
      initial_state_bw = self.cell_fw.initial_state(batch_size) if hasattr(self.cell_bw, 'initial_state') else None

      outputs, state = rnn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, word_repls,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)

      with tf.variable_scope("outputs"):
        outputs = tf.concat(outputs, 2)
        #outputs = linear(outputs, self.hidden_size, 
        #                 activation=self.activation)
        outputs = tf.nn.dropout(outputs, self.keep_prob)
      with tf.variable_scope("state"):
        state = merge_state(state)
        #state = linear(state, self.hidden_size, 
        #               activation=self.activation)
      if self.shared_scope:
        self.reuse = True 
    return word_repls, outputs, state


class MultiEncoderWrapper(SentenceEncoder):
  def __init__(self, encoders):
    """
    Args 
      encoders: a list of SentenceEncoder. 
    """
    self.encoders = encoders
    self.is_training = encoders[0].is_training
    self.vocab = encoders[0].vocab
    self.word_encoder = encoders[0].word_encoder
    self.w_embeddings = encoders[0].w_embeddings
    self.c_embeddings = encoders[0].c_embeddings
    self.cbase = encoders[0].cbase
    self.wbase = encoders[0].wbase

  def encode(self, wc_sentences, sequence_length):
    if not nest.is_sequence(self.encoders):
      return self.encoders.encode(wc_sentences, sequence_length)
    outputs = []
    state = []
    for e in self.encoders:
      word_repls, o, s = e.encode(wc_sentences, sequence_length)
      outputs.append(o)
      state.append(s)
    outputs = tf.concat(outputs, axis=2)
    state = merge_state(state)

    return word_repls, outputs, state
