# coding: utf-8 
import sys
import tensorflow as tf
from core.utils import common, tf_utils
from core.utils.tf_utils import shape, cnn, linear
from core.seq2seq.rnn import setup_cell
from tensorflow.python.ops import rnn

import numpy as np
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
  def __init__(self, config, is_training, w_vocab=None, c_vocab=None,
               activation=tf.nn.tanh):
    self.cbase = config.cbase
    self.wbase = config.wbase
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.hidden_size = config.hidden_size
    self.is_training = is_training
    self.lexical_keep_prob = 1.0 - tf.to_float(self.is_training) * config.lexical_dropout_rate

    if self.wbase:
      w_pretrained = True if isinstance(w_vocab, VocabularyWithEmbedding) else False
      w_trainable = config.trainable_emb or not w_pretrained
      if w_pretrained:
        sys.stderr.write("Initialize word embeddings with the pretrained.\n")
        w_initializer = tf.constant_initializer(w_vocab.embeddings)
        w_emb_shape = w_vocab.embeddings.shape
      else:
        w_initializer = None
        w_emb_shape = [w_vocab.size, config.w_embedding_size] 
      with tf.device('/cpu:0'):
        self.w_embeddings = self.initialize_embeddings('word_emb', w_emb_shape, initializer=w_initializer, trainable=w_trainable)

    if self.cbase:
      c_pretrained = True if isinstance(c_vocab, VocabularyWithEmbedding) else False
      c_trainable = config.trainable_emb or not c_pretrained

      if c_pretrained:
        sys.stderr.write("Initialize character embeddings with the pretrained.\n")
        c_initializer = tf.constant_initializer(c_vocab.embeddings)
        c_emb_shape = c_vocab.embeddings.shape 
      else: 
        c_initializer = None
        c_emb_shape = [c_vocab.size, config.c_embedding_size] 

      with tf.device('/cpu:0'):
        self.c_embeddings = self.initialize_embeddings('char_emb', c_emb_shape, initializer=c_initializer, trainable=c_trainable)

  def encode(self, wc_inputs):
    # inputs: the list of [None, max_sentence_length] or [None, max_sentence_length, max_word_length]
    outputs = []
    for inputs in wc_inputs:
      if inputs is None:
        continue
      if len(inputs.get_shape()) == 3: # char-based
        char_repls = tf.nn.embedding_lookup(self.c_embeddings, inputs)
        batch_size = tf.shape(char_repls)[0]
        max_sentence_length = tf.shape(char_repls)[1]
        flattened_char_repls = tf.reshape(char_repls, [batch_size * max_sentence_length, shape(char_repls, 2), shape(char_repls, 3)])
        flattened_aggregated_char_repls = cnn(flattened_char_repls)
        word_repls = tf.reshape(flattened_aggregated_char_repls, [batch_size, max_sentence_length, shape(flattened_aggregated_char_repls, 1)]) # [num_sentences, max_sentence_length, emb_size]
      else: # word-based
        word_repls = tf.nn.embedding_lookup(self.w_embeddings, inputs)
      outputs.append(word_repls)
    outputs = tf.concat(outputs, axis=-1)
    return tf.nn.dropout(outputs, self.lexical_keep_prob) # [None, max_sentence_length, emb_size]

  def get_input_feed(self, batch):
    input_feed = {}
    return input_feed

class SentenceEncoder(ModelBase):
  def __init__(self, config, is_training, word_encoder, activation=tf.nn.tanh, 
               shared_scope=None):
    self.cbase = config.cbase
    self.wbase = config.wbase
    self.hidden_size = config.hidden_size
    self.is_training = is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.word_encoder = word_encoder
    self.w_vocab = word_encoder.w_vocab
    self.c_vocab = word_encoder.c_vocab
    self.w_embeddings = word_encoder.w_embeddings
    self.c_embeddings = word_encoder.c_embeddings
    self.activation = activation
    self.shared_scope = shared_scope
    do_sharing = True if self.shared_scope else False

    # For 'initial_state' of CustomLSTMCell, different scopes are required in these initializations.
    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(config.cell_type, config.hidden_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob,
                                shared=do_sharing)

    with tf.variable_scope('bw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_bw = setup_cell(config.cell_type, config.hidden_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob,
                                shared=do_sharing)
    self.reuse = None # to reuse variables defined in encode()

  def encode(self, wc_sentences, sequence_length, output_layers=0):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse):
      with tf.variable_scope("Word"):
        word_repls = self.word_encoder.encode(wc_sentences)

        #if word_repls.get_shape()[-1] != self.hidden_size:
        #  word_repls = linear(word_repls, self.hidden_size, 
        #                      activation=self.activation)

      with tf.variable_scope("BiRNN") as scope:
        initial_state_fw = self.cell_fw.initial_state if hasattr(self.cell_fw, 'initial_state') else None
        initial_state_bw = self.cell_bw.initial_state if hasattr(self.cell_bw, 'initial_state') else None

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
      self.reuse = True 
    return word_repls, outputs, state

  # https://stackoverflow.com/questions/44940767/how-to-get-slices-of-different-dimensions-from-tensorflow-tensor
  def extract_span(self, repls, span, entity_indices,
                   max_batch_size):
    with tf.name_scope('ExtractSpan'):
      def loop_func(idx, span_repls, start, end):
        res = tf.reduce_mean(span_repls[idx][start[idx]:end[idx]+1], axis=0)
        return tf.expand_dims(res, axis=0)

      sol, eol = tf.unstack(span, axis=1)
      batch_size = shape(repls, 0)
      hidden_size = shape(repls, -1)
      idx = tf.zeros((), dtype=tf.int32)

      # Continue concatenating the obtained representation of one span in a row of the batch with the results of previous loop (=res).
      res = tf.zeros((0, hidden_size))
      cond = lambda idx, res: idx < batch_size
      body = lambda idx, res: (idx + 1, tf.concat([res, loop_func(idx, repls, sol, eol)], axis=0))
      loop_vars = [idx, res]
      _, res = tf.while_loop(
        cond, body, loop_vars,
        shape_invariants=[idx.get_shape(),
                          tf.TensorShape([None, hidden_size])])
      spans_by_subj = tf.dynamic_partition(res, entity_indices, max_batch_size)

      # Apply max-pooling for spans of an entity.
      spans_by_subj = tf.stack([tf.reduce_max(s, axis=0) for s in spans_by_subj], 
                               axis=0)
      return spans_by_subj

