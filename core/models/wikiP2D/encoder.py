# coding: utf-8 
import tensorflow as tf

from core.utils import common, tf_utils
from core.utils.tf_utils import shape, cnn, linear
from core.seq2seq.rnn import setup_cell
from tensorflow.python.ops import rnn

import numpy as np
from core.models.base import ModelBase

class WordEncoder(ModelBase):
  def __init__(self, config, w_vocab=None, c_vocab=None,
               activation=tf.nn.tanh):
    self.cbase = config.cbase
    self.wbase = config.wbase
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.hidden_size = config.hidden_size
    self.in_keep_prob = config.in_keep_prob
    self.out_keep_prob = config.out_keep_prob

    if self.wbase:
      self.w_embeddings = self.initialize_embeddings('word_emb', [w_vocab.size, config.w_embedding_size])
    if self.cbase:
      self.c_embeddings = self.initialize_embeddings('char_emb', [c_vocab.size, config.c_embedding_size])

  def encode(self, word_sequences):
    # word_sequences: [None, max_sentence_length] or [None, max_sentence_length, max_word_length]
    if len(word_sequences.get_shape()) == 3: # char-based
      char_repls = tf.nn.embedding_lookup(self.c_embeddings, word_sequences)
      batch_size = tf.shape(char_repls)[0]
      max_sentence_length = tf.shape(char_repls)[1]
      flattened_char_repls = tf.reshape(char_repls, [batch_size * max_sentence_length, shape(char_repls, 2), shape(char_repls, 3)])
      flattened_aggregated_char_repls = cnn(flattened_char_repls)
      word_repls = tf.reshape(flattened_aggregated_char_repls, [batch_size, max_sentence_length, shape(flattened_aggregated_char_repls, 1)]) # [num_sentences, max_sentence_length, emb_size]
    else: # word-based
      word_repls = tf.nn.embedding_lookup(self.w_embeddings, word_sequences)
    return word_repls # [None, max_sentence_length, emb_size]

  def get_input_feed(self, batch):
    input_feed = {}
    return input_feed


class SentenceEncoder(ModelBase):
  def __init__(self, config, word_encoder, activation=tf.nn.tanh, 
               shared_scope=None):
    self.cbase = config.cbase
    self.wbase = config.wbase
    self.hidden_size = config.hidden_size
    self.in_keep_prob = config.in_keep_prob
    self.out_keep_prob = config.out_keep_prob
    self.word_encoder = word_encoder
    self.w_vocab = word_encoder.w_vocab
    self.c_vocab = word_encoder.c_vocab
    self.w_embeddings = word_encoder.w_embeddings
    self.c_embeddings = word_encoder.c_embeddings
    self.activation = activation
    self.shared_scope = shared_scope
    do_sharing = True if self.shared_scope else False
    self.cell_fw = setup_cell(config.cell_type, config.hidden_size, 
                              num_layers=config.num_layers, 
                              in_keep_prob=config.in_keep_prob, 
                              out_keep_prob=config.out_keep_prob,
                              state_is_tuple=config.state_is_tuple,
                              shared=do_sharing)
    self.cell_bw = setup_cell(config.cell_type, config.hidden_size, 
                              num_layers=config.num_layers, 
                              in_keep_prob=config.in_keep_prob, 
                              out_keep_prob=config.out_keep_prob,
                              state_is_tuple=config.state_is_tuple,
                              shared=do_sharing)
    self.reuse = None # to reuse variables defined in encode()

  def encode(self, wc_sentences, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse):
      with tf.variable_scope("Word"):
        word_repls = tf.concat([self.word_encoder.encode(x) for x in wc_sentences], axis=-1)

        if word_repls.get_shape()[-1] != self.hidden_size:
          word_repls = linear(word_repls, self.hidden_size, 
                              activation=self.activation,
                              out_keep_prob=self.out_keep_prob)

      with tf.variable_scope("BiRNN") as scope:
        outputs, state = rnn.bidirectional_dynamic_rnn(
          self.cell_fw, self.cell_bw, word_repls,
          sequence_length=sequence_length, dtype=tf.float32, scope=scope)
      with tf.variable_scope("outputs"):
        outputs = linear(tf.concat(outputs, 2), self.hidden_size, 
                         activation=self.activation,
                         out_keep_prob=self.out_keep_prob)
      with tf.variable_scope("state"):
        state = linear(tf.concat(state, 1), self.hidden_size, 
                       activation=self.activation,
                       out_keep_prob=self.out_keep_prob)
      self.reuse = True 
    return outputs, state

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

  def get_input_feed(self, batch):
    input_feed = {}
    return input_feed




    #self.word_length = tf.placeholder(
    #  tf.int32, shape=[None, self.max_sent_length+2], name="word_length")
    # if self.cbase:
    #   with tf.variable_scope('WordEncoder') as scope:
    #     word_encoder = getattr(encoders, config.c_encoder_type)
    #     if word_encoder in [encoders.RNNEncoder, 
    #                         encoders.BidirectionalRNNEncoder]:
    #       self.w_encoder_cell = rnn.setup_cell(
    #         config.cell_type, config.hidden_size,
    #         num_layers=config.num_layers, 
    #         in_keep_prob=config.in_keep_prob, 
    #         out_keep_prob=config.out_keep_prob,
    #         state_is_tuple=config.state_is_tuple,
    #         shared=True)
    #       self.word_encoder = word_encoder(
    #         self.w_encoder_cell, embedding=self.c_embeddings, scope=scope)
    #     elif word_encoder in [encoders.NoneEncoder]:
    #       self.word_encoder = word_encoder(
    #         embedding=self.c_embeddings, scope=scope)

    # with tf.name_scope('EncodeSentence'):
    #   sentences = []
    #   if self.wbase:
    #     sentences.append(self.w_sentences)
    #   if self.cbase:
    #     sentences.append(self.c_sentences)
    #   if not sentences:
    #     raise ValueError('Either FLAGS.wbase or FLAGS.cbase must be True.')
    #   self.outputs, self.states = self.encode_sentence(sentences)
    #   self.link_outputs = self.extract_span(self.outputs, self.link_spans,
    #                                         self.entity_indices)

