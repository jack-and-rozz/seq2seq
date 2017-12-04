# coding: utf-8 
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
    #self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.lexical_keep_prob = 1.0 - tf.to_float(self.is_training) * config.lexical_dropout_rate

    if self.wbase:
      w_pretrained = True if isinstance(w_vocab, VocabularyWithEmbedding) else False
      w_initializer = tf.constant_initializer(w_vocab.embeddings) if w_pretrained else None
      w_emb_shape = w_vocab.embeddings.shape if w_pretrained else [w_vocab.size, config.w_embedding_size] 
      w_trainable = config.trainable_emb or not w_pretrained
      self.w_embeddings = self.initialize_embeddings('word_emb', w_emb_shape, initializer=w_initializer, trainable=w_trainable)

    if self.cbase:
      c_pretrained = True if isinstance(c_vocab, VocabularyWithEmbedding) else False
      c_initializer = tf.constant_initializer(c_vocab.embeddings) if c_pretrained else None
      c_emb_shape = w_vocab.embeddings.shape if c_pretrained else [c_vocab.size, config.c_embedding_size] 

      c_trainable = config.trainable_emb or not c_pretrained
      self.c_embeddings = self.initialize_embeddings('char_emb', c_emb_shape, initializer=c_initializer, trainable=c_trainable)

  def encode(self, inputs):
    # inputs: [None, max_sentence_length] or [None, max_sentence_length, max_word_length]
    if len(inputs.get_shape()) == 3: # char-based
      char_repls = tf.nn.embedding_lookup(self.c_embeddings, inputs)
      batch_size = tf.shape(char_repls)[0]
      max_sentence_length = tf.shape(char_repls)[1]
      flattened_char_repls = tf.reshape(char_repls, [batch_size * max_sentence_length, shape(char_repls, 2), shape(char_repls, 3)])
      flattened_aggregated_char_repls = cnn(flattened_char_repls)
      word_repls = tf.reshape(flattened_aggregated_char_repls, [batch_size, max_sentence_length, shape(flattened_aggregated_char_repls, 1)]) # [num_sentences, max_sentence_length, emb_size]
    else: # word-based
      word_repls = tf.nn.embedding_lookup(self.w_embeddings, inputs)
    return tf.nn.dropout(word_repls, self.lexical_keep_prob) # [None, max_sentence_length, emb_size]

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
        word_repls = tf.concat([self.word_encoder.encode(x) for x in wc_sentences], axis=-1)

        if word_repls.get_shape()[-1] != self.hidden_size:
          word_repls = linear(word_repls, self.hidden_size, 
                              activation=self.activation)

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
        outputs = linear(outputs, self.hidden_size, 
                         activation=self.activation)
      with tf.variable_scope("state"):
        state = merge_state(state)
        #state = linear(state, self.hidden_size, 
        #               activation=self.activation)
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

  #def get_input_feed(self, batch):
  #  input_feed = {}
  #  return input_feed




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

