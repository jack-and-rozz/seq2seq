# coding: utf-8 
import sys
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
import numpy as np

from core.utils.common import dbgprint
from core.utils.tf_utils import shape, cnn, linear, projection, batch_gather, batch_loop 
from core.seq2seq.rnn import setup_cell
from core.models.base import ModelBase
from core.vocabulary.base import VocabularyWithEmbedding
from tensorflow.contrib.rnn import LSTMStateTuple

def merge_state(state, merge_func=tf.concat):
  if isinstance(state[0], LSTMStateTuple):
    #new_c = merge_func([s.c for s in state], axis=1)
    #new_h = merge_func([s.h for s in state], axis=1)
    new_c = merge_func([s.c for s in state], axis=-1)
    new_h = merge_func([s.h for s in state], axis=-1)
    state = LSTMStateTuple(c=new_c, h=new_h)
  else:
    #state = merge_func(state, 1)
    state = merge_func(state, -1)
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
      c_emb_shape = [vocab.char.size, config.embedding_size.char] 
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
    self.is_training = is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.use_boundary = config.use_boundary
    self.model_heads = config.model_heads

    self.word_encoder = word_encoder
    self.vocab = word_encoder.vocab
    self.wbase = word_encoder.wbase
    self.cbase = word_encoder.cbase
    self.w_embeddings = word_encoder.w_embeddings
    self.c_embeddings = word_encoder.c_embeddings
    self.activation = activation
    self.shared_scope = shared_scope

    self.reuse_encode = None # to reuse variables defined in encode()
    self.reuse_mention = None # to reuse variables defined in encode()

    # For 'initial_state' of CustomLSTMCell, different scopes are required in these initializations.

    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(config.cell, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

    with tf.variable_scope('bw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_bw = setup_cell(config.cell, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

  def encode(self, wc_sentences, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse_encode) as scope:
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
        outputs = tf.nn.dropout(outputs, self.keep_prob)
      with tf.variable_scope("state"):
        state = merge_state(state)
      if self.shared_scope:
        self.reuse_encode = True 
    return word_repls, outputs, state

  def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    input_rank = len(text_emb.get_shape())
    if input_rank == 2:
      return self._get_mention_emb(text_emb, text_outputs, mention_starts, mention_ends)
    elif input_rank == 3:
      mention_starts = tf.expand_dims(mention_starts, 1)
      mention_ends = tf.expand_dims(mention_ends, 1)
      # def loop_func(idx, *args):
      #   args = [v[idx] for v in args]
      #   res, _ = self._get_mention_emb(*args) # [1, emb_size]
      #   return res
      # res = batch_loop(loop_func, text_emb, text_outputs, mention_starts, mention_ends)
      with tf.name_scope('batch_loop'):
        def loop_func(idx, *args):
          args = [v[idx] for v in args]
          res, _ = self._get_mention_emb(*args) # [1, emb_size]
        return res

        batch_size = shape(text_emb, 0)
        idx = tf.zeros((), dtype=tf.int32) # Index for loop counter
        res_shape = loop_func(idx, text_emb, text_outputs, mention_starts, mention_ends).get_shape()
        res = tf.zeros((0, *res_shape)) # A fake tensor with the same shape as output
        cond = lambda idx, res: idx < batch_size
        body = lambda idx, res: (
          idx + 1, 
          tf.concat([res, tf.expand_dims(loop_func(idx, text_emb, text_outputs, mention_starts, mention_ends), 0)], axis=0),
        )

        loop_vars = [idx, res]
        _, res = tf.while_loop(
          cond, body, loop_vars,
          shape_invariants=[idx.get_shape(),
                            tf.TensorShape([None, *res_shape])]
        ) # res: [batch_size, 1, emb_size]
      dbgprint(res)
      exit(1)
      hidden_size = shape(res, -1)
      mention_emb = tf.reshape(res, [batch_size, hidden_size])
      return mention_emb, None
    else:
      raise ValueError('Tensor with rank > 3 is not supported')

  def _get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    '''
    Extract multiple mention representations from a text.

    Args:
    - text_emb: [num_words, dim(word_emb + char_emb)]
    - text_outputs: [num_words, dim(encoder_outputs)]
    - mention_starts, mention_ends: [num_mentions]
    Return:
    - mention_emb: [num_mentions, emb]
    - head_scores: [num_words, 1]
    '''

    # dbgprint(text_emb)
    # dbgprint(text_outputs)
    # dbgprint(mention_starts)
    # dbgprint(mention_ends)
    # exit(1)

    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse_mention) as scope:
      with tf.variable_scope('get_mention_emb'):
        mention_emb_list = []
        mention_width = 1 + mention_ends - mention_starts # [num_mentions]
        max_mention_width = tf.reduce_max(mention_width)

        if self.use_boundary:
          with tf.name_scope('mention_boundary'):
            mention_start_emb = tf.gather(text_outputs, mention_starts) #[num_mentions, emb]
            mention_end_emb = tf.gather(text_outputs, mention_ends) #[num_mentions, emb]
            mention_emb_list.append(mention_start_emb)
            mention_emb_list.append(mention_end_emb)

        if self.model_heads:
          with tf.name_scope('mention_attention'):
            mention_indices = tf.expand_dims(tf.range(max_mention_width), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
            mention_indices = tf.minimum(shape(text_outputs, 0) - 1, mention_indices) # [num_mentions, max_mention_width]

            mention_text_emb = tf.gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]

            head_scores = projection(text_outputs, 1) # [num_words, 1]
            mention_head_scores = tf.gather(head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
            mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, max_mention_width, dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]

            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), dim=1) # [num_mentions, max_mention_width, 1]
            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)
        mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]

      if self.shared_scope:
        self.reuse_mention = True 

      return mention_emb, head_scores

  def _get_batched_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    '''
    Extract each mention representation from batched texts. Each text has only one mention.

    Args:
    - text_emb: [batch_size, num_words, dim(word_emb + char_emb)]
    - text_outputs: [batch_size, num_words, dim(encoder_outputs)]
    - mention_starts, mention_ends: [batch_size] 
    '''
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse_mention) as scope:
      with tf.variable_scope('get_mention_emb'):
        mention_emb_list = []
        mention_width = 1 + mention_ends - mention_starts # [num_mentions]
        max_mention_width = tf.reduce_max(mention_width)

        if self.use_boundary:
          with tf.name_scope('mention_boundary'):
            dbgprint(text_outputs, mention_starts, mention_ends)
            mention_start_emb = batch_gather(text_outputs, mention_starts) #[batch_size, emb]
            mention_end_emb = batch_gather(text_outputs, mention_ends) #[batch_size, emb]
            mention_emb_list.append(mention_start_emb)
            mention_emb_list.append(mention_end_emb)
            dbgprint(mention_start_emb, mention_end_emb)

        if self.model_heads:
          with tf.name_scope('mention_attention'):
            mention_indices = tf.expand_dims(tf.range(max_mention_width), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
            mention_indices = tf.minimum(shape(text_outputs, 1) - 1, mention_indices) # [num_mentions, max_mention_width]

            dbgprint(text_emb)
            dbgprint(mention_indices)
            exit(1)
            mention_text_emb = batch_gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]

            head_scores = projection(text_outputs, 1) # [num_words, 1]
            mention_head_scores = batch_gather(head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
            mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, max_mention_width, dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]

            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), dim=1) # [num_mentions, max_mention_width, 1]
            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)
        dbgprint(mention_emb_list)
        mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]

      if self.shared_scope:
        self.reuse_mention = True 

      return mention_emb, head_scores

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

  def encode(self, wc_sentences, sequence_length, merge_func=tf.concat):
    if not nest.is_sequence(self.encoders):
      return self.encoders.encode(wc_sentences, sequence_length)
    outputs = []
    state = []
    for e in self.encoders:
      word_repls, o, s = e.encode(wc_sentences, sequence_length)
      outputs.append(o)
      state.append(s)
    #outputs = merge_func(outputs, axis=2)
    outputs = merge_func(outputs, axis=-1)
    state = merge_state(state, merge_func=merge_func)
    return word_repls, outputs, state

  def get_mention_emb(self, *args, merge_func=tf.reduce_mean):
    mention_embs = [e.get_mention_emb(*args) for e in self.encoders]
    return merge_func(mention_embs, axis=-1)
