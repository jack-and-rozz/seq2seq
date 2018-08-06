# coding: utf-8 
import sys
import tensorflow as tf
from functools import reduce
import numpy as np

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMStateTuple

from core.utils.common import dbgprint
from core.utils.tf_utils import shape, cnn, linear, projection, batch_gather, batch_loop 
from core.seq2seq.rnn import setup_cell
from core.models.base import ModelBase
from core.vocabulary.base import VocabularyWithEmbedding


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

  def word_encode(self, inputs):
    if inputs is None:
      return inputs
    with tf.variable_scope(self.shared_scope or "WordEncoder", reuse=self.reuse):
      outputs = tf.nn.embedding_lookup(self.w_embeddings, inputs)
      outputs = tf.nn.dropout(outputs, self.keep_prob)
    return outputs

  def char_encode(self, inputs):
    '''
    Args:
    - inputs: [*, max_sequence_length, max_word_length]
    Return:
    - outputs: [*, max_sequence_length, cnn_output_size]
    '''
    if inputs is None:
      return inputs


    with tf.variable_scope(self.shared_scope or "WordEncoder", reuse=self.reuse):
      # Flatten the input tensor to each word (rank-3 tensor).
      with tf.name_scope('flatten'):
        char_repls = tf.nn.embedding_lookup(self.c_embeddings, inputs) # [*, max_word_len, char_emb_size]
        other_shapes = [shape(char_repls, i) for i in range(len(char_repls.get_shape()[:-2]))]

        flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)
        max_sequence_len = shape(char_repls, -2)
        char_emb_size = shape(char_repls, -1)

        flattened_char_repls = tf.reshape(
          char_repls, 
          [flattened_batch_size, max_sequence_len, char_emb_size])

      cnn_outputs = cnn(flattened_char_repls) # [flattened_batch_size, cnn_output_size]
      outputs = tf.reshape(cnn_outputs, other_shapes + [shape(cnn_outputs, -1)]) # [*, cnn_output_size]
      outputs = tf.nn.dropout(outputs, self.keep_prob)
    return outputs

  # def encode(self, wc_inputs):
  #   # inputs: the list of [None, max_sentence_length] or [None, max_sentence_length, max_word_length]
  #   if not isinstance(wc_inputs, list):
  #     wc_inputs = [wc_inputs]

  #   outputs = []
  #   with tf.variable_scope(self.shared_scope or "WordEncoder", reuse=self.reuse):
  #     for inputs in wc_inputs:
  #       if inputs is None:
  #         continue
  #       if len(inputs.get_shape()) == 3: # char-based
  #         char_repls = tf.nn.embedding_lookup(self.c_embeddings, inputs)
  #         batch_size = shape(char_repls, 0)
  #         max_sentence_length = shape(char_repls, 1)
  #         flattened_char_repls = tf.reshape(char_repls, [batch_size * max_sentence_length, shape(char_repls, 2), shape(char_repls, 3)])
  #         flattened_aggregated_char_repls = cnn(flattened_char_repls)
  #         word_repls = tf.reshape(flattened_aggregated_char_repls, [batch_size, max_sentence_length, shape(flattened_aggregated_char_repls, 1)]) # [num_sentences, max_sentence_length, emb_size]
  #       else: # word-based
  #         word_repls = tf.nn.embedding_lookup(self.w_embeddings, inputs)
  #       outputs.append(word_repls)
  #     outputs = tf.concat(outputs, axis=-1)
  #     if self.shared_scope:
  #       self.reuse = True
  #   return tf.nn.dropout(outputs, self.keep_prob) # [None, max_sentence_length, emb_size]

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

  def encode(self, inputs, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse_encode) as scope:
      if isinstance(inputs, list):
        inputs = [x for x in inputs if x is not None]
        sent_repls = tf.concat(inputs, axis=-1) # [*, max_sequence_len, hidden_size]
      # Flatten the input tensor to a rank-3 tensor.
      input_hidden_size = shape(sent_repls, -1)
      max_sequence_len = shape(sent_repls, -2)
      other_shapes = [shape(sent_repls, i) for i in range(len(sent_repls.get_shape()[:-2]))]
      flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)

      flattened_sent_repls = tf.reshape(
        sent_repls, 
        [flattened_batch_size, max_sequence_len, input_hidden_size]) 
      flattened_sequence_length = tf.reshape(sequence_length, [flattened_batch_size])

      initial_state_fw = self.cell_fw.initial_state(flattened_batch_size) if hasattr(self.cell_fw, 'initial_state') else None
      initial_state_bw = self.cell_fw.initial_state(flattened_batch_size) if hasattr(self.cell_bw, 'initial_state') else None

      outputs, state = rnn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, flattened_sent_repls,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        sequence_length=flattened_sequence_length, dtype=tf.float32, scope=scope)


      with tf.variable_scope("outputs"):
        outputs = tf.concat(outputs, -1)
        outputs = tf.nn.dropout(outputs, self.keep_prob)
      with tf.variable_scope("state"):
        state = merge_state(state)

      if self.shared_scope:
        self.reuse_encode = True 

      # Reshape the flattened output to that of the original tensor.
      outputs = tf.reshape(outputs, other_shapes + [max_sequence_len, shape(outputs, -1)])
      if isinstance(state, LSTMStateTuple):
        new_c = tf.reshape(state.c, other_shapes + [shape(state.c, -1)])
        new_h = tf.reshape(state.h, other_shapes + [shape(state.h, -1)])
        state = LSTMStateTuple(c=new_c, h=new_h)
      else:
        state = tf.reshape(state, other_shapes + [shape(state, -1)])
    return sent_repls, outputs, state

  def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
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

  def get_batched_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    '''
    Extract one mention representation from batched texts. 
    Each text has only one mention corresponding to the span in mention_starts/ends.

    Args:
    - text_emb: [*, num_words, dim(word_emb + char_emb)]
    - text_outputs: [*, num_words, dim(encoder_outputs)]
    - mention_starts, mention_ends: [*] 
    Return:
    - mention_emb: [batch_size, emb]
    - head_scores: [batch_size, num_words, 1]
    '''
    with tf.name_scope('flatten'):
      # Keep the original shapes of the input tensors and flatten them.
      text_emb_size = shape(text_emb, -1)
      text_outputs_size = shape(text_outputs, -1)
      num_words = shape(text_outputs, -2)
      other_shapes = [shape(text_outputs, i) for i in range(len(text_outputs.get_shape()[:-2]))]

      flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)
      text_emb = tf.reshape(
        text_emb, [flattened_batch_size, num_words, text_emb_size])
      text_outputs = tf.reshape(
        text_outputs, 
        [flattened_batch_size, num_words, text_outputs_size])
      mention_starts = tf.reshape(mention_starts, 
                                  [flattened_batch_size])
      mention_ends = tf.reshape(mention_ends, 
                                [flattened_batch_size])
      
    dbgprint(text_emb)
    dbgprint(text_outputs)
    dbgprint(mention_starts)
    dbgprint(mention_ends)
    with tf.variable_scope(self.shared_scope or "SentenceEncoder", 
                           reuse=self.reuse_mention) as scope:
      with tf.variable_scope('get_mention_emb'):
        mention_emb_list = []
        mention_width = 1 + mention_ends - mention_starts # [num_mentions]
        max_mention_width = tf.reduce_max(mention_width)

        if self.use_boundary:
          with tf.name_scope('mention_boundary'):
            mention_start_emb = batch_gather(text_outputs, mention_starts) #[batch_size, emb]
            mention_end_emb = batch_gather(text_outputs, mention_ends) #[batch_size, emb]
            dbgprint(mention_start_emb, mention_end_emb)
            batch_size = shape(mention_start_emb, 0)
            hidden_size = shape(mention_start_emb, -1)

            mention_start_emb = tf.reshape(mention_start_emb, 
                                           [batch_size, hidden_size])
            mention_end_emb = tf.reshape(mention_end_emb, 
                                         [batch_size, hidden_size])
            mention_emb_list.append(mention_start_emb)
            mention_emb_list.append(mention_end_emb)
            dbgprint(mention_start_emb, mention_end_emb)

        if self.model_heads:
          with tf.name_scope('mention_attention'):
            mention_indices = tf.expand_dims(tf.range(max_mention_width), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
            mention_indices = tf.minimum(shape(text_outputs, 1) - 1, mention_indices) # [num_mentions, max_mention_width]

            dbgprint(mention_indices)

            mention_text_emb = batch_gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]

            dbgprint(mention_text_emb)
            head_scores = projection(text_outputs, 1) # [batch_size, num_words, 1]
            mention_head_scores = batch_gather(head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
            dbgprint(head_scores)
            dbgprint(mention_head_scores)
            mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, max_mention_width, dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]
            dbgprint(mention_mask)

            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), dim=1) # [num_mentions, max_mention_width, 1]
            dbgprint(mention_attention)

            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
            dbgprint(mention_head_emb)
            mention_emb_list.append(mention_head_emb)
        dbgprint(mention_emb_list)
        mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]
        dbgprint(mention_emb)
      self.debug_ops = [mention_starts, mention_ends, mention_indices, mention_attention, mention_mask]
      if self.shared_scope:
        self.reuse_mention = True 

      # Reshape the flattened outputs to the expected shape for the original input.
      mention_emb = tf.reshape(mention_emb, 
                               other_shapes + [shape(mention_emb, -1)])
      head_scores = tf.reshape(head_scores, 
                               other_shapes + [shape(head_scores, -2), 
                                               shape(head_scores, -1)])

      return mention_emb, head_scores

class MultiEncoderWrapper(SentenceEncoder):
  def __init__(self, encoders):
    """
    Args 
      encoders: A list of SentenceEncoders. The first encoder is regarded as the shared encoder.
    """
    self.encoders = encoders
    self.is_training = encoders[0].is_training
    self.vocab = encoders[0].vocab
    self.word_encoder = encoders[0].word_encoder
    self.w_embeddings = encoders[0].w_embeddings
    self.c_embeddings = encoders[0].c_embeddings
    self.cbase = encoders[0].cbase
    self.wbase = encoders[0].wbase
    self.shared_scope = encoders[0].shared_scope

  def encode(self, wc_sentences, sequence_length, merge_func=tf.concat):
    if not nest.is_sequence(self.encoders):
      return self.encoders.encode(wc_sentences, sequence_length)
    outputs = []
    state = []
    for e in self.encoders:
      word_repls, o, s = e.encode(wc_sentences, sequence_length)
      outputs.append(o)
      state.append(s)
    self.output_shapes = [o.get_shape() for o in outputs]
    self.state_shapes = [s.get_shape() for s in state]
    outputs = merge_func(outputs, axis=-1)
    state = merge_state(state, merge_func=merge_func)
    return word_repls, outputs, state

  def get_mention_emb(self, *args, merge_func=tf.reduce_mean):
    mention_embs = []
    head_scores = []
    for encoder in self.encoders:
      m, h = encoder.get_mention_emb(*args)
      mention_embs.append(m)
      head_scores.append(h)
    dbgprint(mention_embs)
    dbgprint(head_scores)
    if merge_func == tf.reduce_mean:
      axis = 0 
    elif merge_func == tf.concat:
      axis = -1
    else:
      raise ValueError('merge_func must be tf.reduce_mean or tf.concat')
    return merge_func(mention_embs, axis=axis), merge_func(head_scores, axis=axis)
