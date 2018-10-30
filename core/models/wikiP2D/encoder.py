# coding: utf-8 
import sys
import tensorflow as tf
from functools import reduce
import numpy as np

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMStateTuple

from core.utils.common import dbgprint, dotDict
from core.utils.tf_utils import shape, cnn, linear, projection, batch_gather, batch_loop, initialize_embeddings, get_available_gpus
from core.seq2seq.rnn import setup_cell
from core.vocabulary.base import VocabularyWithEmbedding


def get_axis(merge_func):
  if merge_func == tf.reduce_mean:
    axis = 0 
  elif merge_func == tf.concat:
    axis = -1
  else:
    raise ValueError('merge_func must be tf.reduce_mean or tf.concat')
  return axis


def merge_state(state, merge_func=tf.concat):
  axis = get_axis(merge_func)
  if isinstance(state[0], LSTMStateTuple):
    #new_c = merge_func([s.c for s in state], axis=1)
    #new_h = merge_func([s.h for s in state], axis=1)
    new_c = merge_func([s.c for s in state], axis=axis)
    new_h = merge_func([s.h for s in state], axis=axis)
    state = LSTMStateTuple(c=new_c, h=new_h)
  else:
    #state = merge_func(state, 1)
    state = merge_func(state, axis)
  return state


class WordEncoder(object):
  def __init__(self, config, is_training, vocab,
               activation=tf.nn.relu, shared_scope=None):
    self.vocab = vocab
    self.is_training = is_training
    self.activation = activation
    self.shared_scope = shared_scope # to reuse variables

    self.wbase = True if vocab.word.size else False
    self.cbase = True if vocab.char.size else False

    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.lexical_dropout_rate
    self.cnn_filter_widths = config.cnn.filter_widths
    self.cnn_filter_size = config.cnn.filter_size

    sys.stderr.write("Initialize word embeddings with pretrained ones.\n")

    device = '/cpu:0' if len(get_available_gpus()) > 1 else None
    with tf.device(device):
      self.embeddings = dotDict()
      self.embeddings.word = initialize_embeddings(
        'word_emb', 
        vocab.word.embeddings.shape,
        initializer=tf.constant_initializer(vocab.word.embeddings), 
        trainable=vocab.word.trainable)

    if self.cbase:
      c_emb_shape = [vocab.char.size, config.embedding_size.char] 
      with tf.device(device):
        self.embeddings.char = initialize_embeddings(
          'char_emb', c_emb_shape, trainable=True)

  def word_encode(self, inputs):
    if inputs is None:
      return inputs
    with tf.variable_scope(self.shared_scope or "WordEncoder"):
      outputs = tf.nn.embedding_lookup(self.embeddings.word, inputs)
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


    with tf.variable_scope(self.shared_scope or "WordEncoder"):
      # Flatten the input tensor to each word (rank-3 tensor).
      with tf.name_scope('flatten'):
        char_repls = tf.nn.embedding_lookup(self.embeddings.char, inputs) # [*, max_word_len, char_emb_size]
        other_shapes = [shape(char_repls, i) for i in range(len(char_repls.get_shape()[:-2]))]

        flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)
        max_sequence_len = shape(char_repls, -2)
        char_emb_size = shape(char_repls, -1)

        flattened_char_repls = tf.reshape(
          char_repls, 
          [flattened_batch_size, max_sequence_len, char_emb_size])

      cnn_outputs = cnn(
        flattened_char_repls,
        filter_widths=self.cnn_filter_widths,
        filter_size=self.cnn_filter_size,
      ) # [flattened_batch_size, cnn_output_size]
      outputs = tf.reshape(cnn_outputs, other_shapes + [shape(cnn_outputs, -1)]) # [*, cnn_output_size]
      outputs = tf.nn.dropout(outputs, self.keep_prob)
    return outputs

  def get_input_feed(self, batch):
    input_feed = {}
    return input_feed

class SentenceEncoder(object):
  def __init__(self, config, is_training, word_encoder, activation=tf.nn.relu, 
               shared_scope=None):
    self.config = config
    self.is_training = is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.use_boundary = config.use_boundary
    self.model_heads = config.model_heads
    self.merge_func = dotDict({k:getattr(tf, func_name) for k, func_name in config.merge_func.items()})
    
    self.word_encoder = word_encoder
    self.vocab = word_encoder.vocab
    self.wbase = word_encoder.wbase
    self.cbase = word_encoder.cbase
    self.embeddings = word_encoder.embeddings

    self.activation = activation
    self.shared_scope = shared_scope

    # For 'initial_state' of CustomLSTMCell, different scopes are required in these initializations.

    with tf.variable_scope('fw_cell'):
      self.cell_fw = setup_cell(config.cell, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

    with tf.variable_scope('bw_cell'):
      self.cell_bw = setup_cell(config.cell, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

  def encode(self, inputs, sequence_length, prop_gradients=True):
    '''
    e.g. When the shape of inputs is [batch_size, max_seq_len, hidden_size], 
         the shape of sequence_length must be [seq_len].
    '''
    for x in inputs:
      assert len(x.get_shape()) == len(sequence_length.get_shape()) + 2

    with tf.variable_scope(self.shared_scope or "SentenceEncoder") as scope:
      if isinstance(inputs, list):
        inputs = tf.concat([x for x in inputs if x is not None], 
                           axis=-1) # [*, max_sequence_len, hidden_size]

      # Flatten the input tensor to a rank-3 tensor.
      input_hidden_size = shape(inputs, -1)
      max_sequence_len = shape(inputs, -2)
      other_shapes = [shape(inputs, i) for i in range(len(inputs.get_shape()[:-2]))]
      flattened_batch_size = reduce(lambda x,y: x*y, other_shapes)

      flattened_inputs = tf.reshape(
        inputs, 
        [flattened_batch_size, max_sequence_len, input_hidden_size]) 
      flattened_sequence_length = tf.reshape(sequence_length, [flattened_batch_size])

      initial_state_fw = self.cell_fw.initial_state(flattened_batch_size) if hasattr(self.cell_fw, 'initial_state') else None
      initial_state_bw = self.cell_bw.initial_state(flattened_batch_size) if hasattr(self.cell_bw, 'initial_state') else None

      outputs, state = rnn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, flattened_inputs,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        sequence_length=flattened_sequence_length, dtype=tf.float32, scope=scope)

      with tf.variable_scope("outputs"):
        axis = get_axis(self.merge_func.birnn)

        outputs = self.merge_func.birnn(outputs, axis=axis)
        outputs = tf.nn.dropout(outputs, self.keep_prob)

      with tf.variable_scope("state"):
        state = merge_state(state, merge_func=self.merge_func.birnn)


      # Reshape the flattened output to that of the original tensor.
      outputs = tf.reshape(outputs, other_shapes + [max_sequence_len, shape(outputs, -1)])
      if isinstance(state, LSTMStateTuple):
        new_c = tf.reshape(state.c, other_shapes + [shape(state.c, -1)])
        new_h = tf.reshape(state.h, other_shapes + [shape(state.h, -1)])
        state = LSTMStateTuple(c=new_c, h=new_h)
      else:
        state = tf.reshape(state, other_shapes + [shape(state, -1)])

    # Make the encoder not to propagate gradients from inputs, outputs, state used in the latter part of (task-specific) layers.
    if not prop_gradients:
      inputs = tf.stop_gradient(inputs)
      outputs = tf.stop_gradient(outputs)
      if isinstance(state, LSTMStateTuple):
        state = LSTMStateTuple(c=tf.stop_gradient(state.c), 
                               h=tf.stop_gradient(state.h))
      else:
        state = tf.stop_gradient(state)
    return inputs, outputs, state

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

    with tf.variable_scope(self.shared_scope or "SentenceEncoder") as scope:
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

            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), axis=1) # [num_mentions, max_mention_width, 1]
            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)
        mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]

      return mention_emb, head_scores

  def get_batched_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    '''
    Extract one mention representation per sentence. 
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

    with tf.variable_scope(self.shared_scope or "SentenceEncoder") as scope:
      with tf.variable_scope('get_mention_emb'):
        mention_emb_list = []
        mention_width = 1 + mention_ends - mention_starts # [num_mentions]
        max_mention_width = tf.reduce_max(mention_width)

        if self.use_boundary:
          with tf.name_scope('mention_boundary'):
            mention_start_emb = batch_gather(text_outputs, mention_starts) #[batch_size, emb]
            mention_end_emb = batch_gather(text_outputs, mention_ends) #[batch_size, emb]
            batch_size = shape(mention_start_emb, 0)
            hidden_size = shape(mention_start_emb, -1)

            mention_start_emb = tf.reshape(mention_start_emb, 
                                           [batch_size, hidden_size])
            mention_end_emb = tf.reshape(mention_end_emb, 
                                         [batch_size, hidden_size])
            mention_emb_list.append(mention_start_emb)
            mention_emb_list.append(mention_end_emb)

        if self.model_heads:
          with tf.name_scope('mention_attention'):
            mention_indices = tf.expand_dims(tf.range(max_mention_width), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
            mention_indices = tf.minimum(shape(text_outputs, 1) - 1, mention_indices) # [num_mentions, max_mention_width]

            mention_text_emb = batch_gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]

            head_scores = projection(text_outputs, 1) # [batch_size, num_words, 1]
            mention_head_scores = batch_gather(head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
            mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, max_mention_width, dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]

            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), axis=1) # [num_mentions, max_mention_width, 1]

            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
            mention_emb_list.append(mention_head_emb)
        mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]
      self.debug_ops = [mention_starts, mention_ends, mention_indices, mention_attention, mention_mask]

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
    <Args> 
     - encoders: A list of SentenceEncoders. The first encoder is regarded as the shared encoder.
    """
    if len(encoders) > 2:
      raise Exception('Current MultiEncoderWrapper can handle up to two encoders.')

    self.encoders = encoders
    self.is_training = encoders[0].is_training
    self.vocab = encoders[0].vocab
    self.word_encoder = encoders[0].word_encoder
    self.embeddings = encoders[0].embeddings
    self.cbase = encoders[0].cbase
    self.wbase = encoders[0].wbase
    self.shared_scope = encoders[0].shared_scope
    self.merge_func = encoders[0].merge_func

  def encode(self, wc_sentences, sequence_length, prop_gradients=True):
    if not nest.is_sequence(self.encoders):
      return self.encoders.encode(wc_sentences, sequence_length)
    outputs = []
    state = []
    for i, e in enumerate(self.encoders):
      word_repls, o, s = e.encode(wc_sentences, sequence_length)
      if not prop_gradients:
        word_repls = tf.stop_gradient(word_repls)
        if i == 0:
          o = tf.stop_gradient(o)
          s = tf.stop_gradient(s)
      outputs.append(o)
      state.append(s)
    self.shared_outputs = outputs[0]
    self.private_outputs = outputs[1]

    merge_func = self.merge_func.shared_private
    axis = get_axis(merge_func)
    outputs = merge_func(outputs, axis=axis)
    state = merge_state(state, merge_func=merge_func)
    return word_repls, outputs, state

  def get_mention_emb(self, *args):
    mention_embs = []
    head_scores = []
    for encoder in self.encoders:
      m, h = encoder.get_mention_emb(*args)
      mention_embs.append(m)
      head_scores.append(h)
    merge_func = self.merge_func.mentions
    axis = get_axis(merge_func)
    return merge_func(mention_embs, axis=axis), merge_func(head_scores, axis=axis)

  def get_batched_mention_emb(self, *args):
    mention_embs = []
    head_scores = []
    for encoder in self.encoders:
      m, h = encoder.get_batched_mention_emb(*args)
      mention_embs.append(m)
      head_scores.append(h)

    merge_func = self.merge_func.mentions
    axis = get_axis(merge_func)
    return merge_func(mention_embs, axis=axis), merge_func(head_scores, axis=axis)
