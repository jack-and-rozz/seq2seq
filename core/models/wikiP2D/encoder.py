# coding: utf-8 
import tensorflow as tf

from core.utils import common, tf_utils
from core.seq2seq import encoders, rnn
import numpy as np
from core.models.base import ModelBase

class WordEncoder(ModelBase):
  pass

class ParagraphEncoder(ModelBase):
  pass

class SentenceEncoder(ModelBase):
  def __init__(self, config, w_vocab, c_vocab,
               activation=tf.nn.tanh):
    self.activation = activation
    self.cbase = config.cbase
    self.wbase = config.wbase
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.hidden_size = config.hidden_size
    self.max_batch_size = config.batch_size

    # Placeholders
    self.max_sent_length = config.max_a_sent_length
    self.max_word_length = tf.placeholder(tf.int32, shape=[], 
                                          name='max_word_length')

    ## Sentences of each entity are fed as a 2D Tensor (this is because the number of sentences are different depending on the linked entity), so we need to dynamically sort them.
    self.entity_indices = tf.placeholder(tf.int32, shape=[None], 
                                     name='entity_indices')
    self.link_spans = tf.placeholder(tf.int32, shape=[None, 2], 
                                     name='link_spans')

    ## The length of placeholders is increased by 2, because BOS and EOS are inserted at the start and the end of a sent.
    if self.wbase:
      self.w_embeddings = self.initialize_embeddings('w_vocab', w_vocab.size, 
                                                     config.hidden_size)
      self.w_sentences = tf.placeholder(tf.int32, name='w_sentences',
                                       shape=[None, self.max_sent_length+2])
    if self.cbase:
      self.c_embeddings = self.initialize_embeddings('c_vocab', c_vocab.size,
                                                     config.hidden_size)
      self.c_sentences = tf.placeholder(tf.int32, name='c_sentences',
                                shape=[None, self.max_sent_length+2, None])

    ## Word-based Encoder
    with tf.variable_scope('SentenceEncoder') as scope:
      self.sentence_length = tf.placeholder(
        tf.int32, shape=[None], name="sentence_length")
      self.s_encoder_cell = rnn.setup_cell(config.cell_type, config.hidden_size, 
                                           num_layers=config.num_layers, 
                                           in_keep_prob=config.in_keep_prob, 
                                           out_keep_prob=config.out_keep_prob,
                                           state_is_tuple=config.state_is_tuple,
                                           shared=True)
      self.sent_encoder = getattr(encoders, config.encoder_type)(
        self.s_encoder_cell, scope=scope)

    ## Char-based Encoder
    #self.word_length = tf.placeholder(
    #  tf.int32, shape=[None, self.max_sent_length+2], name="word_length")
    self.word_length = tf.placeholder(
      tf.int32, shape=[None, self.max_sent_length+2], name="word_length")
    if self.cbase:
      with tf.variable_scope('WordEncoder') as scope:
        word_encoder = getattr(encoders, config.c_encoder_type)
        if word_encoder in [encoders.RNNEncoder, 
                            encoders.BidirectionalRNNEncoder]:
          self.w_encoder_cell = rnn.setup_cell(
            config.cell_type, config.hidden_size,
            num_layers=config.num_layers, 
            in_keep_prob=config.in_keep_prob, 
            out_keep_prob=config.out_keep_prob,
            state_is_tuple=config.state_is_tuple,
            shared=True)
          self.word_encoder = word_encoder(
            self.w_encoder_cell, embedding=self.c_embeddings, scope=scope)
        elif word_encoder in [encoders.NoneEncoder]:
          self.word_encoder = word_encoder(
            embedding=self.c_embeddings, scope=scope)


    with tf.name_scope('EncodeSentence'):
      sentences = []
      if self.wbase:
        sentences.append(self.w_sentences)
      if self.cbase:
        sentences.append(self.c_sentences)
      if not sentences:
        raise ValueError('Either FLAGS.wbase or FLAGS.cbase must be True.')
      self.outputs, self.states = self.encode_sentence(sentences)
      self.link_outputs = self.extract_span(self.outputs, self.link_spans)

  def encode_sentence(self, wc_sentences):
    def encode_word(sentences):
      if len(sentences.get_shape()) == 3: # char-based
        char_repls = tf.nn.embedding_lookup(self.c_embeddings, sentences)
        word_repls = []
        for i, (c, wl) in enumerate(zip(tf.unstack(char_repls, axis=1), 
                                        tf.unstack(self.word_length, axis=1))):
          reuse = True if i > 0 else None
          with tf.variable_scope("WordEncoder", reuse=reuse) as scope:
            outputs, state = self.word_encoder(c, sequence_length=wl, scope=scope)
            if wl != None:
              mask = tf.sequence_mask(wl, dtype=tf.float32,
                                      maxlen=self.max_word_length)
              outputs = tf.expand_dims(mask, axis=2) * outputs
              outputs = tf.reduce_sum(outputs, axis=1) / (tf.expand_dims(tf.cast(wl, tf.float32), axis=1) + 1e-6)
            else:
              outputs = tf.reduce_mean(outputs, axis=1)

          word_repls.append(outputs)
        word_repls = tf.stack(word_repls, axis=1) 
      else: # word-based
        word_repls = tf.nn.embedding_lookup(self.w_embeddings, sentences)
      return word_repls

    word_repls = tf.concat([encode_word(a) for a in wc_sentences], axis=-1)

    # Linearly transform to adjust the vector size if wbase and cbase are both True
    if word_repls.get_shape()[-1] != self.hidden_size:
      with tf.variable_scope('word_and_chars'):
        word_repls = tf_utils.linear_trans_for_seq(word_repls, self.hidden_size,
                                                   activation=self.activation)

    with tf.variable_scope("SentenceEncoder") as scope:
      outputs, state = self.sent_encoder(word_repls, scope=scope,
                                         sequence_length=self.sentence_length,
                                         merge_type='avg')
    return outputs, state

  # https://stackoverflow.com/questions/44940767/how-to-get-slices-of-different-dimensions-from-tensorflow-tensor
  def extract_span(self, repls, span):
    def loop_func(idx, span_repls, start, end):
      res = tf.reduce_mean(span_repls[idx][start[idx]:end[idx]+1], axis=0)
      return tf.expand_dims(res, axis=0)

    sol, eol = tf.unstack(span, axis=1)
    batch_size = tf.shape(repls)[0]
    idx = tf.zeros((), dtype=tf.int32)

    # Continue concatenating the obtained representation of one span in a row of the batch with the results of previous loop (=res).
    res = tf.zeros((0, self.hidden_size))
    cond = lambda idx, res: idx < batch_size
    body = lambda idx, res: (idx + 1, tf.concat([res, loop_func(idx, repls, sol, eol)], axis=0))
    loop_vars = [idx, res]
    _, res = tf.while_loop(
      cond, body, loop_vars,
      shape_invariants=[idx.get_shape(), 
                        tf.TensorShape([None, self.hidden_size])])
    spans_by_subj = tf.dynamic_partition(res, self.entity_indices, 
                                         self.max_batch_size)

    # Apply max-pooling for spans of an entity.
    spans_by_subj = tf.stack([tf.reduce_max(s, axis=0) for s in spans_by_subj], 
                             axis=0)
    return spans_by_subj

  def get_input_feed(self, batch):
    input_feed = {}
    if len(batch['c_articles']) != len(batch['w_articles']) or len(batch['link_spans']) != len(batch['w_articles']):
      raise ValueError('The length of \'w_articles\', \'c_articles\', and \'link_spans\' must be equal (must have the same number of entity)')

    if self.cbase:
      entity_indices, c_sentences = common.flatten_with_idx(batch['c_articles'])
      c_sentences, sentence_length, word_length = self.c_vocab.padding(
        c_sentences, self.max_sent_length)
      input_feed[self.c_sentences] = np.array(c_sentences)
      input_feed[self.word_length] = np.array(word_length)
      input_feed[self.max_word_length] = np.array(c_sentences).shape[-1]

    if self.wbase:
      entity_indices, w_sentences = common.flatten_with_idx(batch['w_articles'])
      w_sentences, sentence_length = self.w_vocab.padding(
        w_sentences, self.max_sent_length)
      input_feed[self.w_sentences] = np.array(w_sentences)
      input_feed[self.sentence_length] = np.array(sentence_length)

    input_feed[self.entity_indices] = np.array(entity_indices)
    entity_indices, link_spans = common.flatten_with_idx(batch['link_spans'])
    link_spans = [(s+1, e+1) for (s, e) in link_spans] # Shifted back one position by BOS.
    input_feed[self.link_spans] = np.array(link_spans)

    return input_feed
