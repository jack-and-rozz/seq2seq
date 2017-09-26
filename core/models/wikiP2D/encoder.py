# coding: utf-8 
import tensorflow as tf

from core.utils import common, tf_utils
from core.seq2seq import encoders, rnn
import numpy as np
from core.models.base import ModelBase

class ArticleEncoder(ModelBase):
  def __init__(self, config, w_vocab, c_vocab,
               activation=tf.nn.tanh):
    self.activation = activation
    self.cbase = config.cbase
    self.wbase = config.wbase
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.hidden_size = config.hidden_size

    ## Placeholder
    '''
    articles : [batch_size, n_words, [n_characters]]
    link_spans : [batch_size, 2 (start, end)]
    triples : [None, 2 (relation_id, object_id)]
    '''
    batch_size = None
    self.max_sent_length = config.max_a_sent_length
    self.max_word_length = tf.placeholder(tf.int32, shape=[], 
                                          name='max_word_length')

    ## Embeddings (BOS and EOS are inserted at the start and the end of a sent.)
    if self.wbase:
      self.w_embeddings = self.initialize_embeddings('w_vocab', w_vocab.size, 
                                                     config.hidden_size)
      self.w_articles = tf.placeholder(tf.int32, name='w_articles',
                                       shape=[batch_size, self.max_sent_length+2])
    if self.cbase:
      self.c_embeddings = self.initialize_embeddings('c_vocab', c_vocab.size,
                                                     config.hidden_size)
      self.c_articles = tf.placeholder(tf.int32, name='c_articles',
                                shape=[batch_size, self.max_sent_length+2, None])

    ## Word-based Encoder
    with tf.variable_scope('SentenceEncoder') as scope:
      self.sentence_length = tf.placeholder(
        tf.int32, shape=[batch_size], name="sentence_length")
      self.s_encoder_cell = rnn.setup_cell(config.cell_type, config.hidden_size, 
                                           num_layers=config.num_layers, 
                                           in_keep_prob=config.in_keep_prob, 
                                           out_keep_prob=config.out_keep_prob,
                                           state_is_tuple=config.state_is_tuple)
      self.sent_encoder = getattr(encoders, config.encoder_type)(
        self.s_encoder_cell, scope=scope)

    ## Char-based Encoder
    #self.word_length = tf.placeholder(
    #  tf.int32, shape=[batch_size, self.max_sent_length+2], name="word_length")
    self.word_length = tf.placeholder(
      tf.int32, shape=[batch_size, self.max_sent_length+2], name="word_length")
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
            state_is_tuple=config.state_is_tuple)
          self.word_encoder = word_encoder(
            self.w_encoder_cell, embedding=self.c_embeddings, scope=scope)
        elif word_encoder in [encoders.NoneEncoder]:
          self.word_encoder = word_encoder(
            embedding=self.c_embeddings, scope=scope)


    with tf.name_scope('EncodeArticle'):
      articles = []
      if self.wbase:
        articles.append(self.w_articles)
      if self.cbase:
        articles.append(self.c_articles)
      if not articles:
        raise ValueError('Either FLAGS.wbase or FLAGS.cbase must be True.')
      self.outputs, self.states = self.encode_article(articles)


  def encode_article(self, wc_articles):
    def encode_word(articles):
      if len(articles.get_shape()) == 3: # char-based
        char_repls = tf.nn.embedding_lookup(self.c_embeddings, articles)
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
        word_repls = tf.nn.embedding_lookup(self.w_embeddings, articles)
      return word_repls

    word_repls = tf.concat([encode_word(a) for a in wc_articles], axis=-1)

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


  def get_input_feed(self, batch):
    input_feed = {}
    w_articles = batch['w_articles']
    c_articles = batch['c_articles']
    if self.cbase:
      c_articles, sentence_length, word_length = self.c_vocab.padding(
        c_articles, self.max_sent_length)
      input_feed[self.c_articles] = np.array(c_articles)
      input_feed[self.word_length] = np.array(word_length)
      input_feed[self.max_word_length] = np.array(c_articles).shape[-1]
    if self.wbase:
      w_articles, sentence_length = self.w_vocab.padding(
        w_articles, self.max_sent_length)
      input_feed[self.w_articles] = np.array(w_articles)
      input_feed[self.sentence_length] = np.array(sentence_length)
    return input_feed
 

