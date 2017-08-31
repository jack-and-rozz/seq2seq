# coding: utf-8 
import math, time
import tensorflow as tf
from core.utils import common, evaluation
from core.models.base import ModelBase
import core.models.graph as graph
from core.seq2seq import encoders, rnn

class WikiP2D(graph.GraphLinkPrediction):
  def read_config(self, config):
    self.cbase = config.cbase

  def __init__(self, sess, config, do_update,
               vocab, o_vocab, r_vocab,
               summary_path=None):
    self.initialize(sess, config, do_update)

    self.read_config(config)

    ## Placeholder
    '''
    articles : [batch_size, n_words, [n_characters]]
    link_spans : [batch_size, 2 (start, end)]
    triples : [batch_size,
               n_triples, 
               2 (relation_id, object_id)]
    '''
    sentence_shape = [None, None, None] if self.cbase else [None, None]
    self.articles = tf.placeholder(tf.int32, shape=sentence_shape, name='articles')
    self.link_spans = tf.placeholder(tf.int32, shape=[None, 2], name='link_spans')
    self.p_triples = tf.placeholder(tf.int32, shape=[None, None, 2], 
                                    name='positive_triples')
    self.n_triples = tf.placeholder(tf.int32, shape=[None, None, 2],
                                    name='negative_triples')

    ## Embeddings
    self.embeddings = self.initialize_embeddings('vocab', vocab)
    self.o_embeddings = self.initialize_embeddings('o_vocab', o_vocab)
    self.r_embeddings = self.initialize_embeddings('r_vocab', r_vocab)

    ## Encoder
    with tf.variable_scope('sentence_encoder') as scope:
      self.sentence_length = tf.placeholder(tf.int32, shape=[None], 
                                            name="sentence_length")
      cell = rnn.setup_cell(config.cell_type, config.num_layers, 
                            config.in_keep_prob, config.out_keep_prob)
      self.sent_encoder = getattr(encoders, FLAGS.encoder_type)(
        cell, self.embedding, scope=scope, sequence_length=self.sentence_length)

    if self.cbase:
      with tf.variable_scope('word_encoder') as scope:
        self.word_length = tf.placeholder(tf.int32, shape=[None, None], 
                                          name="word_length")
        cell = rnn.setup_cell(config.cell_type, config.num_layers, 
                              config.in_keep_prob, config.out_keep_prob)
        self.word_encoder = encoders.RNNEncoder(
          cell, self.embedding, scope=scope, sequence_length=self.word_length)


    ## Loss and Update
    with tf.name_scope("loss"):
      positives = self.inference(self.articles, self.link_spans, self.p_triples)
      negatives = self.inference(self.articles, self.link_spans, self.n_triples)
      self.loss = self.cross_entropy(positives, negatives)

    if summary_path:
      with tf.name_scope("summary"):
        self.summary_writer = tf.summary.FileWriter(summary_path,
                                                    self.sess.graph)
        self.summary_loss = tf.placeholder(tf.float32, shape=[],
                                           name='summary_loss')
        self.summary_mrr = tf.placeholder(tf.float32, shape=[],
                                          name='summary_mrr')
        self.summary_hits_10 = tf.placeholder(tf.float32, shape=[],
                                              name='summary_hits_10')
    if do_update:
      with tf.name_scope("update"):
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = [grad for grad, _ in opt.compute_gradients(self.loss)]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  self.max_gradient_norm)
        grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
        self.updates = opt.apply_gradients(
          grad_and_vars, global_step=self.global_step)

    ## About outputs
    self.output_feed = {
      'train' : [
        self.loss,
      ],
      'test' : [
        self.loss,
      ]
    }
    if self.do_update:
      self.output_feed['train'].append(self.updates)

  def initialize_embeddings(self, name, vocab, initializer=None):
    if not initializer:
      initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embeddings = tf.get_variable(name, [vocab.size, self.hidden_size],
                                 initializer=initializer)
    return embeddings

  def get_input_feed(self, raw_batch):
    input_feed = {}
    input_feed[self.p_triples] = raw_batch[0]

    # in test, raw_batch = [triples, []] 
    if raw_batch[1]:
      input_feed[self.n_triples] = raw_batch[1]
    return input_feed

  def inference(self, articles, link_spans, triples):
    return
  #def train_or_valid(self, data, batch_size, do_shuffle=False):


