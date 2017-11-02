# coding: utf-8 
import math, time
import tensorflow as tf
from core.utils import common, evaluation, tf_utils
from core.models.base import ModelBase
import core.models.graph as graph
from core.seq2seq import encoders, rnn
import numpy as np
##############################
##    Scoring Functions
##############################

def distmult(subjects, relations, objects):
  with tf.name_scope('DistMult'):
    subjects = tf.expand_dims(subjects, 1)
    score = tf_utils.batch_dot(relations, subjects * objects, n_unk_dims=2)
    score = tf.sigmoid(score)
    return score


##############################
##      Model classes
##############################

class WikiP2D(graph.GraphLinkPrediction):
  def __init__(self, sess, config, do_update,
               w_vocab, c_vocab, o_vocab, r_vocab,
               summary_path=None):
    self.initialize(sess, config, do_update)
    self.cbase = config.cbase
    self.wbase = config.wbase
    #self.ns_rate = config.negative_sampling_rate
    self.scoring_function = distmult
    self.activation = tf.nn.tanh
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.max_batch_size = config.batch_size
    ## Placeholder
    '''
    articles : [batch_size, n_words, [n_characters]]
    link_spans : [batch_size, 2 (start, end)]
    triples : [batch_size,
               n_triples, 
               2 (relation_id, object_id)]
    '''

    batch_size = None
    self.max_sentence_length = max_sentence_length = config.max_sentence_length
    self.max_word_length = max_word_length = config.max_word_length if config.max_word_length else None

    #batch_size, max_sentence_length, max_word_length = None, 10, None

    self.batch_size = tf.placeholder(tf.int32, shape=[], 
                                    name='batch_size')
    self.w_articles = tf.placeholder(tf.int32, name='w_articles',
                                     shape=[batch_size, max_sentence_length+2])
    self.c_articles = tf.placeholder(tf.int32, name='c_articles',
                    shape=[batch_size, max_sentence_length+2, max_word_length])

    self.link_spans = tf.placeholder(tf.int32, shape=[batch_size, 2], 
                                     name='link_spans')
    # self.p_triples = tf.placeholder(tf.int32, shape=[batch_size, None, 2], 
    #                                 name='positive_triples')
    # self.n_triples = tf.placeholder(tf.int32, shape=[batch_size, None, 2],
    #                                 name='negative_triples')
    self.p_triples = tf.placeholder(tf.int32, shape=[None, 2], 
                                    name='positive_triples')
    self.n_triples = tf.placeholder(tf.int32, shape=[None, 2],
                                    name='negative_triples')
    self.p_triples_bidx = tf.placeholder(tf.int32, shape=[None], 
                                    name='positive_triples_indices')
    self.n_triples_bidx = tf.placeholder(tf.int32, shape=[None], 
                                    name='negative_triples_indices')

    ## Embeddings
    self.w_embeddings = self.initialize_embeddings('w_vocab', w_vocab)
    self.c_embeddings = self.initialize_embeddings('c_vocab', c_vocab)
    self.o_embeddings = self.initialize_embeddings('o_vocab', o_vocab)
    self.r_embeddings = self.initialize_embeddings('r_vocab', r_vocab)

    ## Encoder
    with tf.variable_scope('sentence_encoder') as scope:
      self.sentence_length = tf.placeholder(
        tf.int32, shape=[batch_size], name="sentence_length")
      self.s_encoder_cell = rnn.setup_cell(config.cell_type, config.hidden_size, 
                                           num_layers=config.num_layers, 
                                           in_keep_prob=config.in_keep_prob, 
                                           out_keep_prob=config.out_keep_prob,
                                           state_is_tuple=config.state_is_tuple)
      self.sent_encoder = getattr(encoders, config.encoder_type)(
        self.s_encoder_cell, scope=scope)

    if self.cbase:
      with tf.variable_scope('word_encoder') as scope:
        self.word_length = tf.placeholder(
          tf.int32, shape=[batch_size, max_sentence_length+2], name="word_length")
        self.w_encoder_cell = rnn.setup_cell(config.cell_type, config.hidden_size,
                                             num_layers=config.num_layers, 
                                             in_keep_prob=config.in_keep_prob, 
                                             out_keep_prob=config.out_keep_prob,
                                             state_is_tuple=config.state_is_tuple)
        self.word_encoder = getattr(encoders, config.encoder_type)(
          self.w_encoder_cell, scope=scope)


    ## Loss and Update
    with tf.name_scope("loss"):
      with tf.name_scope('encode_article'):
        articles = []
        if self.wbase:
          articles.append(self.w_articles)
        if self.cbase:
          articles.append(self.c_articles)
        if not articles:
          raise ValueError('Either FLAGS.wbase or FLAGS.cbase must be True.')
        sent_repls = self.encode_article(articles)

        with tf.name_scope('extract_span'):
          span_repls = self.extract_span(sent_repls, self.link_spans)

      with tf.name_scope('positives'):
        self.positives = self.inference(span_repls, self.p_triples, self.p_triples_bidx)
      with tf.name_scope('negatives'):
        self.negatives = self.inference(span_repls, self.n_triples, self.n_triples_bidx)
      self.loss = self.cross_entropy(self.positives, self.negatives)
    
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
        self.positives,
        self.negatives,
      ],
      'test' : [
        self.positives,
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

  def encode_article(self, wc_articles):
    def _encode(articles):
      if len(articles.get_shape()) == 3: # char-based
        char_repls = tf.nn.embedding_lookup(self.c_embeddings, articles)
        word_repls = []
        for i, (c, wl) in enumerate(zip(tf.unstack(char_repls, axis=1), 
                                        tf.unstack(self.word_length, axis=1))):
          reuse = True if i > 0 else None
          with tf.variable_scope("word_encoder", reuse=reuse) as scope:
            outputs, state = self.word_encoder(c, sequence_length=wl, scope=scope)
          word_repls.append(outputs)
        word_repls = tf.stack(word_repls, axis=1) 
        word_repls = tf.reduce_mean(word_repls, axis=2)
      else: # word-based
        word_repls = tf.nn.embedding_lookup(self.w_embeddings, articles)
      return word_repls

    word_repls = tf.concat([_encode(a) for a in wc_articles], axis=-1)
    # Linearly transformetsu to adjust the vector size if wbase and cbase are both True.
    if word_repls.get_shape()[-1] != self.hidden_size:
      with tf.variable_scope('word_and_chars'):
        word_repls = tf_utils.linear_trans_for_seq(word_repls, self.hidden_size,
                                                   activation=self.activation)

    with tf.variable_scope("sentence_encoder") as scope:
      outputs, state = self.sent_encoder(word_repls, scope=scope,
                                         sequence_length=self.sentence_length,
                                         merge_type='avg')
    return outputs

  # https://stackoverflow.com/questions/44940767/how-to-get-slices-of-different-dimensions-from-tensorflow-tensor
  def extract_span(self, repls, span):
    def reduce_func(idx, span_repls, start, end):
      res = tf.reduce_mean(span_repls[idx][start[idx]:end[idx]+1], axis=0)
      return tf.expand_dims(res, axis=0)

    sol, eol = tf.unstack(span, axis=1)
    batch_size = tf.shape(repls)[0]
    idx = tf.zeros((), dtype=tf.int32)

    # Continue concatenating the obtained representation of one span in a row of the batch with the results of previous loop (=res).
    res = tf.zeros((0, self.hidden_size))
    cond = lambda idx, res: idx < batch_size
    body = lambda idx, res: (idx + 1, tf.concat([res, reduce_func(idx, repls, sol, eol)], axis=0))
    loop_vars = [idx, res]
    _, res = tf.while_loop(
      cond, body, loop_vars,
      shape_invariants=[idx.get_shape(), 
                        tf.TensorShape([None, self.hidden_size])])
    return res
    #span_repls = extract_span(sent_repls, link_spans)
    #return sent_repls, span_repls

  def inference(self, span_repls, triples):
    
    relations, objects = tf.unstack(triples, axis=2)
    relations = self.activation(tf.nn.embedding_lookup(self.r_embeddings, relations))
    objects = self.activation(tf.nn.embedding_lookup(self.o_embeddings, objects))
    score = self.scoring_function(span_repls, relations, objects)
    score = tf.reshape(score, [-1])
    return score

  def reshape_triples(self, triples):
    # reshape triples from [batch_size, None, 2] to [None, 2] and generate in-batch indices, shape of [None].
    idx = common.flatten([[i for _ in p] for i, p in enumerate(triples)])
    triples = common.flatten(triples)
    return triples, idx

  def get_input_feed(self, raw_batch):
    input_feed = {}
    w_articles, c_articles, link_spans, p_triples, n_triples = raw_batch
    input_feed[self.batch_size] = len(link_spans)
    if c_articles:
      c_articles, sentence_length, word_length = self.c_vocab.padding(
        c_articles, self.max_sentence_length, self.max_word_length)
      input_feed[self.c_articles] = np.array(c_articles)
      input_feed[self.word_length] = np.array(word_length)
    if w_articles:
      w_articles, sentence_length = self.w_vocab.padding(
        w_articles, self.max_sentence_length)
      input_feed[self.w_articles] = np.array(w_articles)
      input_feed[self.sentence_length] = np.array(sentence_length)

    input_feed[self.link_spans] = np.array(link_spans)

    p_triples, p_idx = self.reshape_triples(p_triples)
    input_feed[self.p_triples] = np.array(p_triples)
    input_feed[self.p_triples_bidx] = np.array(p_idx)
    if n_triples:
      n_triples, n_idx = self.reshape_triples(n_triples)
      input_feed[self.n_triples] = np.array(n_triples)
      input_feed[self.n_triples_bidx] = np.array(n_idx)

    # # DEBUG
    #for ca, wa, ls in zip(c_articles, w_articles, link_spans):
    #   print self.c_vocab.ids2tokens(ca, ls)
    #   print self.w_vocab.ids2tokens(wa, ls)
    return input_feed

  
  def train_or_valid(self, data, batch_size, do_shuffle=False):
    start_time = time.time()
    loss = 0.0
    output_feed = self.output_feed['train']
    batches = data.get_batch(batch_size,
                             do_shuffle=do_shuffle,
                             n_pos_triples=None,
                             min_sentence_length=None,
                             max_sentence_length=self.max_sentence_length)
    for i, raw_batch in enumerate(batches):
      input_feed = self.get_input_feed(raw_batch)
      outputs = self.sess.run(output_feed, input_feed)
      step_loss = math.exp(outputs[0])
      loss += step_loss
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)
    loss = loss
    return epoch_time, step_time, loss

  def test(self, data, batch_size):
    output_feed = self.output_feed['test']
    batches = data.get_test_batch(batch_size, max_sentence_length=self.max_sentence_length)

    for i, raw_batch in enumerate(batches):
      input_feed = self.get_input_feed(raw_batch)
      outputs = self.sess.run(output_feed, input_feed)
