# coding: utf-8 
import math, time
import tensorflow as tf
from core.utils import common, evaluation
from core.models.base import ModelBase
import core.models.graph as graph
from core.seq2seq import encoders, rnn

class WikiP2D(graph.GraphLinkPrediction):
  def __init__(self, sess, config, do_update,
               vocab, o_vocab, r_vocab,
               summary_path=None):
    self.initialize(sess, config, do_update)
    self.cbase = config.cbase
    self.ns_rate = config.negative_sampling_rate
    #self.read_config(config)

    ## Placeholder
    '''
    articles : [batch_size, n_words, [n_characters]]
    link_spans : [batch_size, 2 (start, end)]
    triples : [batch_size,
               n_triples, 
               2 (relation_id, object_id)]
    '''

    batch_size, max_sentence_length, max_word_length = None, config.max_sentence_length, config.max_word_length
    #batch_size, max_sentence_length, max_word_length = None, 10, None
    
    sentence_shape = [batch_size, max_sentence_length, max_word_length] if self.cbase else [batch_size, max_sentence_length]
    self.articles = tf.placeholder(tf.int32, shape=sentence_shape, name='articles')
    self.link_spans = tf.placeholder(tf.int32, shape=[batch_size, 2], 
                                     name='link_spans')
    self.p_triples = tf.placeholder(tf.int32, shape=[batch_size, None, 2], 
                                    name='positive_triples')
    self.n_triples = tf.placeholder(tf.int32, shape=[batch_size, None, 2],
                                    name='negative_triples')

    ## Embeddings
    self.embeddings = self.initialize_embeddings('vocab', vocab)
    self.o_embeddings = self.initialize_embeddings('o_vocab', o_vocab)
    self.r_embeddings = self.initialize_embeddings('r_vocab', r_vocab)

    ## Encoder
    with tf.variable_scope('sentence_encoder') as scope:
      self.sentence_length = tf.placeholder(tf.int32, shape=[batch_size], 
                                            name="sentence_length")
      self.s_encoder_cell = rnn.setup_cell(config.cell_type, config.hidden_size, 
                                           num_layers=config.num_layers, 
                                           in_keep_prob=config.in_keep_prob, 
                                           out_keep_prob=config.out_keep_prob,
                                           state_is_tuple=config.state_is_tuple)
      self.sent_encoder = getattr(encoders, config.encoder_type)(
        self.s_encoder_cell, self.embeddings, scope=scope)

    if self.cbase:
      with tf.variable_scope('word_encoder') as scope:
        self.word_length = tf.placeholder(tf.int32, 
                                          shape=[batch_size, max_sentence_length], 
                                          name="word_length")
        self.w_encoder_cell = rnn.setup_cell(config.cell_type, config.hidden_size,
                                             num_layers=config.num_layers, 
                                             in_keep_prob=config.in_keep_prob, 
                                             out_keep_prob=config.out_keep_prob,
                                             state_is_tuple=config.state_is_tuple)
        self.word_encoder = getattr(encoders, config.encoder_type)(
          self.w_encoder_cell, self.embeddings, scope=scope)


    ## Loss and Update
    with tf.name_scope("loss"):
      with tf.name_scope('encode_article'):
        sent_repls, span_repls = self.encode_article(self.articles, 
                                                     self.link_spans)
      return

      with tf.name_scope('positives'):
        positives = self.inference(span_repls, self.p_triples)
      with tf.name_scope('negatives', reuse=True):
        negatives = self.inference(span_repls, self.n_triples)
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

  def encode_article(self, articles, link_spans):
    if self.cbase:
      char_repls = tf.nn.embedding_lookup(self.embeddings, articles)
      word_repls = []
      for i, (c, wl) in enumerate(zip(tf.unstack(char_repls, axis=1), 
                                    tf.unstack(self.word_length, axis=1))):
        reuse = True if i > 0 else None
        with tf.variable_scope("word_encoder", reuse=reuse) as scope:
          #print c, wl
          do_merge = False
          outputs, state = self.word_encoder(c, sequence_length=wl, 
                                             scope=scope, merge_type='avg')
        word_repls.append(outputs)
      word_repls = tf.stack(word_repls, axis=1) 
      word_repls = tf.reduce_mean(word_repls, axis=2)
    else:
      word_repls = tf.nn.embedding_lookup(self.embeddings, articles)

    #print 'words_repl', word_repls
    with tf.variable_scope("sentence_encoder") as scope:
      outputs, state = self.sent_encoder(word_repls, scope=scope,
                                         sequence_length=self.sentence_length,
                                         merge_type='avg')
    sent_repls = outputs

    # https://stackoverflow.com/questions/44940767/how-to-get-slices-of-different-dimensions-from-tensorflow-tensor
    def extract_span(repls, span):
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
    span_repls = extract_span(sent_repls, link_spans)
    return sent_repls, span_repls

  def inference(self, span_repls, triples):
    print span_repls
    print triples 
    relations, objects = tf.unstack(triples, axis=2)
    relations = tf.nn.embedding_lookup(self.r_embeddings, relations)
    objects = tf.nn.embedding_lookup(self.o_embeddings, objects)
    
    print relations, objects
    print relations + objects
    exit(1)
    pass

  def get_input_feed(self, raw_batch):
    input_feed = {}
    sentences, link_spans, p_triples, n_triples = raw_batch
    input_feed[self.articles] = sentences
    input_feed[self.link_spans] = link_spans
    input_feed[self.p_triples] = p_triples
    if n_triples:
      input_feed[self.n_triples] = n_triples
    return input_feed

  def train_or_valid(self, data, batch_size, do_shuffle=False):
    start_time = time.time()
    loss = 0.0
    batches = data.get_batch(batch_size,
                             do_shuffle=do_shuffle,
                             negative_sampling_rate=self.ns_rate)
    for i, raw_batch in enumerate(batches):
      input_feed = self.get_input_feed(raw_batch)
      print input_feed
      exit(1)
      outputs = self.sess.run(output_feed['train'], input_feed)
      step_loss = outputs[0]
      loss += step_loss
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)
