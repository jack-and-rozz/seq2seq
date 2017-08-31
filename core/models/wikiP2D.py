# coding: utf-8 

from core.models.base import ModelBase
import core.models.graph as graph

class WikiP2D(graph.GraphLinkPrediction):
  def __init__(self, sess, config, do_update,
               vocab, node_vocab, edge_vocab,
               summary_path=None):
    self.initialize(sess, config, do_update)

    self.vocab = vocab
    self.node_vocab = node_vocab
    self.edge_vocab = edge_vocab
    self.read_config(config)
    self.initialize_embeddings()

    with tf.name_scope("loss"):
      pass
      #self.loss = self.cross_entropy()

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

  def get_input_feed(self, raw_batch):
    input_feed = {}
    input_feed[self.p_triples] = raw_batch[0]

    # in test, raw_batch = [triples, []] 
    if raw_batch[1]:
      input_feed[self.n_triples] = raw_batch[1]
    return input_feed

  #def train_or_valid(self, data, batch_size, do_shuffle=False):


