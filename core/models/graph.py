# coding: utf-8 
import math, time
import tensorflow as tf
import core.utils.tf_utils as tf_utils
from core.utils import common

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops

class GraphLinkPrediction(object):
  def __init__(self, sess, config, do_update,
               node_vocab, edge_vocab,
               summary_path=None):
    self.sess = sess
    self.do_update = do_update
    self.node_vocab = node_vocab
    self.edge_vocab = edge_vocab
    self.read_config(config)
    self.p_triples = tf.placeholder(tf.int32, shape=[None, 3], 
                                    name='positive_triples')
    self.n_triples = tf.placeholder(tf.int32, shape=[None, 3],
                                    name='negative_triples')

    self.learning_rate = variable_scope.get_variable(
      "learning_rate", trainable=False, shape=[],
      initializer=tf.constant_initializer(float(config.learning_rate), 
                                          dtype=tf.float32))
    self.global_step = variable_scope.get_variable(
      "global_step", trainable=False, shape=[],  dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

    self.epoch = variable_scope.get_variable(
      "epoch", trainable=False, shape=[], dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 
    self.initialize_embeddings()

    with tf.name_scope("loss"):
      self.loss = self.cross_entropy()

    with tf.name_scope("summary"):
      #We shouldn't use tf.summary.merge_all() when separating a train model and a valid model. (https://stackoverflow.com/questions/37621340/error-with-feed-values-for-placeholders-when-running-the-merged-summary-op)
      self.summary_writer = tf.summary.FileWriter(summary_path,
                                                  self.sess.graph)
      self.summary_loss = tf.placeholder(tf.float32, shape=[],
                                         name='summary_loss')
      self.summary_op = tf.summary.merge([
        tf.summary.scalar('loss', self.summary_loss),
      ])

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

  # loss functions
  def cross_entropy(self):
    # calculate cross-entropy by hand.
    ce1 = -tf.log(self.inference(self.p_triples))
    ce2 = -tf.log(1 - self.inference(self.n_triples))
    return tf.reduce_mean(tf.concat([ce1, ce2], axis=0))

  def margin_loss(self, margin=1.0):
    # margin_loss
    return tf.reduce_mean(tf.maximum(margin - self.inference(self.p_triples) + self.inference(self.n_triples), 0))

  def read_config(self, config):
    self.hidden_size = config.hidden_size
    self.max_gradient_norm = config.max_gradient_norm
    self.keep_prob = config.keep_prob if self.do_update else 1.0
    self.share_embedding = common.str_to_bool(config.share_embedding)
    self.ns_rate = 1.0 #config.negative_sampling_rate

  def add_epoch(self):
    sess = self.sess
    sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))

  def get_input_feed(self, raw_batch):
    input_feed = {}
    input_feed[self.p_triples] = raw_batch[0]
    input_feed[self.n_triples] = raw_batch[1]
    return input_feed

  def step(self, raw_batch):
    input_feed = self.get_input_feed(raw_batch)
    output_feed = [
      self.loss,
      self.inference(self.p_triples),
      self.inference(self.n_triples)
    ]
    if self.do_update:
      output_feed.append(self.updates)
    outputs = self.sess.run(output_feed, input_feed)
    return outputs

  def train_or_valid(self, data, batch_size, do_shuffle=False):
    start_time = time.time()
    loss = 0.0
    batches = data.get_train_batch(batch_size,
                                   do_shuffle=do_shuffle,
                                   negative_sampling_rate=self.ns_rate)
    for i, raw_batch in enumerate(batches):
      outputs = self.step(raw_batch)
      step_loss = outputs[0]
      loss += step_loss
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)

    # summaryはまとめたものをplaceholderに入れる形で
    if self.summary_writer:
      input_feed = {
        self.summary_loss: loss
      }
      summary = self.sess.run(self.summary_op, input_feed)
      self.summary_writer.add_summary(summary, self.epoch.eval())
    return epoch_time, step_time, loss

  def test(self, data, batch_size):
    pass

  def initialize_embeddings(self):
    raise NotImplementedError

  def inference(self, triple):
    raise NotImplementedError


class DistMult(GraphLinkPrediction):
  def initialize_embeddings(self):
    initializer = init_ops.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    if self.share_embedding:
      e_syn = variable_scope.get_variable(
        "synsets", [self.node_vocab.size, self.hidden_size],
        initializer=initializer)
      e_syn = tf.nn.dropout(e_syn, self.keep_prob)
      e_s_syn = e_o_syn = e_syn
    else:
      e_s_syn = variable_scope.get_variable(
        "s_synsets", [self.node_vocab.size, self.hidden_size],
        initializer=initializer)
      e_o_syn = variable_scope.get_variable(
        "o_synsets", [self.node_vocab.size, self.hidden_size],
        initializer=initializer)
      e_s_syn = tf.nn.dropout(e_s_syn, self.keep_prob)
      e_o_syn = tf.nn.dropout(e_o_syn, self.keep_prob)

    e_rel = variable_scope.get_variable(
      "relations", [self.edge_vocab.size, self.hidden_size],
      initializer=initializer)
    e_rel = tf.nn.dropout(e_rel, self.keep_prob)

    self.e_s_syn, self.e_rel, self.e_o_syn = e_s_syn, e_rel, e_o_syn

  def inference(self, triple):
    with tf.name_scope('inference'):
      ph_subj, ph_rel, ph_obj = tf.unstack(triple, axis=1)
      s_syn = tf.tanh(tf.nn.embedding_lookup(self.e_s_syn, ph_subj))
      o_syn = tf.tanh(tf.nn.embedding_lookup(self.e_o_syn, ph_obj))
      rel = tf.sigmoid(tf.nn.embedding_lookup(self.e_rel, ph_rel))
      score = tf_utils.batch_dot(rel, s_syn * o_syn)
      score = tf.sigmoid(score)
    return score



