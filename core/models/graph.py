# coding: utf-8 
import math, time
import tensorflow as tf
import core.utils.tf_utils as tf_utils
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops

class SynsetLinkPrediction(object):
  def __init__(self, sess, FLAGS, do_update, syn_vocab, rel_vocab):
    self.sess = sess
    self.do_update = do_update
    self.syn_vocab = syn_vocab
    self.rel_vocab = rel_vocab
    self.read_flags(FLAGS)

    self.learning_rate = variable_scope.get_variable(
      "learning_rate", trainable=False, shape=[],
      initializer=tf.constant_initializer(float(FLAGS.learning_rate), 
                                          dtype=tf.float32))
    self.global_step = variable_scope.get_variable(
      "global_step", trainable=False, shape=[],  dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

    self.epoch = variable_scope.get_variable(
      "epoch", trainable=False, shape=[], dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 
    #self.ph_subj = tf.placeholder(tf.int32, shape=[None])
    #self.ph_obj = tf.placeholder(tf.int32, shape=[None])
    #self.ph_rel = tf.placeholder(tf.int32, shape=[None])
    self.p_triples = tf.placeholder(tf.int32, shape=[None, 3])
    self.n_triples = tf.placeholder(tf.int32, shape=[None, 3])
    #self.ph_target = tf.placeholder(tf.float32, shape=[None])

    self.e_subj, self.e_obj, self.e_rel = self.initialize_embeddings

    self.loss = tf.reduce_mean(
      tf.maximum(1 - self.inference(self.p_triples) + self.inference(self.n_triples), 0), name='loss'
    )

    params = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = [grad for grad, _ in opt.compute_gradients(self.loss)]
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  self.max_gradient_norm)
    grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
    self.updates = opt.apply_gradients(grad_and_vars, global_step=self.global_step)

  @property
  def initialize_embeddings(self):
    raise NotImplementedError

  @property
  def inference(self):
    raise NotImplementedError

  def read_flags(self, FLAGS):
    self.hidden_size = FLAGS.hidden_size
    self.max_gradient_norm = FLAGS.max_gradient_norm
    self.keep_prob = FLAGS.keep_prob

  def add_epoch(self):
    sess = self.sess
    sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))

  def get_input_feed(self, raw_batch):
    input_feed = {}
    input_feed[self.p_triples] = raw_batch[0]
    input_feed[self.n_triples] = raw_batch[1]
    
    #input_feed[self.ph_subj] = [x[1][0] for x in raw_batch]
    #input_feed[self.ph_rel] = [x[1][1] for x in raw_batch]
    #input_feed[self.ph_obj] = [x[1][2] for x in raw_batch]
    #input_feed[self.ph_target] = [x[0] for x in raw_batch]
    return input_feed

  def step(self, raw_batch):
    input_feed = self.get_input_feed(raw_batch)
    output_feed = [
      self.loss, 
      tf.reduce_mean(self.inference(self.p_triples)), 
      tf.reduce_mean(self.inference(self.n_triples))
    ]
    
    if self.do_update:
      output_feed.append(self.updates)
    outputs = self.sess.run(output_feed, input_feed)
    print outputs[:3]
    return outputs[0] #outputs[0], outputs[1]

  def run_batch(self, data, batch_size, do_shuffle=False):
    start_time = time.time()
    loss = 0.0
    #ns_rate = 1.0 if self.do_update else 0.0
    ns_rate = 1.0
    for i, raw_batch in enumerate(data.get_batch(batch_size,
                                                 do_shuffle=do_shuffle,
                                                 negative_sampling_rate=ns_rate)):
      step_loss = self.step(raw_batch)

      loss += step_loss
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)
    return epoch_time, step_time, loss

class DistMult(SynsetLinkPrediction):
  @property
  def initialize_embeddings(self):
    initializer = init_ops.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    e_syn = variable_scope.get_variable(
      "synsets", [self.syn_vocab.size, self.hidden_size],
      initializer=initializer)
    e_rel = variable_scope.get_variable(
      #"relations", [self.rel_vocab.size, self.hidden_size, self.hidden_size],
      "relations", [self.rel_vocab.size, self.hidden_size],
      initializer=initializer)
    if self.keep_prob < 1.0:
      e_syn = tf.nn.dropout(e_syn, self.keep_prob)
      e_rel = tf.nn.dropout(e_rel, self.keep_prob)

    return e_syn, e_syn, e_rel

  #@property
  def inference(self, triple):
    ph_subj, ph_rel, ph_obj = tf.unstack(triple, axis=1)
    subj = tf.nn.embedding_lookup(self.e_subj, ph_subj)
    obj = tf.nn.embedding_lookup(self.e_obj, ph_obj)
    rel = tf.nn.embedding_lookup(self.e_rel, ph_rel)
    score = tf_utils.batch_dot(rel, subj * obj)
    score = tf.sigmoid(score)

    #score = tf.expand_dims(score, 1)
    #score = tf.concat([1 - score, score], 1)
    return score
