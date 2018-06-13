# coding: utf-8
import math
import tensorflow as tf


class ManagerBase(object):
  def __init__(self, sess, config):
    self.sess = sess
    self.max_gradient_norm = config.max_gradient_norm

    with tf.name_scope('global_variables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.learning_rate = config.learning_rate
      self.decay_rate = config.decay_rate
      self.decay_frequency = config.decay_frequency
      # self.learning_rate = tf.train.exponential_decay(
      #   config.learning_rate, self.global_step,
      #   config.decay_frequency, config.decay_rate, staircase=True)

    # Define operations in advance not to create ops in the loop.
    self._add_epoch = tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32)))

  def get_updates(self, loss, global_step):
    with tf.name_scope("update"):
      learning_rate = tf.train.exponential_decay(
        self.learning_rate, global_step,
        self.decay_frequency, self.decay_rate, staircase=True)

      # TODO: root_scopeの下でroot_scopeを含む変数を呼んでるからスコープが重なる
      params = tf.contrib.framework.get_trainable_variables()
      opt = tf.train.AdamOptimizer(learning_rate)
      gradients = [grad for grad, _ in opt.compute_gradients(loss)]
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
      grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
      updates = opt.apply_gradients(
        grad_and_vars, global_step=global_step)
    return updates

  def add_epoch(self):
    self.sess.run(self._add_epoch)


class ModelBase(object):
  def __init__(self, sess, config):
    self.sess = sess
    self.debug_ops = []
    self.global_step = tf.get_variable(
      "global_step", trainable=False, shape=[],  dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

    self.max_score = tf.get_variable(
      "max_score", trainable=False, shape=[],  dtype=tf.float32,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32)) 

    # Define operations in advance not to create ops in the loop.
    self._add_step = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1, dtype=tf.int32)))
    self._next_score = tf.placeholder(tf.float32, name='max_score_ph', shape=[])
    self._update_max_score = tf.assign(self.max_score, self._next_score)

  def initialize_embeddings(self, name, emb_shape, initializer=None, 
                            trainable=True):
    if not initializer:
      initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embeddings = tf.get_variable(name, emb_shape, trainable=trainable,
                                 initializer=initializer)
    return embeddings

  def get_input_feed(self, batch, is_training):
    return {}


  def add_step(self):
    #self.sess.run(tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1, dtype=tf.int32))))
    self.sess.run(self._add_step)

  def update_max_score(self, score):
    #self.sess.run(tf.assign(self.max_score, tf.constant(score, dtype=tf.float32)))
    
    self.sess.run(self._update_max_score, feed_dict={self._next_score:score})
