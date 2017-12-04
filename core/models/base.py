# coding: utf-8
import math
import tensorflow as tf

class ModelBase(object):
  def initialize(self, sess, config):
    self.sess = sess
    self.hidden_size = config.hidden_size
    self.max_gradient_norm = config.max_gradient_norm

    with tf.name_scope('global_variables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      # self.learning_rate = tf.get_variable(
      #   "learning_rate", trainable=False, shape=[],
      #   initializer=tf.constant_initializer(float(config.learning_rate), 
      #                                       dtype=tf.float32))
      self.learning_rate = tf.train.exponential_decay(
        config.learning_rate, self.global_step,
        config.decay_frequency, config.decay_rate, staircase=True)


  def add_epoch(self):
    self.sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))

  def initialize_embeddings(self, name, emb_shape, initializer=None, 
                            trainable=True):
    # if embeddings is not None:
    #   initializer = tf.constant_initializer(embeddings)
    #   emb_shape = embeddings.shape
    # else:
    if not initializer:
      initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embeddings = tf.get_variable(name, emb_shape, trainable=trainable,
                                 initializer=initializer)
    return embeddings

  def get_updates(self, loss):
    with tf.name_scope("update"):
      params = tf.contrib.framework.get_trainable_variables()
      opt = tf.train.AdamOptimizer(self.learning_rate)
      gradients = [grad for grad, _ in opt.compute_gradients(loss)]
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
      grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
      updates = opt.apply_gradients(
        grad_and_vars, global_step=self.global_step)
    return updates
