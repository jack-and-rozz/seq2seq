# coding: utf-8
import math
import tensorflow as tf

class ModelBase(object):
  def initialize(self, sess, config, do_update):
    self.sess = sess
    self.do_update = do_update

    self.hidden_size = config.hidden_size
    self.max_gradient_norm = config.max_gradient_norm

    self.learning_rate = tf.get_variable(
      "learning_rate", trainable=False, shape=[],
      initializer=tf.constant_initializer(float(config.learning_rate), 
                                          dtype=tf.float32))
    self.global_step = tf.get_variable(
      "global_step", trainable=False, shape=[],  dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

    self.epoch = tf.get_variable(
      "epoch", trainable=False, shape=[], dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

  def add_epoch(self):
    self.sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))

  def initialize_embeddings(self, name, vocab_size, hidden_size, initializer=None):
    if not initializer:
      initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embeddings = tf.get_variable(name, [vocab_size, hidden_size],
                                 initializer=initializer)
    return embeddings
