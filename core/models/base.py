# coding: utf-8


class ModelBase(object):
  def initialize((self, sess, FLAGS, do_update):
    self.sess = sess
    self.do_update = do_update
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

  def add_epoch(self):
    sess = self.sess
    sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))
