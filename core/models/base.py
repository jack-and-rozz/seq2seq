# coding: utf-8
import tensorflow as tf
from core.models.wikiP2D.encoder import SentenceEncoder, MultiEncoderWrapper #

class ManagerBase(object):
  def __init__(self, sess, config):
    self.sess = sess

    self.optimizer_type = config.optimizer.optimizer_type
    self.learning_rate = config.optimizer.learning_rate
    self.decay_rate = config.optimizer.decay_rate
    self.decay_frequency = config.optimizer.decay_frequency
    self.max_gradient_norm = config.optimizer.max_gradient_norm

    with tf.name_scope('global_variables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

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
      opt = getattr(tf.train, self.optimizer_type)(learning_rate)
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
    self.scope = tf.get_variable_scope()
    self.loss_weight = config.loss_weight if 'loss_weight' in config else 1.0
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

  def get_input_feed(self, batch, is_training):
    return {}


  def add_step(self):
    #self.sess.run(tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1, dtype=tf.int32))))
    self.sess.run(self._add_step)

  def update_max_score(self, score):
    #self.sess.run(tf.assign(self.max_score, tf.constant(score, dtype=tf.float32)))
    
    self.sess.run(self._update_max_score, feed_dict={self._next_score:score})

  def setup_encoder(self, shared_encoder, use_local_rnn, scope=None):
    if use_local_rnn:
      private_encoder = SentenceEncoder(shared_encoder.config, 
                                        shared_encoder.is_training, 
                                        shared_encoder.word_encoder,
                                        shared_scope=scope)
      encoders = [shared_encoder, private_encoder]
      return MultiEncoderWrapper(encoders)
    else:
      return shared_encoder


  def define_combination(self, other_models):
    '''
    This function is to define combined executions across different models. 
    (e.g. run a description decoder to coref dataset)
    <Args>
    - all_models : a list of models.
    '''
    pass
