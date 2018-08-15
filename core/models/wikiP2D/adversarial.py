#coding: utf-8
import math, time, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from core.models.base import ModelBase
from core.models.wikiP2D.encoder import MultiEncoderWrapper
from core.utils.tf_utils import linear, shape
from core.utils.common import dbgprint

class FlipGradientBuilder(object):
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      #return [tf.neg(grad) * l]
      return [tf.negative(grad) * l]
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)
    self.num_calls += 1
    return y

flip_gradient = FlipGradientBuilder()


def squared_frobenius_norm(tensor):
  return tf.reduce_sum(tf.square(tensor))
  

class TaskAdversarial(ModelBase):
  def __init__(self, sess, config, encoder, tasks):
    super().__init__(sess, config)

    self.sess = sess
    self.encoder = encoder
    adv_outputs = []
    task_ids = []
    for i, t in enumerate(tasks):
      # inputs = []
      # if self.encoder.wbase:
      #   inputs.append(t.text_ph.word)
      # if self.encoder.cbase:
      #   inputs.append(t.text_ph.char)

      if isinstance(t.encoder, MultiEncoderWrapper):
        # Split the encoders' represantions into the task-shared and the task-private.
        assert len(t.adv_outputs.get_shape()) == 3 # [*, max_sentence_length, hidden_size]
        shared_repls, private_repls = tf.split(t.adv_outputs, 2, axis=2)

        # Take average of the representations across all the time step.
        shared_repls = tf.reduce_mean(shared_repls, axis=1)
        private_repls = tf.reduce_mean(private_repls, axis=1)

        # 論文ではこうなっているけど, 違う文を読んだベクトル同士も引き離す必要あるのか？
        #similarities = tf.matmul(tf.transpose(shared_repls), private_repls)
        similarities = tf.matmul(tf.transpose(shared_repls), private_repls)
        l_diff = squared_frobenius_norm(similarities) 

      else:
        shared_repls = t.adv_outputs
        l_diff = 0.0

      task_id = tf.tile([i], [shape(shared_repls, 0)])
      adv_outputs.append(shared_repls)
      task_ids.append(task_id)
    adv_outputs = flip_gradient(tf.concat(adv_outputs, axis=0))
    task_ids = tf.concat(task_ids, axis=0)
    task_ids = tf.one_hot(task_ids, len(tasks))
    self.outputs = tf.nn.softmax(linear(adv_outputs, len(tasks)))
    l_adv = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs,
                                                     labels=task_ids)
    l_adv = tf.reduce_sum(l_adv)
    self.loss = config.adv_weight * l_adv + config.diff_weight * l_diff
