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
  return tf.reduce_mean(tf.square(tensor))
  

class TaskAdversarial(ModelBase):
  def __init__(self, sess, config, manager):
    super(TaskAdversarial, self).__init__(sess, config)
    self.adv_weight = config.adv_weight
    self.diff_weight = config.diff_weight

  def define_combination(self, other_models):
    l_adv = 0
    l_diff = 0
    n_tasks =len(other_models)
    for i, t in enumerate(other_models):
      print(t, t.adv_inputs)
      assert isinstance(t.encoder, MultiEncoderWrapper)
      assert len(t.adv_inputs.get_shape()) == 2 # [num_examples, hidden_size]

      # Split the encoders' represantions into the task-shared and the task-private.
      shared_repls, private_repls = tf.split(t.adv_inputs, 2, axis=-1)

      # 論文ではこうなっているけど, 違う文を読んだベクトル同士も引き離す必要あるのか？
      similarities = tf.matmul(tf.transpose(shared_repls), private_repls)

      task_ids = tf.tile([i], [shape(shared_repls, 0)])
      task_ids = tf.one_hot(task_ids, n_tasks)
      adv_inputs = flip_gradient(shared_repls)

      logits = tf.nn.softmax(linear(adv_inputs, n_tasks, activation=None))
      l_adv += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=task_ids))
      l_diff += squared_frobenius_norm(similarities) # ここはflip_gradientなしでok?

    l_adv /= n_tasks
    l_diff /= n_tasks
    self.loss = self.adv_weight * l_adv + self.diff_weight * l_diff
