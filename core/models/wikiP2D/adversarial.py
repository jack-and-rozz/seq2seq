#coding: utf-8
import math, time, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from core.models.base import ModelBase
from core.models.wikiP2D.encoder import MultiEncoderWrapper
from core.utils.tf_utils import linear, shape, get_available_gpus, compute_gradients, sum_gradients
from core.utils.common import dbgprint
from collections import defaultdict

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
  def __init__(self, sess, config, manager, activation=tf.nn.relu):
    super(TaskAdversarial, self).__init__(sess, config)
    self.is_training = manager.is_training
    self.activation = activation
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.adv_weight = config.adv_weight
    self.diff_weight = config.diff_weight
    self.ffnn_depth = config.ffnn_depth

  def define_combination(self, other_models):
    l_adv = 0
    l_diff = 0
    n_tasks =len(other_models)
    num_gpus = len(get_available_gpus())
    gradients = []
    self.loss = 0

    for task_idx, t in enumerate(other_models):
      assert isinstance(t.encoder, MultiEncoderWrapper)
      assert len(t.adv_inputs.get_shape()) == 3 # [num_examples, max_sequence_length, hidden_size]

      device = '/gpu:%d' % (task_idx % num_gpus) if num_gpus else '/cpu:0'
      with tf.device(device):
      #with tf.name_scope('kari'):
        # Split the encoders' represantions into the task-shared and the task-private.
        shared_repls, private_repls = tf.split(t.adv_inputs, 2, axis=-1)
        # 論文ではこうなっているけど, 違う単語に対するベクトル同士も引き離す必要あるのか？


        similarities = tf.matmul(shared_repls, 
                                 tf.transpose(private_repls, [0, 2, 1]))

        hidden = flip_gradient(shared_repls)
        for depth in range(self.ffnn_depth - 1):
          with tf.variable_scope('hidden%d' % (depth+1)) as scope:
            hidden = tf.nn.dropout(
              linear(hidden, shape(hidden, -1), 
                     activation=self.activation, scope=scope), self.keep_prob)

        with tf.variable_scope('output') as scope:
          logits = linear(hidden, n_tasks, activation=None, scope=scope)
        logits = tf.reshape(logits, [-1, n_tasks])
        task_ids = tf.tile([task_idx], [shape(logits, 0)])
        #task_ids = tf.one_hot(task_ids, n_tasks)

        # l_adv += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #   logits=logits, labels=task_ids)) / n_tasks
        l_adv += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=task_ids, logits=logits)) / n_tasks
        l_diff += squared_frobenius_norm(similarities) / n_tasks

        loss = self.adv_weight * l_adv + self.diff_weight * l_diff
        gradients.append(compute_gradients(loss))
        self.loss += l_adv 
    self.gradients = sum_gradients(gradients)
    
