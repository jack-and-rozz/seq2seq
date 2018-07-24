#coding: utf-8
import math, time, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from core.models.base import ModelBase
from core.utils.tf_utils import linear, shape

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

class TaskAdversarial(ModelBase):
  def __init__(self, sess, config, encoder, tasks):
    super().__init__(sess, config)

    self.sess = sess
    self.encoder = encoder

    encoder_outputs = []
    task_ids = []
    for i, t in enumerate(tasks):
      inputs = []
      if self.encoder.wbase:
        inputs.append(t.text_ph.word)
      if self.encoder.cbase:
        inputs.append(t.text_ph.char)
      #_, encoder_output, _  = self.encoder.encode(inputs, t.sentence_length)

      # t.encoder_outputs: [batch_size, max_sequence_length, hidden_size of shared + private encoder]
      #output_dim = shape(t.encoder_outputs, -1) / 2 
      shared_repls, private_repls = tf.split(t.encoder_outputs, [shape(s, -1) for s in t.encoder.output_shapes], axis=-1)

      # TODO: Is there any other better representations?
      shared_repls = tf.reduce_mean(shared_repls, axis=1)
      task_id = tf.tile([i], [shape(inputs[0], 0)])
      encoder_outputs.append(shared_repls)
      task_ids.append(task_id)
    encoder_outputs = flip_gradient(tf.concat(encoder_outputs, axis=0))
    task_ids = tf.concat(task_ids, axis=0)
    task_ids = tf.one_hot(task_ids, len(tasks))
    self.outputs = tf.nn.softmax(linear(encoder_outputs, len(tasks)))
    l_task = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs,
                                                        labels=task_ids)
    self.loss = l_task
    self.loss = tf.reduce_mean(self.loss)
