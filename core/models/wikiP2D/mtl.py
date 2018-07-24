# coding: utf-8 
import math, time, sys
from collections import defaultdict, OrderedDict
from orderedset import OrderedSet
from pprint import pprint
import numpy as np
import tensorflow as tf

from core.utils import evaluation, tf_utils
from core.utils.common import dbgprint, dotDict
from core.models.base import ModelBase, ManagerBase
#import core.models.graph as graph
#import core.models.wikiP2D.encoder as encoder
from core.models.wikiP2D.encoder import SentenceEncoder, WordEncoder, MultiEncoderWrapper
from core.models.wikiP2D.gen_desc.gen_desc import DescriptionGeneration
from core.models.wikiP2D.category.category import CategoryClassification
from core.models.wikiP2D.graph.graph import GraphLinkPrediction
from core.models.wikiP2D.coref.coref import CoreferenceResolution
from core.models.wikiP2D.adversarial import TaskAdversarial

available_models = [
  CategoryClassification,
  DescriptionGeneration,
  GraphLinkPrediction,
  CoreferenceResolution,
  TaskAdversarial,
]
available_models = dotDict({c.__name__:c for c in available_models})


def get_multi_encoder(config, shared_sent_encoder, word_encoder, 
                      is_training, scope):
  with tf.variable_scope(scope):
    private_sent_encoder = SentenceEncoder(config, is_training, word_encoder,
                                           shared_scope=scope)
    encoders = [shared_sent_encoder, private_sent_encoder]
  return MultiEncoderWrapper(encoders)


##############################
##      MTL Manager
##############################

class MTLManager(ManagerBase):
  def __init__(self, sess, config, vocab, activation=tf.nn.relu):
    super(MTLManager, self).__init__(sess, config)
    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 

    # with tf >= 1.2, the scope where a RNNCell is called first is cached and the variables are automatically reused.
    with tf.variable_scope("WordEncoder") as scope:
      self.word_encoder = WordEncoder(config.encoder, self.is_training, vocab,
                                      shared_scope=scope)
    with tf.variable_scope("GlobalEncoder") as scope:
      self.shared_sent_encoder = SentenceEncoder(config.encoder, self.is_training,
                                                 self.word_encoder,
                                                 shared_scope=scope)
    ## Define each task
    self.tasks = dotDict()#OrderedDict()
    for i, (task_name, task_config) in enumerate(config.tasks.items()):
      num_gpus = len(tf_utils.get_available_gpus())
      if num_gpus:
        device = '/gpu:%d' % (i % num_gpus)
        sys.stderr.write('Building %s model to %s...\n' % (task_name, device))
      else:
        device = None
        sys.stderr.write('Building %s model to cpu ...\n' % (task_name))

      with tf.variable_scope(task_name) as encoder_scope:
        #with tf.variable_scope('SentenceEncoder') as encoder_scope:
        encoder = self.get_sent_encoder(
          config.encoder, task_config.use_local_rnn, encoder_scope)
        task = self.define_task(sess, task_config, encoder, device)
      self.tasks[task_name] = task

    self.losses = [t.loss for t in self.tasks.values()]
    self.updates = self.get_updates()

  def define_task(self, sess, task_config, encoder, device=None):
    task_class = available_models[task_config.model_type]
    args = [sess, task_config, encoder]

    if issubclass(task_class, TaskAdversarial):
      args.append([t for t in self.tasks.values()
                   if not isinstance(t, TaskAdversarial) ])
      # To use adversarial MTL, training must be done simulteneously for now.
      assert isinstance(self, MeanLoss)

    if device:
      with tf.device(device):
        task = task_class(*args)
    else:
      task = task_class(*args)
    return task

  def get_sent_encoder(self, config, use_local_rnn, scope):
    if use_local_rnn:
      encoder = get_multi_encoder(config, self.shared_sent_encoder, 
                                  self.word_encoder, self.is_training, scope)
    else:
      encoder = self.shared_sent_encoder
    return encoder

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    for task_name, task_model in self.tasks.items():
      if task_name in batch:
        input_feed.update(task_model.get_input_feed(batch[task_name], is_training))
    return input_feed

  def train(self, *args):
    return self.run_epoch(*args, True)

  def valid(self, *args):
    return self.run_epoch(*args, False)

  def test(self, batches):
    raise NotImplementedError("Directly call MTLManager.tasks[task_name].test()")

  def run_epoch(self, batches, is_training):
    raise NotImplementedError

  def get_updates(self):
    raise NotImplementedError("Define in each subclass how to combine losses")

  def get_updates_by_task(self):
    updates = dotDict()
    reuse = False

    for task_name, task_model in self.tasks.items():
      with tf.variable_scope(task_name):
        updates[task_name] = super(MTLManager, self).get_updates(
          task_model.loss, task_model.global_step) 
      reuse = True
    return updates


class BatchIterative(MTLManager):
  def get_updates(self):
    return self.get_updates_by_task()

  def run_epoch(self, batches, is_training):
    start_time = time.time()
    num_steps_in_epoch = [0 for _ in self.tasks]
    loss = [0.0 for _ in self.tasks]
    is_uncomplete = True
    while is_uncomplete:
      is_uncomplete = False
      t = time.time()
      for i, (task_name, task_model) in enumerate(self.tasks.items()):
        try:
          raw_batch = batches[task_name].__next__()
          batch = {task_name:raw_batch}
          input_feed = self.get_input_feed(batch, is_training)
          if task_model.debug_ops:
            print(task_model)
            print(task_model.debug_ops)
            for ops, res in zip(task_model.debug_ops, 
                                self.sess.run(task_model.debug_ops, input_feed)):
              print(ops, res.shape)
              print(res)
            exit(1)
          output_feed = [task_model.loss]
          if is_training:
            output_feed.append(self.updates[task_name])
          t = time.time()
          outputs = self.sess.run(output_feed, input_feed)
          t = time.time() - t
          step_loss = outputs[0]

          print('epoch: %d,' % self.epoch.eval(), 
                'step: %d,' % num_steps_in_epoch[i],
                'task: %s,' % task_name, 
                'step_loss: %.3f,' % step_loss, 
                'step_time: %f,' % t)
          sys.stdout.flush()
          if math.isnan(step_loss):
            raise ValueError(
              "Nan loss detection ... (%s: step %d)" % (task_name, num_steps_in_epoch[i])
            )
          num_steps_in_epoch[i] += 1
          loss[i] += step_loss
          is_uncomplete = True
        except StopIteration as e:
          pass
        except ValueError as e:
          print (e)
          #print('subj.position\n', raw_batch.subj.position)
          #print('obj.position\n', raw_batch.obj.position)
          #print('text.raw\n', raw_batch.text.raw)
          #print('text.word\n', raw_batch.text.word)
          # print('text.char\n', raw_batch.text.char)
          # print('rel.word\n', raw_batch.rel.word)
          # print('rel.char\n', raw_batch.rel.char)
          exit(1)

    epoch_time = (time.time() - start_time)
    loss = [l/num_steps for l, num_steps in zip(loss, num_steps_in_epoch)]
    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_name, mode): l for task_name, l in zip(self.tasks, loss)}
    summary = tf_utils.make_summary(summary_dict)
    return epoch_time, loss, summary


class MeanLoss(MTLManager):
  def get_updates(self):
    loss = tf.reduce_mean([t.loss_weight * t.loss for t in self.tasks.values()])
    updates = super(MTLManager, self).get_updates(loss, self.global_step)
    return updates
 
  def run_epoch(self, batches, is_training):
    start_time = time.time()
    num_steps_in_epoch = [0 for _ in self.tasks]
    loss = np.array([0.0 for _ in self.tasks])

    while True:
      t = time.time()
      batch = {}
      for i, (task_name, task_model) in enumerate(self.tasks.items()):
        try:
          if task_name in batches:
            raw_batch = batches[task_name].__next__()
            batch.update({task_name:raw_batch})
          else:
            batch.update({task_name:{}})
          num_steps_in_epoch[i] += 1
        except StopIteration as e:
          pass
        except ValueError as e:
          print (e)
          exit(1)

      # Once one of the batches of a task stops iteration in an epoch, go to the next epoch.
      if False in [task_name in batch for task_name in self.tasks]:
        break

      input_feed = self.get_input_feed(batch, is_training)
      output_feed = self.losses
      if is_training:
        output_feed.append(self.updates)
      t = time.time()
      outputs = self.sess.run(output_feed, input_feed)
      t = time.time() - t
      step_loss = outputs[:len(self.tasks)]
      loss += np.array(step_loss)
      
      print('epoch: %d,' % self.epoch.eval(), 
            'step: %d,' % self.global_step.eval(),
            'task: %s,' % ' '.join(self.tasks.keys()),
            'step_loss: %s,' % ' '.join(["%.3f" % l for l in step_loss]), 
            'step_time: %f,' % t)
      sys.stdout.flush()

    epoch_time = (time.time() - start_time)
    loss = [l/num_steps for l, num_steps in zip(loss, num_steps_in_epoch)]
    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_name, mode): l for task_name, l in zip(self.tasks, loss)}
    summary = tf_utils.make_summary(summary_dict)
    return epoch_time, loss, summary


class OneByOne(MTLManager):
  def get_updates(self):
    return self.get_updates_by_task()

  def run_epoch(self, *args):
    return self.run_epoch_one_task(*args)

  def get_loss(self, task_name):
    return self.tasks[task_name].loss

  def run_epoch_one_task(self, task_name, batches, is_training):
    task_model = self.tasks[task_name]
    loss = 0.0
    start_time = time.time()
    for i, raw_batch in enumerate(batches[task_name]):
      batch = {task_name:raw_batch}
      input_feed = self.get_input_feed(batch, is_training)
      output_feed = [self.tasks[task_name].loss]
      if is_training:
        output_feed.append(self.updates[task_name])

      t = time.time()
      outputs = self.sess.run(output_feed, input_feed)
      t = time.time() - t
      
      step_loss = outputs[0]
      loss += step_loss

      print('epoch: %d,' % self.epoch.eval(), 
            'step: %d,' % i,
            'task: %s,' % task_name, 
            'step_loss: %.3f,' % step_loss, 
            'step_time: %f,' % t)
      sys.stdout.flush()
      #break # DEBUG
      if math.isnan(step_loss):
        raise ValueError(
          "Nan loss detection ... (%s: step %d)" % (task_name, i))
    loss /= i
    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_name, mode): loss}
    summary = tf_utils.make_summary(summary_dict)
    epoch_time = (time.time() - start_time)
    return epoch_time, loss, summary

class EWC(OneByOne):
  '''
  https://arxiv.org/pdf/1612.00796.pdf
  '''
  def __init__(self, sess, config, vocab, activation=tf.nn.relu):
    super().__init__(sess, config, vocab, activation=activation)

  # def compute_fisher(self,  num_samples=200, ):

  #   # initialize Fisher information for most recent task
  #   self.F_accum = []
  #   for v in range(len(self.var_list)):
  #     self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
  #   # sampling a random class from softmax
  #   probs = tf.nn.softmax(self.y)
  #   class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

  #   for i in range(num_samples):
  #     # select random input image
  #     im_ind = np.random.randint(imgset.shape[0])
  #     # compute first-order derivatives
  #     ders = self.sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
  #     # square the derivatives and add to total
  #     for v in range(len(self.F_accum)):
  #       self.F_accum[v] += np.square(ders[v])

  #     # divide totals by number of samples
  #     for v in range(len(self.F_accum)):
  #       self.F_accum[v] /= num_samples

  def get_updates_by_task(self):
    updates = dotDict()
    reuse = False

    for task_name, task_model in self.tasks.items():
      with tf.variable_scope(task_name):
        updates[task_name] = super().get_updates(task_model.loss, 
                                                 task_model.global_step) 
      reuse = True
    return updates
