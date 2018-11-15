# coding: utf-8 
import math, time, sys
from collections import defaultdict, OrderedDict
from orderedset import OrderedSet
from pprint import pprint
import numpy as np
import tensorflow as tf

from core.utils.tf_utils import get_available_gpus, make_summary, sum_gradients
from core.utils.common import dbgprint, dotDict, recDotDefaultDict
from core.models.base import ModelBase, ManagerBase

from core.models.wikiP2D.encoder import SentenceEncoder, WordEncoder, MultiEncoderWrapper
from core.models.wikiP2D.desc.desc import DescriptionGeneration
# from core.models.wikiP2D.category.category import CategoryClassification
# from core.models.wikiP2D.graph.graph import GraphLinkPrediction
# from core.models.wikiP2D.relex.relex_base import RelationExtraction
from core.models.wikiP2D.coref.coref import CoreferenceResolution
from core.models.wikiP2D.adversarial import TaskAdversarial

available_models = [
  DescriptionGeneration,
  # CategoryClassification,
  # GraphLinkPrediction,
  # RelationExtraction,
  CoreferenceResolution,
  TaskAdversarial,
]
available_models = dotDict({c.__name__:c for c in available_models})

##############################
##      MTL Manager
##############################

class MTLManager(ManagerBase):
  def __init__(self, sess, config, vocab, activation=tf.nn.relu):
    super(MTLManager, self).__init__(sess, config)
    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
    self.vocab = vocab
    self.use_local_rnn = config.use_local_rnn

    # Define shared layers (Encoder, Decoder, etc.)
    #self.shared_layers = self.setup_shared_layers(config, vocab)
    self.restore_shared_layers = self.setup_shared_layers(config, vocab)
    
    # Define each task
    self.tasks = self.setup_tasks(sess, config)

    self.losses = [t.loss for t in self.tasks.values()]
    self.updates = self.get_updates()

  def setup_shared_layers(self, config, vocab):
    self.scope = tf.get_variable_scope()
    def restore_shared_layers():
      with tf.variable_scope(self.scope):
        with tf.variable_scope("Shared", reuse=tf.AUTO_REUSE) as scope:
          shared_layers = recDotDefaultDict()
          shared_layers.is_training = self.is_training
          with tf.variable_scope("WordEncoder") as scope:
          # Layers to encode a word to a word embedding, and characters to a representation via CNN..
            word_encoder = WordEncoder(config.encoder, self.is_training, 
                                       vocab.encoder, shared_scope=scope)

          # Layers to encode a sequence of word embeddings and outputs from char CNN.
          with tf.variable_scope("SentEncoder") as scope:
            shared_layers.encoder = SentenceEncoder(
              config.encoder, self.is_training, word_encoder, shared_scope=scope)
      return shared_layers
    return restore_shared_layers

  # def setup_shared_layers(self, config, vocab):
  #   with tf.variable_scope("Shared", reuse=tf.AUTO_REUSE) as scope:
  #     shared_layers = recDotDefaultDict()
  #     shared_layers.scope = scope
  #     shared_layers.is_training = self.is_training
  #     with tf.variable_scope("WordEncoder") as scope:
  #       # Layers to encode a word to a word embedding, and characters to a representation via CNN..
  #       word_encoder = WordEncoder(config.encoder, self.is_training, 
  #                                  vocab.encoder, shared_scope=scope)

  #     # Layers to encode a sequence of word embeddings and outputs from char CNN.
  #     with tf.variable_scope("SentEncoder") as scope:
  #       shared_layers.encoder = SentenceEncoder(
  #         config.encoder, self.is_training, word_encoder, shared_scope=scope)
  #   return shared_layers

  def setup_tasks(self, sess, config):
    num_gpus = len(get_available_gpus())
    tasks = dotDict()
    for i, (task_name, task_config) in enumerate(config.tasks.items()):
      device = '/gpu:%d' % (i % num_gpus) if num_gpus else "/cpu:0"
      sys.stderr.write('Building %s model to %s...\n' % (task_name, device))
      with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE):
        task = self.define_task(sess, task_config, device)
      tasks[task_name] = task

    for i, (task_name, task_model) in enumerate(tasks.items()):
      with tf.variable_scope(task_model.scope, reuse=tf.AUTO_REUSE):
        device = '/gpu:%d' % (i % num_gpus) if num_gpus else "/cpu:0"
        with tf.device(device):
          other_models = [t for t in tasks.values() if task_model != t]
          task_model.define_combination(other_models)
    return tasks

  def define_task(self, sess, task_config, device=None):
    task_class = available_models[task_config.model_type]
    args = [sess, task_config, self]
    if device:
      with tf.device(device):
        task = task_class(*args)
    else:
      task = task_class(*args)
    return task

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    for task_name, task_model in self.tasks.items():
      if task_name in batch:
        input_feed.update(task_model.get_input_feed(batch[task_name], is_training))
    return input_feed

  def train(self, *args, summary_writer=None):
    return self.run_epoch(*args, True, summary_writer=summary_writer)

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

class MeanLoss(MTLManager):
  def get_updates(self):
    loss = tf.reduce_mean([t.loss_weight * t.loss for t in self.tasks.values()])
    updates = super(MTLManager, self).get_updates(loss, self.global_step)
    return updates
 
  def run_epoch(self, batches, is_training, summary_writer=None):
    start_time = time.time()
    loss = np.array([0.0 for _ in self.tasks])
    total_execution_time = 0.0
    forward_execution_time = 0.0
    batch_creation_time = 0.0
    num_steps = 0
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
        except StopIteration as e:
          pass
        except ValueError as e:
          print (e)
          exit(1)

      # Once one of the batches of a task stops iteration in an epoch, go to the next epoch.
      if False in [task_name in batch for task_name in self.tasks]:
        break
      batch_creation_time += time.time() - t
      input_feed = self.get_input_feed(batch, is_training)
      output_feed = []
      output_feed += self.losses
      # t = time.time()
      # outputs = self.sess.run(output_feed, input_feed)
      # forward_execution_time += time.time() - t
      if is_training:
        output_feed.append(self.updates)
      t = time.time()
      if summary_writer and self.global_step.eval() % 500 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        outputs = self.sess.run(output_feed, input_feed,
                                options=run_options,
                                run_metadata=run_metadata)
        summary_writer.add_run_metadata(run_metadata, 
                                        'step%d' % self.global_step.eval())
      else:
        outputs = self.sess.run(output_feed, input_feed)
      execution_time = time.time() - t
      total_execution_time += execution_time
      step_loss = outputs[:len(self.tasks)]
      loss += np.array(step_loss)
      for i, l in enumerate(step_loss):
        task_name = list(self.tasks.keys())[i]
        if math.isnan(l):
          if task_name == 'desc':
            print(batch[task_name].qid)
          raise ValueError("Nan loss has been detected... (%s: step %d)" % (task_name, self.global_step.eval()))

      print('epoch: %d,' % self.epoch.eval(), 
            'step: %d,' %  self.global_step.eval(),
            'task: %s,' % ' '.join(self.tasks.keys()),
            'step_loss: %s,' % ' '.join(["%.3f" % l for l in step_loss]), 
            'step_time: %f,' % execution_time)
      num_steps += 1
      sys.stdout.flush()
    epoch_time = (time.time() - start_time)

    print('batch creation time:', batch_creation_time)
    # print('forward execution time:', forward_execution_time)
    print('execution time:', total_execution_time)
    print('epoch time:', epoch_time)
    print('============================')
    # exit(1)

    loss = [l/num_steps for l in loss]
    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_name, mode): l for task_name, l in zip(self.tasks, loss)}
    summary = make_summary(summary_dict)
    return epoch_time, loss, summary

class GradientSum(MeanLoss):
  def get_updates(self):
    with tf.name_scope("update"):
      losses = [t.loss_weight * t.loss for t in self.tasks.values()]
      learning_rate = tf.train.exponential_decay(
        self.learning_rate, self.global_step,
        self.decay_frequency, self.decay_rate, staircase=True)

      opt = getattr(tf.train, self.optimizer_type)(learning_rate)
      num_gpus = len(get_available_gpus())

      params = tf.contrib.framework.get_trainable_variables()

      # gradients = [t.gradients for t in self.tasks.values()]
      # for p in params:
      #   grads = [grad[p] for grad in gradients 
      #            if grad[p] is not None]
      #   print('=================')
      #   print(p, len(grads))
      #   print(grads)
      # #exit(1)

      gradients = sum_gradients([t.gradients for t in self.tasks.values()])
      gradients = [gradients[p] for p in params]
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
      grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
      updates = opt.apply_gradients(
        grad_and_vars, global_step=self.global_step)
    return updates

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
          "Nan loss Nan loss has been detected... (%s: step %d)" % (task_name, i))

    loss /= i
    mode = 'train' if is_training else 'valid'
    summary_dict = {'%s/%s/loss' % (task_name, mode): loss}
    summary = make_summary(summary_dict)
    epoch_time = (time.time() - start_time)
    return epoch_time, loss, summary

##########################################
##              Legacy
##########################################

# class BatchIterative(MTLManager):
#   def get_updates(self):
#     return self.get_updates_by_task()

#   def run_epoch(self, batches, is_training):
#     start_time = time.time()
#     num_steps_in_epoch = [0 for _ in self.tasks]
#     loss = [0.0 for _ in self.tasks]
#     is_uncomplete = True
#     while is_uncomplete:
#       is_uncomplete = False
#       t = time.time()
#       for i, (task_name, task_model) in enumerate(self.tasks.items()):
#         try:
#           raw_batch = batches[task_name].__next__()
#           batch = {task_name:raw_batch}
#           input_feed = self.get_input_feed(batch, is_training)
#           if task_model.debug_ops:
#             for ops, res in zip(task_model.debug_ops, 
#                                 self.sess.run(task_model.debug_ops, input_feed)):
#               #print(ops, res.shape)
#               print(ops)
#               print(res)
#             #exit(1)
#           output_feed = [task_model.loss]
#           if is_training:
#             output_feed.append(self.updates[task_name])
#           t = time.time()
#           outputs = self.sess.run(output_feed, input_feed)
#           execution_time = time.time() - t
#           step_loss = outputs[0]

#           print('epoch: %d,' % self.epoch.eval(), 
#                 'step: %d,' % num_steps_in_epoch[i],
#                 'task: %s,' % task_name, 
#                 'step_loss: %.3f,' % step_loss, 
#                 'step_time: %f,' % execution_time)
#           sys.stdout.flush()
#           if math.isnan(step_loss):
#             raise ValueError(
#               "Nan loss has been detected... (%s: step %d)" % (task_name, num_steps_in_epoch[i])
#             )
#           num_steps_in_epoch[i] += 1
#           loss[i] += step_loss
#           is_uncomplete = True
#         except StopIteration as e:
#           pass
#         except ValueError as e:
#           print (e)
#           # print('subj.position\n', raw_batch.subj.position)
#           # print('obj.position\n', raw_batch.obj.position)
#           # print('text.raw\n', raw_batch.text.raw)
#           # print('text.word\n', raw_batch.text.word)
#           # print('text.char\n', raw_batch.text.char)
#           # print('rel.word\n', raw_batch.rel.word)
#           # print('rel.char\n', raw_batch.rel.char)
#           exit(1)

#     epoch_time = (time.time() - start_time)
#     loss = [l/num_steps for l, num_steps in zip(loss, num_steps_in_epoch)]
#     mode = 'train' if is_training else 'valid'
#     summary_dict = {'%s/%s/loss' % (task_name, mode): l for task_name, l in zip(self.tasks, loss)}
#     summary = make_summary(summary_dict)
#     return epoch_time, loss, summary


# class SeparatelyUpdate(MeanLoss):
#   # NOTE: Only one of these several update ops is executed and other updates are overwritten. This can cause a failure in training. (https://stackoverflow.com/questions/49953379/tensorflow-multiple-loss-functions-vs-multiple-training-ops)
  
#   def get_updates(self):
#     updates = []
#     num_gpus = len(get_available_gpus())

#     for i, t in enumerate(self.tasks.values()):
#       device = '/gpu:%d' % (i % num_gpus) if num_gpus else "/cpu:0"
#       with tf.device(device):
#         loss = t.loss_weight * t.loss
#         update = super(MTLManager, self).get_updates(loss, self.global_step)
#         updates.append(update)
#     return updates



# class EWC(OneByOne):
#   '''
#   https://arxiv.org/pdf/1612.00796.pdf
#   '''
#   def __init__(self, sess, config, vocab, activation=tf.nn.relu):
#     super().__init__(sess, config, vocab, activation=activation)

#   # def compute_fisher(self,  num_samples=200, ):

#   #   # initialize Fisher information for most recent task
#   #   self.F_accum = []
#   #   for v in range(len(self.var_list)):
#   #     self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
#   #   # sampling a random class from softmax
#   #   probs = tf.nn.softmax(self.y)
#   #   class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

#   #   for i in range(num_samples):
#   #     # select random input image
#   #     im_ind = np.random.randint(imgset.shape[0])
#   #     # compute first-order derivatives
#   #     ders = self.sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
#   #     # square the derivatives and add to total
#   #     for v in range(len(self.F_accum)):
#   #       self.F_accum[v] += np.square(ders[v])

#   #     # divide totals by number of samples
#   #     for v in range(len(self.F_accum)):
#   #       self.F_accum[v] /= num_samples

#   def get_updates_by_task(self):
#     updates = dotDict()
#     reuse = False

#     for task_name, task_model in self.tasks.items():
#       with tf.variable_scope(task_name):
#         updates[task_name] = super().get_updates(task_model.loss, 
#                                                  task_model.global_step) 
#       reuse = True
#     return updates


