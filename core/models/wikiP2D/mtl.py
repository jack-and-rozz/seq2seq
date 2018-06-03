# coding: utf-8 
import math, time, sys
from collections import defaultdict
from orderedset import OrderedSet
from pprint import pprint
import numpy as np
import tensorflow as tf

from core.utils import common, evaluation, tf_utils
from core.models.base import ModelBase, ManagerBase
#import core.models.graph as graph
#import core.models.wikiP2D.encoder as encoder
from core.models.wikiP2D.encoder import SentenceEncoder, WordEncoder, MultiEncoderWrapper
from core.models.wikiP2D.gen_desc.gen_desc import DescriptionGeneration
from core.models.wikiP2D.graph.graph import GraphLinkPrediction
from core.models.wikiP2D.coref.coref import CoreferenceResolution
from core.models.wikiP2D.adversarial import AdversarialLearning

available_models = [
   DescriptionGeneration,
   GraphLinkPrediction,
   CoreferenceResolution,
   AdversarialLearning
]
available_models = common.dotDict({c.__name__:c for c in available_models})


def get_multi_encoder(config, shared_sent_encoder, word_encoder, 
                      is_training, scope):
  with tf.variable_scope(scope):
    private_sent_encoder = SentenceEncoder(config, is_training, word_encoder)
    encoders = [shared_sent_encoder, private_sent_encoder]
  return MultiEncoderWrapper(encoders)

def define_task(sess, task_config, encoder, vocab, device=None):
  task_class = available_models[task_config.model_type]
  #task_class = getattr(core.models.wikiP2D, task_config.model_type)
  if device:
    with tf.device(device):
      task = task_class(sess, task_config, encoder, vocab)
  else:
    task = task_class(sess, task_config, encoder, vocab)
  return task

##############################
##      MTL Manager
##############################

class MTLManager(ManagerBase):
  def __init__(self, sess, config, vocab,
               activation=tf.nn.relu):
    super(MTLManager, self).__init__(sess, config)
    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 

    if config.use_local_rnn:
      config.rnn_size = config.rnn_size / 2

    # with tf >= 1.2, the scope where a RNNCell is called first is cached and the variables are automatically reused.
    with tf.variable_scope("WordEncoder") as scope:
      self.word_encoder = WordEncoder(config, self.is_training, vocab,
                                      shared_scope=scope)
    with tf.variable_scope("GlobalEncoder") as scope:
      self.shared_sent_encoder = SentenceEncoder(config, self.is_training,
                                                 self.word_encoder,
                                                 shared_scope=scope)
    ## Define each task
    self.tasks = common.dotDict()
    for task_name, task_config in config.tasks.items():
      #device = '/gpu:1' if config.coref_task and len(tf_utils.get_available_gpus()) > 1 else '/gpu:0'
      device = None # TODO

      sys.stderr.write('Building %s model...\n' % task_name)
      with tf.variable_scope(task_name) as task_scope:
        encoder = self.get_sent_encoder(config, task_scope)
        task = define_task(sess, task_config, encoder, vocab, device)
      self.tasks[task_name] = task

    self.losses = [t.loss for t in self.tasks.values()]
    _, self.updates = self.get_loss_and_updates(self.losses)


  def get_sent_encoder(self, config, scope):
    if config.use_local_rnn:
      encoder = get_multi_encoder(config, self.shared_sent_encoder, 
                                  self.word_encoder, self.is_training, scope)
    else:
      encoder = self.shared_sent_encoder
    return encoder

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    for t in self.tasks.values():
      input_feed.update(t.get_input_feed(batch[t.dataset], is_training))
    return input_feed

  def train(self, batches, summary_writer=None):
    output_feed = self.losses + [self.updates]
    # DEBUG
    #output_feed = [self.tasks['coref'].w_sentences, 
    #               tf.reduce_sum(self.tasks['coref'].sentence_length)]
    return self.step(batches, output_feed, True, summary_writer=summary_writer)

  def valid(self, batches, summary_writer=None):
    output_feed = self.losses
    return self.step(batches, output_feed, False, summary_writer=summary_writer)

  def test(self, batches, summary_writer=None):
    raise NotImplementedError("Directly call MTLManager.tasks[task_name].test()")

  def step(self, batches, output_feed, is_training, summary_writer=None):
    start_time = time.time()
    n_losses = len(self.losses)
    loss = np.array([0.0] * n_losses)

    # Pass a batch of each dataset that is necessary for each task.
    dataset_names = list(OrderedSet([t.dataset for t in self.tasks.values() if hasattr(t, 'dataset')]))
    datasets = OrderedSet([batches[d] for d in dataset_names]) 
    for i, data in enumerate(zip(*datasets)):
      raw_batch = {t.dataset:data[dataset_names.index(t.dataset)] 
                   for t in self.tasks.values()}
      input_feed = self.get_input_feed(raw_batch, is_training)
      t = time.time()
      # if summary_writer is not None and i % 30 == 0 and self.debug:
      #   run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      #   run_metadata = tf.RunMetadata()
      #   outputs = self.sess.run(output_feed, input_feed,
      #                           options=run_options,
      #                           run_metadata=run_metadata)
      #   summary_writer.add_run_metadata(run_metadata, 'step%d' % (self.global_step.eval()))
      # else:
      outputs = self.sess.run(output_feed, input_feed)
      t = time.time() - t
      step_loss = np.array([l for l in outputs[:n_losses]])
      print(self.epoch.eval(), i, step_loss, 'step_time: %f' % t)
      loss += step_loss

      if math.isnan(step_loss[0]):
        pprint(raw_batch)
        raise ValueError("Nan loss detection ...")

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)
    mode = 'train' if is_training else 'valid'
    input_feed = {t.summary_loss:l for t, l in zip(self.tasks.values(), loss[1:])}
    summary_dict = {'%s/%s/loss' % (mode, task_name): l for task_name, l in zip(self.tasks.keys(), loss)}
    summary = tf_utils.make_summary(summary_dict)
    return epoch_time, step_time, loss, summary

  def get_loss_and_updates(self, losses):
    raise NotImplementedError()


class MeanLoss(MTLManager):
  def get_loss_and_updates(self, losses):
    loss = tf.reduce_mean(losses)
    updates = self.get_updates(loss)
    return loss, updates

class WeightedLoss(MTLManager):
  def get_loss_and_updates(self, losses):
    #weights = tf.get_variable("loss_weights", [len(losses)])
    #loss = tf.reduce_sum(tf.nn.softmax(weights) * losses)
    loss = tf.reduce_mean(losses)
    updates = self.get_updates(loss)
    return loss, updates



