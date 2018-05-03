# coding: utf-8 
import math, time, sys
import tensorflow as tf
from orderedset import OrderedSet
from core.utils import common, evaluation, tf_utils
from core.models.base import ModelBase, ManagerBase
#import core.models.graph as graph
#import core.models.wikiP2D.encoder as encoder
from core.models.wikiP2D.encoder import SentenceEncoder, WordEncoder, MultiEncoderWrapper
from core.models.wikiP2D.gen_desc.gen_desc import DescriptionGeneration
from core.models.wikiP2D.graph.graph import GraphLinkPrediction
from core.models.wikiP2D.coref.coref import CoreferenceResolution
from core.models.wikiP2D.adversarial import AdversarialLearning
import numpy as np
from pprint import pprint


def get_multi_encoder(scope, shared_encoder, word_encoder, is_training):
  encoders = [shared_encoder]
  if config.task_private_encoder:
    with tf.variable_scope(scope):
      private_encoder = SentenceEncoder(config, is_training,
                                        word_encoder,
                                        shared_scope=None)
    encoders.append(private_encoder)
  return MultiEncoderWrapper(encoders)

##############################
##      MTL Manager
##############################

class MTLManager(ManagerBase):
  def __init__(self, sess, config, vocab,
               genre_vocab, 
               activation=tf.nn.relu):
    super(MTLManager, self).__init__(sess, config)
    self.activation = activation
    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
    self.debug = config.debug
    w_vocab = vocab.word if 'word' in vocab else None # for encoder
    c_vocab = vocab.char if 'char' in vocab else None # for encoder
    o_vocab = vocab.obj if 'obj' in vocab else None # for graph
    r_vocab = vocab.rel if 'rel' in vocab else None # for graph
    genre_vocab = vocab.genre if 'genre' in vocab else None # for coref

    if config.task_private_encoder:
      config.rnn_size = config.rnn_size / 2

    # with tf >= 1.2, the scope where a RNNCell is called first is cached and the variables are automatically reused.
    with tf.variable_scope("WordEncoder") as scope:
      self.word_encoder = WordEncoder(config, self.is_training, w_vocab, c_vocab,
                                      shared_scope=scope)
    with tf.variable_scope("GlobalEncoder") as scope:
      self.shared_sent_encoder = SentenceEncoder(config, self.is_training,
                                                 self.word_encoder,
                                                 shared_scope=scope)
    ## Define each task
    self.tasks = []
    self.coref, self.graph, self.desc = None, None, None

    if config.coref_task:
      with tf.variable_scope("Coreference") as scope:
        print("Building Coref Model...")
        device = '/gpu:0' 
        with tf.device(device):
          encoder = self.get_sent_encoder(config, scope)
          self.coref = CoreferenceResolution(sess, config.coref, self.is_training,
                                             encoder, genre_vocab, 
                                             activation=self.activation)
        self.tasks.append(self.coref)

    if config.graph_task:
      with tf.variable_scope("Graph") as scope:
        print("Building Graph Model...")
        encoder = self.get_sent_encoder(config, scope)
        device = '/gpu:1' if config.coref_task and len(tf_utils.get_available_gpus()) > 1 else '/gpu:0'

        with tf.device(device):
          self.graph = GraphLinkPrediction(sess, config.wikiP2D, self.is_training,
                                           encoder, o_vocab, r_vocab,
                                           activation=self.activation)
        self.tasks.append(self.graph)

    if config.desc_task:
      with tf.variable_scope("Description") as scope:
        encoder = self.get_sent_encoder(config, scope)
        with tf.device(device):
          self.desc = DescriptionGeneration(config.wikiP2D, self.is_training,
                                            encoder, w_vocab,
                                            activation=self.activation)
        self.tasks.append(self.desc)

    
    if config.adv_task:
      with tf.variable_scope("Adversarial") as scope:
        device = '/gpu:1' if config.coref_task and len(tf_utils.get_available_gpus()) > 1 else '/gpu:0'
        with tf.device(device):
          self.adv = AdversarialLearning(sess, config.adversarial, 
                                         self.shared_sent_encoder, self.tasks)

    self.losses = [t.loss for t in self.tasks]
    weighted_loss = [t.loss * t.loss_weight for t in self.tasks]
    if config.adv_task:
      self.losses.append(self.adv.loss)
      weighted_loss.append(self.adv.loss * self.adv.loss_weight)
    _, self.updates = self.get_loss_and_updates(weighted_loss)

    ## About outputs
    self.output_feed = {
      'train' : self.losses + [self.updates], 
      'valid': self.losses,
      'test' : [
        [t.outputs for t in self.tasks],
      ]
    }
  def get_sent_encoder(self, config, scope):
    if config.use_local_rnn:
      encoder = get_multi_encoder(scope, self.shared_sent_encoder, 
                                  self.word_encoder, self.is_training)
    else:
      encoder = self.shared_sent_encoder

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    for t in self.tasks:
      input_feed.update(t.get_input_feed(batch[t.dataset], is_training))
    return input_feed

  def train(self, batches, summary_writer=None):
    return self.step(batches, summary_writer=summary_writer)

  def valid(self, batches, summary_writer=None):
    return self.step(batches, summary_writer=summary_writer)

  def step(self, batches, summary_writer=None):
    start_time = time.time()
    n_losses = len(self.losses)
    loss = np.array([0.0] * n_losses)
    is_training = batches['is_training']
    output_feed = self.output_feed['train'] if is_training else self.output_feed['valid']

    # Pass a batch of each dataset that is necessary for each task.
    dataset_names = list(OrderedSet([t.dataset for t in self.tasks if hasattr(t, 'dataset')]))
    datasets = OrderedSet([batches[d] for d in dataset_names]) 
    for i, data in enumerate(zip(*datasets)):
      raw_batch = {t.dataset:data[dataset_names.index(t.dataset)] 
                   for t in self.tasks}
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
        raise ValueError("Nan loss ...")
      if False and self.debug and i == 31:
        break

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)

    assert len(self.tasks)+1 == len(loss)

    input_feed = {t.summary_loss:l for t, l in zip(self.tasks, loss[1:])}
    summary_ops = tf.summary.merge([tf.summary.scalar(t.name + '_loss', t.summary_loss) for t in self.tasks])
    summary_dict = {'%s_loss' % (t.name): l for t,l in zip(self.tasks, loss)}
    summary = tf_utils.make_summary(summary_dict)
    loss = " ".join(["%.3f" % l for l in loss])
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



def MultiGPUTrainWrapper(objects):
  def __init__(self, sess, config, is_training,
               w_vocab, c_vocab, o_vocab, r_vocab,
               summary_path=None):
    pass
  def train_or_valid(self):
    pass


