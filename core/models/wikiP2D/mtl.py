# coding: utf-8 
import math, time, sys
import tensorflow as tf
from orderedset import OrderedSet
from core.utils import common, evaluation, tf_utils
from core.models.base import ModelBase
#import core.models.graph as graph
from core.models.wikiP2D.encoder import WordEncoder, SentenceEncoder
from core.models.wikiP2D.gen_desc.gen_desc import DescriptionGeneration
from core.models.wikiP2D.graph.graph import GraphLinkPrediction
from core.models.wikiP2D.coref.coref import CoreferenceResolution
import numpy as np
from pprint import pprint

##############################
##      MTL Manager
##############################

class MTLManager(ModelBase):
  def __init__(self, sess, config, mode,
               w_vocab, c_vocab, o_vocab, r_vocab,
               genre_vocab, 
               activation=tf.nn.tanh):
    self.initialize(sess, config)
    self.activation = activation
    self.is_training = tf.placeholder(tf.bool, name='is_training', shape=[]) 
    self.mode = mode
    print self.mode
    self.debug = config.debug

    with tf.variable_scope("Encoder") as scope:
      self.word_encoder = WordEncoder(config, self.is_training, w_vocab, c_vocab)
      self.sentence_encoder = SentenceEncoder(config, self.is_training,
                                              self.word_encoder,
                                              shared_scope=scope)
      self.encoder = self.sentence_encoder

    ## About subtasks
    self.tasks = []
    self.coref, self.graph, self.desc = None, None, None
    if config.coref_task:
      with tf.variable_scope("Coreference") as scope:
        device = '/gpu:0' 
        with tf.device(device):
          self.coref = CoreferenceResolution(sess, config.coref, self.is_training,
                                             self.encoder, genre_vocab, 
                                             activation=self.activation)
        self.tasks.append(self.coref)

    if config.graph_task:
      with tf.variable_scope("Graph") as scope:
        device = '/gpu:1' if config.coref_task and len(tf_utils.get_available_gpus()) > 1 else '/gpu:0'

        with tf.device(device):
          self.graph = GraphLinkPrediction(sess, config.wikiP2D, self.is_training,
                                           self.encoder, 
                                           o_vocab, r_vocab,
                                           activation=self.activation)
        self.tasks.append(self.graph)

    if config.desc_task:
      with tf.variable_scope("Description") as scope:
        with tf.device(device):
          self.desc = DescriptionGeneration(config.wikiP2D, self.is_training,
                                            self.encoder, w_vocab,
                                            activation=self.activation)
        self.tasks.append(self.desc)
    self.loss, self.updates = self.get_loss_and_updates([t.loss * t.loss_weight for t in self.tasks])

    ## About outputs
    self.output_feed = {
      'train' : [self.loss] + [t.loss for t in self.tasks] + [self.updates], 
      'valid': [self.loss] + [t.loss for t in self.tasks],
      'test' : [
        [t.outputs for t in self.tasks],
      ]
    }

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    #input_feed.update(self.encoder.get_input_feed(batch))
    for t in self.tasks:
      input_feed.update(t.get_input_feed(batch[t.name], is_training))
    return input_feed

  def train_or_valid(self, batches, summary_writer=None):
    start_time = time.time()
    n_losses = len(self.tasks) + 1
    loss = np.array([0.0] * n_losses)
    is_training = batches['is_training']
    output_feed = self.output_feed['train'] if is_training else self.output_feed['valid']

    # Pass a batch of each dataset that is necessary for each task.
    dataset_names = list(OrderedSet([t.dataset for t in self.tasks]))
    datasets = OrderedSet([batches[d] for d in dataset_names]) 
    for i, data in enumerate(zip(*datasets)):
      raw_batch = {t.name:data[dataset_names.index(t.dataset)] 
                   for t in self.tasks}
      input_feed = self.get_input_feed(raw_batch, is_training)
      t = time.time()
      if summary_writer is not None and i % 50 == 0 and self.debug:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        outputs = self.sess.run(output_feed, input_feed,
                                options=run_options,
                                run_metadata=run_metadata)
        summary_writer.add_run_metadata(run_metadata, 'step%d' % (self.global_step.eval()))
      else:
        outputs = self.sess.run(output_feed, input_feed)
      t = time.time() - t

      step_loss = np.array([l for l in outputs[:n_losses]])
      print self.epoch.eval(), i, step_loss, 'step_time: %f' % t
      loss += step_loss
      if math.isnan(step_loss[0]):
        raise ValueError("Nan loss is detected.")
      if self.debug and i == 51:
         #exit(1)
         #print loss / (i+1)
        break

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)

    assert len(self.tasks)+1 == len(loss)

    input_feed = {t.summary_loss:l for t, l in zip(self.tasks, loss[1:])}
    summary_ops = tf.summary.merge([tf.summary.scalar(t.name + '_loss', t.summary_loss) for t in self.tasks])
    summary_dict = {'%s_loss' % (t.name): l for t,l in zip(self.tasks, loss[1:])}
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


