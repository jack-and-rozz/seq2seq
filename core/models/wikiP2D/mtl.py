# coding: utf-8 
import math, time, sys
import tensorflow as tf
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
##      Model Classes
##############################

class WikiP2D(ModelBase):
  def __init__(self, sess, config, do_update,
               w_vocab, c_vocab, o_vocab, r_vocab,
               speaker_vocab, genre_vocab,
               activation=tf.nn.tanh, summary_path=None):
    self.initialize(sess, config, do_update)
    self.activation = activation

    self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph) if summary_path else None

    with tf.variable_scope("Encoder") as scope:
      self.word_encoder = WordEncoder(config, w_vocab, c_vocab)
      self.sentence_encoder = SentenceEncoder(config, self.word_encoder,
                                              shared_scope=scope)
      self.encoder = self.sentence_encoder

    ## About subtasks
    self.tasks = []
    self.coref, self.graph, self.desc = None, None, None
    if config.coref_task:
      with tf.variable_scope("Coreference") as scope:
        self.coref = CoreferenceResolution(sess, config, self.encoder, 
                                           speaker_vocab, genre_vocab,)
        self.tasks.append(self.coref)

    if config.graph_task:
      with tf.variable_scope("Graph") as scope:
        self.graph = GraphLinkPrediction(sess, config, self.encoder, 
                                         o_vocab, r_vocab,
                                         activation=self.activation, )
        self.tasks.append(self.graph)

    if config.desc_task:
      with tf.variable_scope("Description") as scope:
        self.desc = DescriptionGeneration(config, self.encoder, w_vocab,
                                          activation=self.activation)
        self.tasks.append(self.desc)
    self.loss, self.updates = self.get_loss_and_updates([t.loss for t in self.tasks], do_update)

    ## About outputs
    self.output_feed = {
      'train' : [self.loss] + [t.loss for t in self.tasks], 
      'test' : [
        [t.outputs for t in self.tasks],
      ]
    }
    if self.updates:
      self.output_feed['train'].append(self.updates)

  def get_input_feed(self, batch):
    input_feed = {}
    input_feed.update(self.encoder.get_input_feed(batch))
    for t in self.tasks:
      input_feed.update(t.get_input_feed(batch[t.name]))
    return input_feed

  def train_or_valid(self, batches):
    start_time = time.time()
    n_losses = len(self.tasks) + 1
    loss = np.array([0.0] * n_losses)
    output_feed = self.output_feed['train']

    for i, (w2p, coref) in enumerate(zip(batches['wikiP2D'], batches['coref'])):
      raw_batch = {'coref': coref, 'desc':w2p, 'graph': w2p}
      input_feed = self.get_input_feed(raw_batch)
      try:
        outputs = self.sess.run(output_feed, input_feed)
      except Exception as e:
        print e
        print (input_feed)
        exit(1)
      step_loss = np.array([l for l in outputs[:n_losses]])
      print self.epoch.eval(), i, step_loss
      loss += step_loss
      if math.isnan(step_loss[0]):
        raise ValueError("Nan loss is detected.")

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)

    if self.summary_writer and False:
      input_feed = {
        self.graph.summary_loss: loss[0]
      }
      summary_ops = tf.summary.merge([
        tf.summary.scalar('loss', self.graph.summary_loss),
      ])
      summary = self.sess.run(summary_ops, input_feed)
      self.summary_writer.add_summary(summary, self.epoch.eval())
    loss = " ".join(["%.3f" % l for l in loss])
    return epoch_time, step_time, loss


  def get_loss_and_updates(self, losses):
    raise NotImplementedError()


class MeanLoss(WikiP2D):
  def get_loss_and_updates(self, losses, do_update):
    loss = tf.reduce_mean(losses)

    updates = self.get_updates(loss) if do_update else None
    return loss, updates

class WeightedLoss(WikiP2D):
  def get_loss_and_updates(self, losses):
    weights = tf.get_variable("loss_weights", [len(losses)])
    loss = tf.reduce_sum(tf.nn.softmax(weights) * losses)
    updates = self.get_updates(loss) if do_update else None

    return loss, updates



def MultiGPUTrainWrapper(objects):
  def __init__(self, sess, config, do_update,
               w_vocab, c_vocab, o_vocab, r_vocab,
               summary_path=None):
    pass
  def train_or_valid(self):
    pass


