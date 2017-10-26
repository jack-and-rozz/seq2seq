# coding: utf-8 
import math, time, sys
import tensorflow as tf
from core.utils import common, evaluation, tf_utils
from core.models.base import ModelBase
#import core.models.graph as graph
from core.models.wikiP2D.encoder import SentenceEncoder
from core.models.wikiP2D.gen_desc.gen_desc import DescriptionGeneration
from core.models.wikiP2D.graph.graph import GraphLinkPrediction

import numpy as np

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
    self.summary_writer = None

    with tf.variable_scope("Encoder") as scope:
      self.encoder = SentenceEncoder(config, w_vocab, c_vocab,
                                     activation=self.activation)

    ## About subtasks
    self.tasks = []
    self.coref, self.graph, self.desc = None, None, None
    if config.coref_task:
      with tf.variable_scope("Coreference") as scope:
        self.coref = CoreferenceResolution(config, self.encoder, 
                                           speaker_vocab, genre_vocab)
        self.tasks.append(self.coref)

    if config.graph_task:
      with tf.variable_scope("Graph") as scope:
        self.graph = GraphLinkPrediction(config, self.encoder, o_vocab, r_vocab,
                                         activation=self.activation)
        self.tasks.append(self.graph)

    if config.desc_task:
      with tf.variable_scope("Description") as scope:
        self.desc = DescriptionGeneration(config, self.encoder, w_vocab,
                                          activation=self.activation)
        self.tasks.append(self.desc)
    self.loss, self.updates = self.get_loss_and_updates(
      [t.loss for t in self.tasks], do_update)

    if summary_path:
      with tf.name_scope("summary"):
        self.summary_writer = tf.summary.FileWriter(summary_path,
                                                    self.sess.graph)
        if self.graph:
          self.summary_loss = tf.placeholder(tf.float32, shape=[],
                                             name='summary_loss')
          self.summary_mean_rank = tf.placeholder(tf.float32, shape=[],
                                                  name='summary_mean_rank')
          self.summary_mrr = tf.placeholder(tf.float32, shape=[],
                                            name='summary_mrr')
          self.summary_hits_10 = tf.placeholder(tf.float32, shape=[],
                                                name='summary_hits_10')
        if self.desc:
          pass

    ## About outputs
    self.output_feed = {
      'train' : [self.loss] + [t.loss for t in self.tasks] + [self.graph.positives, self.graph.negatives, self.encoder.link_outputs, self.graph.positives],
      'test' : [
        self.graph.positives,
        self.graph.negatives,
      ]
    }
    if self.do_update:
      self.output_feed['train'].append(self.updates)


  def get_input_feed(self, batch):
    input_feed = {}
    input_feed.update(self.encoder.get_input_feed(batch))
    for t in self.tasks:
      input_feed.update(t.get_input_feed(batch))
    return input_feed

  def train_or_valid(self, batches):
    start_time = time.time()
    n_losses = len(self.tasks) + 1
    loss = np.array([0.0] * n_losses)
    output_feed = self.output_feed['train']
    for i, raw_batch in enumerate(batches):
      input_feed = self.get_input_feed(raw_batch)
      outputs = self.sess.run(output_feed, input_feed)
      step_loss = np.array([math.exp(l) for l in outputs[:n_losses]])
      loss += step_loss
      if math.isnan(step_loss[0]):
        raise ValueError("Nan loss is detected.")

    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss /= (i+1)

    if self.summary_writer:
      input_feed = {
        self.summary_loss: loss[0]
      }
      summary_ops = tf.summary.merge([
        tf.summary.scalar('loss', self.summary_loss),
      ])
      summary = self.sess.run(summary_ops, input_feed)
      self.summary_writer.add_summary(summary, self.epoch.eval())
    loss = " ".join(["%.3f" % l for l in loss])
    return epoch_time, step_time, loss

  def test(self, batches):
    output_feed = self.output_feed['test']
    scores = []
    ranks = []
    t = time.time()
    for i, raw_batch in enumerate(batches):
      input_feed = self.get_input_feed(raw_batch)
      outputs = self.sess.run(output_feed, input_feed)
      #loss, positives, negatives = outputs
      positives, negatives = outputs
      _scores, _ranks = self.graph.summarize_results(raw_batch, positives, negatives)

      #sys.stderr.write("%i\t%.3f\n" % (i, time.time() - t))
      scores.append(_scores)
      ranks.append(_ranks)
      t = time.time()
      break

    f_ranks = [x[0] for x in common.flatten(common.flatten(ranks))] # batch-loop, article-loop
    mean_rank = sum(f_ranks) / len(f_ranks)
    mrr = evaluation.mrr(f_ranks)
    hits_10 = evaluation.hits_k(f_ranks)

    if self.summary_writer:
      input_feed = {
        self.summary_mean_rank: mean_rank,
        self.summary_mrr: mrr,
        self.summary_hits_10: hits_10,
      }
      summary_ops = tf.summary.merge([
        tf.summary.scalar('Mean Rank', self.summary_mean_rank),
        tf.summary.scalar('hits@10', self.summary_hits_10),
        tf.summary.scalar('MRR', self.summary_mrr),
      ])
      summary = self.sess.run(summary_ops, input_feed)
      self.summary_writer.add_summary(summary, self.epoch.eval())
    return scores, ranks, mrr, hits_10

  def get_loss_and_updates(self, losses, do_update):
    raise NotImplementedError()


class MeanLoss(WikiP2D):
  def get_loss_and_updates(self, losses, do_update):
    loss = tf.reduce_mean(losses)
    updates = None
    if do_update:
      with tf.name_scope("update"):
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = [grad for grad, _ in opt.compute_gradients(loss)]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                      self.max_gradient_norm)
        grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
        updates = opt.apply_gradients(
          grad_and_vars, global_step=self.global_step)
    return loss, updates

class WeightedLoss(WikiP2D):
  def get_loss_and_updates(self, losses, do_update):
    weights = tf.get_variable("loss_weights", [len(losses)])
    loss = tf.reduce_sum(tf.nn.softmax(weights) * losses)
    updates = None
    if do_update:
      with tf.name_scope("update"):
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = [grad for grad, _ in opt.compute_gradients(loss)]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                      self.max_gradient_norm)
        grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
        updates = opt.apply_gradients(
          grad_and_vars, global_step=self.global_step)
    return loss, updates



def MultiGPUTrainWrapper(objects):
  def __init__(self, sess, config, do_update,
               w_vocab, c_vocab, o_vocab, r_vocab,
               summary_path=None):
    pass
  def train_or_valid(self):
    pass


