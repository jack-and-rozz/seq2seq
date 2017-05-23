# coding: utf-8
from __future__ import absolute_import
from __future__ import division

import random, sys, os, math, copy
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell

#from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
#import tensorflow.contrib.legacy_seq2seq as seq2seq
#from seq2seq import RNNEncoder, BidirectionalRNNEncoder, RNNDecoder, BasicSeq2Seq
import seq2seq
from utils import common
from utils.dataset import PAD_ID, GO_ID, EOS_ID, UNK_ID, padding_and_format
dtype=tf.float32

class Baseline(object):
  def __init__(self, FLAGS, max_sequence_length, forward_only, do_update):
    self.max_sequence_length = max_sequence_length
    self.forward_only=forward_only
    self.do_update = do_update
    self.read_flags(FLAGS)
    with tf.name_scope('placeholders'):
      self.setup_placeholders(use_sequence_length=self.use_sequence_length)
    cell = self.setup_cell(do_update)
    with variable_scope.variable_scope("Encoder") as encoder_scope:
      self.encoder_embedding = self.initialize_embedding(FLAGS.source_vocab_size,
                                                         FLAGS.embedding_size)
      encoder = getattr(seq2seq, FLAGS.encoder_type)(
        cell, self.encoder_embedding,
        scope=encoder_scope, 
        sequence_length=self.sequence_length)

    with variable_scope.variable_scope("Decoder") as decoder_scope:
      self.decoder_embedding = self.initialize_embedding(FLAGS.target_vocab_size,
                                                    FLAGS.embedding_size)
      decoder = getattr(seq2seq, FLAGS.decoder_type)(
        copy.deepcopy(cell), self.decoder_embedding, scope=decoder_scope)
    self.seq2seq = getattr(seq2seq, FLAGS.seq2seq_type)(
      encoder, decoder, FLAGS.num_samples, feed_previous=forward_only)
    # The last tokens in decoder_inputs are not to set each length of placeholders to be same.
    self.logits, self.losses, self.e_states, self.d_states = self.seq2seq(
      self.encoder_inputs, self.decoder_inputs[:-1],
      self.targets, self.target_weights[:-1])
    if do_update:
      self.updates = self.setup_updates(self.losses)
    self.saver = tf.train.Saver(tf.global_variables())

  def initialize_embedding(self, vocab_size, embedding_size):
    sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
    initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
    embedding = variable_scope.get_variable(
      "embedding", [vocab_size, embedding_size],
      initializer=initializer)
    return embedding

  def setup_updates(self, loss):
    params = tf.trainable_variables()
    gradients = []
    updates = []
    opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     self.max_gradient_norm)
    updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    return updates

  def setup_placeholders(self, use_sequence_length=True):
    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(self.max_sequence_length):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(self.max_sequence_length + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))
    # Our targets are decoder inputs shifted by one.
    self.targets = [self.decoder_inputs[i + 1]
                    for i in xrange(len(self.decoder_inputs) - 1)]
    self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length") if use_sequence_length else None


  def setup_cell(self, do_update):
    
    cell = getattr(rnn_cell, self.cell_type)(self.hidden_size, reuse=tf.get_variable_scope().reuse) 
    if self.keep_prob < 1.0 and do_update:
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
    if self.num_layers > 1:
      cell = rnn_cell.MultiRNNCell([copy.deepcopy(cell) for _ in xrange(self.num_layers)])
    return cell

  def read_flags(self, FLAGS):
    self.keep_prob = FLAGS.keep_prob
    self.hidden_size = FLAGS.hidden_size
    self.max_gradient_norm = FLAGS.max_gradient_norm
    self.num_samples = FLAGS.num_samples
    self.num_layers = FLAGS.num_layers
    self.max_to_keep = FLAGS.max_to_keep
    self.embedding_size = FLAGS.embedding_size
    self.source_vocab_size = FLAGS.source_vocab_size
    self.target_vocab_size = FLAGS.target_vocab_size
    self.seq2seq_type = FLAGS.seq2seq_type
    self.cell_type = FLAGS.cell_type
    self.use_sequence_length=FLAGS.use_sequence_length
    self.learning_rate = variable_scope.get_variable(
      "learning_rate", trainable=False, shape=[],
      initializer=tf.constant_initializer(float(FLAGS.learning_rate), 
                                          dtype=tf.float32))
    self.global_step = variable_scope.get_variable(
      "global_step", trainable=False, shape=[],  dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

    self.epoch = variable_scope.get_variable(
      "epoch", trainable=False, shape=[], dtype=tf.int32,
      initializer=tf.constant_initializer(0, dtype=tf.int32)) 

  def add_epoch(self, sess):
    sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))

  def get_input_feed(self, batch):
    input_feed = {}
    batch_size = batch.encoder_inputs
    encoder_size = decoder_size = self.max_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = batch.encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = batch.decoder_inputs[l]
      input_feed[self.target_weights[l].name] = batch.target_weights[l]
    if self.sequence_length != None: 
      input_feed[self.sequence_length.name] = batch.sequence_length

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([batch.batch_size], dtype=np.int32)
    return input_feed 

  def step(self, sess, raw_batch):
    batch = padding_and_format(raw_batch, self.max_sequence_length,
                               use_sequence_length=self.use_sequence_length)
    input_feed = self.get_input_feed(batch)
    output_feed = [self.losses]
    if self.do_update:
      output_feed.append(self.updates)
    outputs = sess.run(output_feed, input_feed)
    return outputs[0]


  def decode(self, sess, raw_batch):
    batch = padding_and_format(raw_batch, self.max_sequence_length,
                               use_sequence_length=self.use_sequence_length)
    input_feed = self.get_input_feed(batch)
    output_feed = [self.losses, self.e_states, self.d_states]
    for l in xrange(self.max_sequence_length):
      output_feed.append(self.logits[l])
    outputs = sess.run(output_feed, input_feed)
    losses = outputs[0]
    e_states = outputs[1]
    d_states = outputs[2]
    logits = outputs[3:]
    def greedy_argmax(logit):
      ex_list = []
      output_ids = []
      for l in logit:
        _argsorted = np.argsort(-l)
        _id = _argsorted[0] if not _argsorted[0] in ex_list else _argsorted[1]
        output_ids.append(_id)
      return output_ids
    results = [greedy_argmax(logit) for logit in logits]
    results = list(map(list, zip(*results))) # transpose to batch-major
    return losses, results


class MultiGPUTrainWrapper(object):
  def __init__(self, model_type):
    pass
