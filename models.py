# coding: utf-8
from __future__ import absolute_import
from __future__ import division

import random, sys, os, math, copy
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
from tensorflow.python.ops import init_ops

#from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
#import tensorflow.contrib.legacy_seq2seq as seq2seq
from seq2seq import RNNEncoder, BidirectionalRNNEncoder, RNNDecoder, BasicSeq2Seq

from utils import common
from utils.dataset import PAD_ID, GO_ID, EOS_ID, UNK_ID
dtype=tf.float32

class Baseline(object):
  def __init__(self, FLAGS, buckets, forward_only=False):
    self.buckets = buckets
    self.read_flags(FLAGS)
    with tf.name_scope('placeholders'):
      self.setup_placeholders(use_sequence_length=True)
    cell = self.setup_cell(forward_only)
    with variable_scope.variable_scope("Encoder") as encoder_scope:
      encoder_embedding = self.initialize_embedding(FLAGS.source_vocab_size,
                                                    FLAGS.embedding_size)
      encoder = BidirectionalRNNEncoder(cell, encoder_embedding,
                           scope=encoder_scope, 
                           sequence_length=self.sequence_length)

    with variable_scope.variable_scope("Decoder") as decoder_scope:
      decoder_embedding = self.initialize_embedding(FLAGS.target_vocab_size,
                                                    FLAGS.embedding_size)
      decoder = RNNDecoder(copy.deepcopy(cell), decoder_embedding, 
                           scope=decoder_scope)
    with variable_scope.variable_scope('Seq2Seq') as seq2seq_scope:
      self.seq2seq = BasicSeq2Seq(encoder, decoder, FLAGS.num_samples,
                                  feed_previous=forward_only)
      self.outputs, self.losses = self.seq2seq(
        self.encoder_inputs, self.decoder_inputs,
        self.targets, self.target_weights, self.buckets)
      if not forward_only:
        self.updates = self.setup_updates(self.losses)

  def initialize_embedding(self, vocab_size, embedding_size):
    sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
    initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
    embedding = variable_scope.get_variable(
      "embedding", [vocab_size, embedding_size],
      initializer=initializer)
    return embedding

  def setup_updates(self, losses):
    params = tf.trainable_variables()
    self.gradient_norms = []
    updates = []
    opt = tf.train.AdamOptimizer(self.learning_rate)

    for loss in losses:
      gradients = tf.gradients(loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       self.max_gradient_norm)
      for c,p in zip(gradients, params):
        print c, p
      self.gradient_norms.append(norm)
      updates.append(opt.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step))
      exit(1)
    return updates

  def setup_placeholders(self, use_sequence_length=True):
    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(self.buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    self.targets = [self.decoder_inputs[i + 1]
                    for i in xrange(len(self.decoder_inputs) - 1)]
    self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length") if use_sequence_length else None


  def setup_cell(self, forward_only):
    cell = getattr(rnn_cell, self.cell_type)(self.hidden_size, reuse=tf.get_variable_scope().reuse) 
    if self.keep_prob < 1.0 and not forward_only:
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
    if self.num_layers > 1:
      cell = rnn_cell.MultiRNNCell([cell for _ in xrange(self.num_layers)])
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
    self.learning_rate = variable_scope.get_variable(
      "learning_rate", trainable=False, shape=[1],
      initializer=tf.constant_initializer(float(FLAGS.learning_rate), 
                                          dtype=np.float32))
    self.global_step = variable_scope.get_variable(
      "global_step", trainable=False, shape=[1],
      initializer=tf.constant_initializer(0, dtype=np.int32)) 

  def get_input_feed(self, batch):
    input_feed = {}
    batch_size = batch.encoder_inputs
    encoder_size, decoder_size = self.buckets[batch.bucket_id]
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = batch.encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = batch.decoder_inputs[l]
      input_feed[self.target_weights[l].name] = batch.target_weights[l]
    if self.sequence_length != None: 
      input_feed[self.sequence_length.name] = batch.target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([batch.size], dtype=np.int32)
    return input_feed 

  def step(self, sess, batch):
    bucket_id = batch.bucket_id
    input_feed = self.get_input_feed(batch)
    exit(1)
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])
    outputs = sess.run(output_feed, input_feed)


