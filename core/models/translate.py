# coding: utf-8
from __future__ import absolute_import
from __future__ import division

import random, sys, os, math, copy, time
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell

from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from core.model.base import ModelBase
from core.utils import common
from core.utils.dataset import padding_and_format
from core.utils.vocabulary.base import  PAD_ID, GO_ID, EOS_ID, UNK_ID, VecVocabulary
from core.seq2seq import seq2seq, encoders, decoders, rnn
from core.seq2seq.beam_search import follow_path

class Baseline(ModelBase):
  def __init__(self, sess, FLAGS, forward_only, do_update, 
               s_vocab=None, t_vocab=None):
    #super(Baseline, self).__init__(sess, FLAGS, do_update)
    self.initialize(sess, config, do_update)
    self.forward_only=forward_only
    self.read_flags(FLAGS)

    with tf.name_scope('placeholders'):
      self.setup_placeholders(use_sequence_length=self.use_sequence_length)
    #cell = self.setup_cell(self.cell_type, self.num_layers, 1.0, self.keep_prob,
    #                       state_is_tuple=False)
    cell = rnn.setup_cell(self.cell_type, self.num_layers, 1.0, self.keep_prob,
                          state_is_tuple=False)

    with tf.variable_scope("Encoder") as encoder_scope:
      encoder_embedding = self.initialize_embedding(
        FLAGS.source_vocab_size,
        FLAGS.embedding_size, 
        vocab=s_vocab,
        trainable=FLAGS.trainable_source_embedding)
      encoder = getattr(encoders, FLAGS.encoder_type)(
        cell, encoder_embedding,
        scope=encoder_scope, 
        sequence_length=self.sequence_length)

    with tf.variable_scope("Decoder") as decoder_scope:
      decoder_embedding = self.initialize_embedding(
        FLAGS.target_vocab_size,
        FLAGS.embedding_size,
        trainable=FLAGS.trainable_target_embedding)
      decoder = getattr(decoders, FLAGS.decoder_type)(
        copy.deepcopy(cell), decoder_embedding, scope=decoder_scope)
    self.seq2seq = getattr(seq2seq, FLAGS.seq2seq_type)(
      encoder, decoder, FLAGS.num_samples, self.batch_size,
      feed_previous=forward_only, beam_size=FLAGS.beam_size)

    # The last tokens in decoder_inputs are not to set each length of placeholders to be same.
    res = self.seq2seq(
      self.encoder_inputs, self.decoder_inputs[:-1],
      self.targets, self.target_weights[:-1])
    if self.forward_only and FLAGS.beam_size > 1:
      self.beam_pathes, self.beam_symbols, self.d_states = res
    else:
      self.logits, self.losses, self.d_states = res

    if do_update:
      with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        self.updates = self.setup_updates(self.losses)
    self.saver = tf.train.Saver(tf.global_variables())

  def initialize_embedding(self, vocab_size, embedding_size, 
                           vocab=None, trainable=True):
    if isinstance(vocab, VecVocabulary): # if pre-trained embeddings are provided
      initializer = tf.constant_initializer(vocab.embedding)
    else:
      sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
      initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

    if not trainable: # use cpu
      with tf.device('/cpu:0'):
        embedding = tf.get_variable(
          "embedding", [vocab_size, embedding_size],
          initializer=initializer, trainable=trainable)
    else:
      embedding = tf.get_variable(
        "embedding", [vocab_size, embedding_size],
        initializer=initializer, trainable=trainable)
    return embedding

  def setup_updates(self, loss):
    params = tf.trainable_variables()
    gradients = []
    updates = []
    opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = [grad for grad, _ in opt.compute_gradients(loss)]
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  self.max_gradient_norm)
    self.grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
    updates = opt.apply_gradients(self.grad_and_vars, global_step=self.global_step)
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
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))
    # Our targets are decoder inputs shifted by one.
    self.targets = [self.decoder_inputs[i + 1]
                    for i in xrange(len(self.decoder_inputs) - 1)]
    self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length") if use_sequence_length else None
    # For beam_decoding, we have to feed the size of a current batch.
    self.batch_size = tf.placeholder(tf.int32, shape=[])

  # def setup_cell(self, cell_type, num_layers=1, 
  #                in_keep_prob=1.0, out_keep_prob=1.0):
  #   cell = getattr(rnn_cell, self.cell_type)(self.hidden_size, reuse=tf.get_variable_scope().reuse) 
  #   if in_keep_prob < 1.0 or out_keep_prob < 1.0:
  #     cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
  #   if num_layers > 1:
  #     cell = rnn_cell.MultiRNNCell([copy.deepcopy(cell) for _ in xrange(self.num_layers)],
  #                                  state_is_tuple=False)
  #   return cell

  def read_flags(self, FLAGS):
    self.max_sequence_length = FLAGS.max_sequence_length

    self.keep_prob = FLAGS.keep_prob if self.do_update else 1.0
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
    self.beam_size = FLAGS.beam_size

  def get_input_feed(self, raw_batch):
    batch = self.padding_and_format(raw_batch,
                                    use_sequence_length=self.use_sequence_length)
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
    input_feed[self.batch_size] = batch.batch_size
    return input_feed 

  def step(self, raw_batch):
    sess = self.sess
    input_feed = self.get_input_feed(raw_batch)
    output_feed = [self.losses]
    if self.do_update:
      output_feed.append(self.updates)
    if self.global_step.eval() == 500 and False:
       run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
       run_metadata = tf.RunMetadata()
       outputs = sess.run(output_feed, input_feed,
                          options=run_options,
                          run_metadata=run_metadata)
       tl = timeline.Timeline(run_metadata.step_stats)
       with tf.gfile.Open(self.SUMMARIES_PATH + '/timeline.json', mode='w') as trace: 
         trace.write(tl.generate_chrome_trace_format(show_memory=True))
    else:
      outputs = sess.run(output_feed, input_feed)
    return outputs[0]

  def run_batch(self, data, batch_size, do_shuffle=False):
    start_time = time.time()
    loss = 0.0
    for i, raw_batch in enumerate(data.get_batch(batch_size, do_shuffle=do_shuffle)):
      step_loss = self.step(raw_batch)
      loss += step_loss 
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss = loss / (i+1)
    return epoch_time, step_time, loss

  def decode(self, raw_batch):
    sess = self.sess
    input_feed = self.get_input_feed(raw_batch)
    if self.beam_size > 1:
      losses = None
      output_feed = [self.beam_pathes, self.beam_symbols]
      # The pathes and symbols are a list of array([batch_size, beam_size])
      beam_pathes, beam_symbols = sess.run(output_feed, input_feed)
      results = []
      for p, s in zip(beam_pathes, beam_symbols):
        result = follow_path(p, s, self.beam_size)
        results.append(result)
    else:
      output_feed = [self.losses, self.d_states]
      for l in xrange(self.max_sequence_length):
        output_feed.append(self.logits[l])
      outputs = sess.run(output_feed, input_feed)
      losses = outputs[0]
      d_states = outputs[1]
      logits = outputs[2:]
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


  def padding_and_format(self, data, use_sequence_length=True):
    '''
    Caution:  if both do_reverse and use_sequence_length are True at the same time, many PAD_IDs and only a small part of a sentence are read.
    '''
    max_sequence_length = self.max_sequence_length
    do_reverse = not use_sequence_length
    batch_size = len(data)
    encoder_size, decoder_size = max_sequence_length, max_sequence_length
    encoder_inputs, decoder_inputs, encoder_sequence_length = [], [], []
    for _, encoder_input, decoder_input in data:
      encoder_sequence_length.append(len(encoder_input))
      # Encoder inputs are padded and then reversed if do_reverse=True.
      encoder_pad = [PAD_ID for _ in xrange((encoder_size - len(encoder_input)))] 
      encoder_input = encoder_input + encoder_pad
      if do_reverse:
        encoder_input = list(reversed(encoder_input))
      encoder_inputs.append(encoder_input)

      # Decoder inputs get an extra "GO" and "EOS" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 2
      decoder_inputs.append([GO_ID] + decoder_input + [EOS_ID] +
                            [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
        np.array([encoder_inputs[batch_idx][length_idx]
                  for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
        np.array([decoder_inputs[batch_idx][length_idx]
                  for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    if not use_sequence_length:
      encoder_sequence_length = None 
    batch = common.dotDict({
      'encoder_inputs' : batch_encoder_inputs,
      'decoder_inputs' : batch_decoder_inputs,
      'target_weights' : batch_weights,
      'sequence_length' : encoder_sequence_length,
      'batch_size' : batch_size,
    })
    return batch

# class AverageGradientMultiGPUTrainManager(object):
#   def __init__(self, sess, FLAGS, forward_only, do_update):
#     if do_update and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
#       m = AverageGradientMultiGPUTrainWrapper(sess, FLAGS)
#     else:
#       m = Baseline(sess, FLAGS, forward_only, do_update)

    #super(AverageGradientMultiGPUTrainWrapper, self).__init__(sess, FLAGS, forward_only, do_update, s_vocab=s_vocab, t_vocab=t_vocab)

