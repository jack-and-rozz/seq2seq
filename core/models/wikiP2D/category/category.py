# coding: utf-8 
import math, time, sys, random 
import numpy as np
import tensorflow as tf
from pprint import pprint 

from core.utils.tf_utils import shape, linear, make_summary
from core.utils.common import dotDict, recDotDefaultDict, flatten_batch, dbgprint, RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, timewatch
from core.vocabulary.base import _PAD
from core.models.base import ModelBase
from core.models.wikiP2D.category.evaluation import evaluate_and_print

class CategoryClassification(ModelBase):
  def __init__(self, sess, config, encoder,
               activation=tf.nn.relu):
    super(CategoryClassification, self).__init__(sess, config)
    self.sess = sess
    self.encoder = encoder
    self.activation = activation
    self.is_training = encoder.is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.vocab = encoder.vocab

    with tf.name_scope('Placeholder'):
      self.ph = recDotDefaultDict()
      # [batch_size, max_num_context, max_num_words]
      self.ph.text.word = tf.placeholder(
        tf.int32, name='contexts.word',
        shape=[None, None, None]) if self.encoder.wbase else None
      self.ph.text.char = tf.placeholder(
        tf.int32, name='contexts.char',
        shape=[None, None, None, None]) if self.encoder.cbase else None

      self.ph.link = tf.placeholder(
        tf.int32, name='link', shape=[None, None, 2]) 
      self.ph.target = tf.placeholder(
        tf.int32, name='link', shape=[None]) 

      self.sentence_length = tf.count_nonzero(self.ph.text.word, axis=-1)
      self.num_contexts = tf.cast(tf.count_nonzero(self.sentence_length, axis=-1),
                                  tf.float32)

    with tf.name_scope('Encoder'):
      word_repls = encoder.word_encoder.word_encode(self.ph.text.word)
      char_repls = encoder.word_encoder.char_encode(self.ph.text.char)

      text_emb, text_outputs, state = encoder.encode([word_repls, char_repls], 
                                                     self.sentence_length) 
      mention_starts, mention_ends = tf.unstack(self.ph.link, axis=-1)

      mention_repls, head_scores = encoder.get_batched_mention_emb(
        text_emb, text_outputs, mention_starts, mention_ends) # [batch_size, max_n_contexts, mention_size]
      self.adv_outputs = tf.reshape(text_outputs, [shape(text_outputs, 0) * shape(text_outputs, 1), shape(text_outputs, 2), shape(text_outputs, 3)]) # [batch_size * max_n_contexts, max_sentence_length, output_size]

    with tf.variable_scope('Inference'):
      self.outputs = self.inference(mention_repls)
      self.predictions = tf.argmax(self.outputs, axis=-1)

    with tf.name_scope("Loss"):
      self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.outputs, labels=self.ph.target)

      self.loss = tf.reduce_mean(self.losses)
    #self.debug_ops = [self.ph.text.word, self.sentence_length, self.num_contexts] 

  def inference(self, mention_repls):
    # Take sum of all the context representation by entity.
    mention_repls = tf.reduce_sum(mention_repls, axis=1) # [batch_size, output_size]

    # Devide the aggregated mention representations by the actual number of the contexts, since some of sentences fed to placeholders can be dummy.
    mention_repls /= tf.expand_dims(self.num_contexts, axis=1)

    # <memo> Don't apply tf.softmax when tf.nn.sparse_softmax_cross_entropy_with_logits as is employed as loss function, which contains softmax on the inside.
    #outputs = tf.nn.softmax(linear(mention_repls, self.vocab.category.size))
    outputs = linear(mention_repls, self.vocab.category.size)
    return outputs

  @timewatch()
  def test(self, batches, mode, logger, output_path):
    return self.evaluate(
      batches, mode, output_path=output_path)

  def evaluate(self, batches, mode, output_path=None):
    results = []
    used_batches = []
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      #outputs = np.array([random.randint(0, self.vocab.category.size-1)
      #                    for _ in range(batch.contexts.word.shape[0])])
      outputs = self.sess.run(self.predictions, input_feed)
      try:
        used_batches += flatten_batch(batch)
      except Exception as e:
        pprint(batch)
        print(e)
        exit(1)
      results.append(outputs)
    results = np.concatenate(results, axis=0)

    sys.stdout = open(output_path, 'w') if output_path else sys.stdout
    accuracy =  evaluate_and_print(used_batches, results, 
                                   vocab=self.encoder.vocab)
    sys.stdout = sys.__stdout__
    if output_path:
      sys.stderr.write("Output the testing results to \'{}\' .\n".format(output_path))


    summary_dict = {}
    summary_dict['category/%s/Accuracy' % mode] = accuracy
    summary = make_summary(summary_dict)
    return accuracy, summary

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    input_feed[self.ph.text.word] = batch.contexts.word
    input_feed[self.ph.text.char] = batch.contexts.char
    input_feed[self.ph.link] = batch.contexts.link
    input_feed[self.ph.target] = batch.category.label
    return input_feed

