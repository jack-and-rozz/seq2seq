# coding: utf-8 
import math, time, sys
import tensorflow as tf
from core.utils import common, evaluation
from core.utils.tf_utils import shape, batch_dot, linear, cnn
from core.models.base import ModelBase
#import core.models.graph as graph
import numpy as np

##############################
##    Scoring Functions
##############################

def distmult(subjects, relations, objects):
  # tf.dynamic_partition を用いて
  # subjects : [1, hidden_size]
  # relations, objects : [triple_size, hidden_size] の場合
  
  with tf.name_scope('DistMult'):
    score = batch_dot(relations, subjects * objects, n_unk_dims=1)
    score = tf.sigmoid(score)
    return score

  ###################################################################
  # subjects : [batch_size, hidden_size]
  # relations, objects : [batch_size, n_triples, hidden_size] の場合

  # subjects = tf.expand_dims(subjects, 1)
  # score = batch_dot(relations, subjects * objects, n_unk_dims=2)
  # score = tf.sigmoid(score)
  # return score
  ###################################################################

# https://stackoverflow.com/questions/44940767/how-to-get-slices-of-different-dimensions-from-tensorflow-tensor
def extract_span(encoder_outputs, spans):
  '''
  Args:
  - encoder_outputs: [batch_size, max_num_word, hidden_size]
  '''
  with tf.name_scope('ExtractSpan'):
    def loop_func(idx, span_repls, begin, end):
      res = tf.reduce_mean(span_repls[idx][begin[idx]:end[idx]+1], axis=0)
      return tf.expand_dims(res, axis=0)

    beginning_of_link, end_of_link = tf.unstack(spans, axis=1)
    batch_size = shape(encoder_outputs, 0)
    hidden_size = shape(encoder_outputs, -1)
    idx = tf.zeros((), dtype=tf.int32)

    # Continue concatenating the obtained representation of each span. 
    res = tf.zeros((0, hidden_size))
    cond = lambda idx, res: idx < batch_size
    body = lambda idx, res: (idx + 1, tf.concat([res, loop_func(idx, encoder_outputs, beginning_of_link, end_of_link)], axis=0))
    loop_vars = [idx, res]
    _, res = tf.while_loop(
      cond, body, loop_vars,
      shape_invariants=[idx.get_shape(),
                        tf.TensorShape([None, hidden_size])])
    return res
    #spans_by_subj = tf.dynamic_partition(res, entity_indices, max_batch_size)
    # Apply max-pooling for spans of an entity.
    #spans_by_subj = tf.stack([tf.reduce_max(s, axis=0) for s in spans_by_subj], 
    #                           axis=0)
    #return spans_by_subj

def calc_precision_recall(results):
  pass

def evaluate(results):
  for batch, batched_results in results:
    for l, prediction in zip(batch, batched_results):
      
      print (l, prediction)
      exit(1)
      
class GraphLinkPrediction(ModelBase):
  def __init__(self, sess, config, encoder, vocab,
               activation=tf.nn.relu):
    super(GraphLinkPrediction, self).__init__(sess, config)
    self.dataset = config.dataset.name
    self.sess = sess
    self.encoder = encoder
    self.vocab = vocab
    self.activation = activation

    self.is_training = encoder.is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.ffnn_size = config.ffnn_size 
    self.scoring_function = distmult
    #self.max_batch_size = config.batch_size # for tf.dynamic_partition

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.text_ph = common.dotDict()
      self.text_ph.word = tf.placeholder(
        tf.int32, name='text.word',
        shape=[None, None]) if self.encoder.wbase else None
      self.text_ph.char = tf.placeholder(
        tf.int32, name='text.char',
        shape=[None, None, None]) if self.encoder.cbase else None


      self.subj_ph = tf.placeholder(
        tf.int32, name='subj.position', shape=[None, 2]) 
      self.obj_ph = tf.placeholder(
        tf.int32, name='obj.position', shape=[None, 2]) 

      self.rel_ph = common.dotDict()
      self.rel_ph.word =  tf.placeholder(
        tf.int32, name='rel.word',
        shape=[None, None]) if self.encoder.wbase else None
      self.rel_ph.char =  tf.placeholder(
        tf.int32, name='rel.char',
        shape=[None, None, None]) if self.encoder.cbase else None
      self.target_ph = tf.placeholder(
        tf.int32, name='target', shape=[None])
      self.sentence_length = tf.count_nonzero(self.text_ph.word, axis=1)

    with tf.name_scope('Encoder'):
      text_emb, outputs, state = self.encoder.encode([self.text_ph.word, self.text_ph.char], self.sentence_length)


    with tf.variable_scope('Subject') as scope:
      subj_outputs = extract_span(outputs, self.subj_ph)

    with tf.variable_scope('Relation') as scope:
      rel_outputs = tf.stop_gradient(self.encoder.word_encoder.encode([self.rel_ph.word, self.rel_ph.char]))
      rel_outputs = cnn(rel_outputs, filter_sizes=[2, 3])

    with tf.variable_scope('Object') as scope:
      obj_outputs = extract_span(outputs, self.obj_ph)

    with tf.variable_scope('Inference'):
      self.outputs = self.inference(subj_outputs, rel_outputs, obj_outputs) # [batch_size]

    with tf.name_scope("Loss"):
      self.losses = self.cross_entropy(self.outputs, self.target_ph)
      self.loss = tf.reduce_mean(self.losses)
    self.debug_ops = [self.sentence_length]

  def inference(self, subj, rel, obj):
    with tf.variable_scope('ffnn1'):
      triple = tf.concat([subj, rel, obj], axis=1)

      triple = tf.nn.dropout(triple, self.keep_prob)
      triple = linear(triple, output_size=self.ffnn_size, 
                      activation=self.activation)
      triple = tf.nn.dropout(triple, self.keep_prob)

    with tf.variable_scope('ffnn2'):
      score = linear(triple, output_size=1, activation=tf.nn.sigmoid) # true or false
    return score

  def cross_entropy(self, scores, labels):
    with tf.name_scope('logits'):
      logits = tf.concat([
        tf.maximum(1.0 - scores, tf.constant(1e-5)),  # 0 = false
        tf.maximum(scores, tf.constant(1e-5)),        # 1 = true
      ], axis=1) 

    with tf.name_scope('cross_entropy'):
      return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                            labels=labels)

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training
    input_feed[self.text_ph.word] = batch.text.word
    input_feed[self.text_ph.char] = batch.text.char
    input_feed[self.rel_ph.word] = batch.rel.word
    input_feed[self.rel_ph.char] = batch.rel.char
    input_feed[self.subj_ph] = batch.subj.position
    input_feed[self.obj_ph] = batch.obj.position
    input_feed[self.target_ph] = batch.label
    return input_feed

  def test(self, batches):
    t = time.time()
    results = []
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      outputs = self.sess.run(self.outputs, input_feed)
      results.append((batch, outputs))
    results, summary = evaluate(results)
    pprint(results)
    exit(1)
    return results, summary

  def print_batch(self, batch):
    return

#def summarize_results

