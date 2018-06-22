# coding: utf-8 
from pprint import pprint
import math, time, sys, copy
import tensorflow as tf
#from core.utils import common, evaluation
from core.utils.common import dotDict, flatten_batch, RED, BLUE, RESET, UNDERLINE, BOLD, GREEN
from core.utils.tf_utils import shape, batch_dot, linear, cnn, make_summary
from core.models.base import ModelBase
from core.vocabulary.base import UNK_ID, PAD_ID
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

def print_batch(batch, prediction=None, vocab=None,
                color_link=True, underline_unk=True):
  text = copy.deepcopy(batch.text.raw)
  num_words = sum([1 for w_id in batch.text.word if w_id != PAD_ID])
  if color_link:
    title_color = BLUE
    link_color = BLUE if batch.label == 1 else RED
    begin, end = batch.subj.position
    for i in range(begin, end+1):
      text[i] = title_color + text[i] + RESET
    begin, end = batch.obj.position
    for i in range(begin, end+1):
      text[i] = link_color + text[i] + RESET
  if underline_unk:
    for i, w_id in enumerate(batch.text.word):
      if w_id == UNK_ID:
        text[i] = UNDERLINE + text[i] + RESET
  text = ' '.join(text[:num_words])

  label = True if batch.label == 1 else False
  subj_str = BLUE  + ' '.join(batch.subj.raw) + RESET
  rel_str = GREEN + ' '.join(batch.rel.raw) + RESET
  obj_color = BLUE if label else RED
  obj_str = obj_color  + ' '.join(batch.obj.raw) + RESET
  triple = '(%s, %s, %s)' % (subj_str, rel_str, obj_str)

  print ('<text>:\t', text)
  print ('<triple>:\t', triple)
  if vocab:
    print ("<word-encode>:\t", vocab.word.ids2tokens(batch.text.word))
    print ("<char-encode>:\t", vocab.char.ids2tokens(batch.text.char))

  if prediction is not None:
    label = True if batch.label == 1 else False
    prediction = True if prediction == 1 else False
    print ('<label/prediction>:\t', '%s/%s' % (str(label), str(prediction)))
  print ('')
  return

def evaluate(flat_batches, predictions, vocab=None):
  def _calc_acc_prec_recall(labels, predictions):
    TP, FP, FN, TN = 0, 0, 0, 0
    for l, p in zip(labels, predictions):
      if l == 1 and p == 1:
        TP +=1
      elif l == 1 and p == 0:
        FN += 1
      elif l == 0 and p == 1:
        FP += 1
      elif l == 0 and p == 0:
        TN += 1
      else:
        raise ValueError
    print("TP, FN, FP, TN = %d, %d, %d, %d\n" % (TP, FN, FP, TN))
    acc = 1.0 * (TP+TN) / (TP+TN+FP+FN)
    prec = 1.0 * (TP) / (TP+FP) if TP+FP > 0 else 0
    recall = 1.0 * (TP) / (TP+FN) if TP+FN > 0 else 0
    return acc, prec, recall

  labels = [] 
  for i, (b, p) in enumerate(zip(flat_batches, predictions)):
    is_success = 'Success' if b.label == p else 'Fail'
    _id = '<%04d:%s>' % (i, is_success)
    if b.label == p:
      _id = BOLD + _id + RESET

    print (_id)
    print_batch(b, prediction=p, vocab=vocab)
    labels.append(b.label)
  return _calc_acc_prec_recall(labels, predictions)


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
    self.cnn_filter_widths = config.cnn.filter_widths
    self.cnn_filter_size = config.cnn.filter_size

    self.scoring_function = distmult
    #self.max_batch_size = config.batch_size # for tf.dynamic_partition

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.text_ph = dotDict()
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

      self.rel_ph = dotDict()
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
      text_emb, encoder_outputs, encoder_state = self.encoder.encode([self.text_ph.word, self.text_ph.char], self.sentence_length)


    with tf.variable_scope('Subject') as scope:
      subj_outputs = extract_span(encoder_outputs, self.subj_ph)

    with tf.variable_scope('Relation') as scope:
      # Stop gradient to prevent biased learning to the words used as relation labels.
      rel_words_emb = tf.stop_gradient(self.encoder.word_encoder.encode([self.rel_ph.word, self.rel_ph.char])) 
      with tf.name_scope("compose_words"):
        rel_outputs = cnn(rel_words_emb, 
                          self.cnn_filter_widths, 
                          self.cnn_filter_size)

    with tf.variable_scope('Object') as scope:
      obj_outputs = extract_span(encoder_outputs, self.obj_ph)

    with tf.variable_scope('Inference'):
      score_outputs = self.inference(subj_outputs, rel_outputs, obj_outputs) # [batch_size, 1]
      self.outputs = tf.round(tf.reshape(score_outputs, [shape(score_outputs, 0)])) # [batch_size]
    with tf.name_scope("Loss"):
      self.losses = self.cross_entropy(score_outputs, self.target_ph)
      self.loss = tf.reduce_mean(self.losses)
    self.debug_ops = [
      self.sentence_length, 
      tf.is_nan(encoder_outputs),
      tf.is_nan(subj_outputs),
      tf.is_nan(rel_outputs),
      tf.is_nan(obj_outputs),
      tf.is_nan(score_outputs),
    ]

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

  def test(self, batches, mode, output_path=None):
    start_time = time.time()
    results = []
    used_batches = []
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      ce = self.sess.run(self.loss, input_feed)
      outputs = self.sess.run(self.outputs, input_feed)
      used_batches += flatten_batch(batch)
      results.append(outputs)
    results = np.concatenate(results, axis=0)
    epoch_time = time.time() - start_time 
    sys.stdout = open(output_path, 'w') if output_path else sys.stdout
    acc, prec, recall = evaluate(used_batches, results, vocab=self.vocab)
    print ('acc, p, r, f = %.2f %.2f %.2f %.2f' % (
      100.0 * acc,
      100.0 * prec,
      100.0 * recall,
      100.0 * (prec + recall) /2
    ))
    sys.stdout = sys.__stdout__

    summary_dict = {}
    summary_dict['graph/%s/Accuracy' % mode] = acc
    summary_dict['graph/%s/Precision' % mode] = prec
    summary_dict['graph/%s/Recall' % mode] = recall
    summary_dict['graph/%s/F1' % mode] = (prec + recall) / 2
    summary = make_summary(summary_dict)
    return (acc, prec, recall), summary


#def summarize_results

