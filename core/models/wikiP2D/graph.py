# coding: utf-8 
import math, time, sys
import tensorflow as tf
from core.utils import common, evaluation, tf_utils
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
    score = tf_utils.batch_dot(relations, subjects * objects, n_unk_dims=1)
    score = tf.sigmoid(score)
    return score

  ###################################################################
  # subjects : [batch_size, hidden_size]
  # relations, objects : [batch_size, n_triples, hidden_size] の場合

  # subjects = tf.expand_dims(subjects, 1)
  # score = tf_utils.batch_dot(relations, subjects * objects, n_unk_dims=2)
  # score = tf.sigmoid(score)
  # return score
  ###################################################################


"""
articles : [batch_size, n_words, [n_characters]]
link_spans : [batch_size, 2 (start, end)]
triples : [None, 2 (relation_id, object_id)]
"""

class GraphLinkPrediction(ModelBase):
  def __init__(self, config, encoder, o_vocab, r_vocab,
               activation=tf.nn.tanh):
    self.name = 'graph'
    self.hidden_size = config.hidden_size
    self.encoder = encoder
    self.activation = activation
    self.scoring_function = distmult
    self.max_batch_size = config.batch_size # for tf.dynamic_partition


    self.o_embeddings = self.initialize_embeddings('o_vocab', o_vocab.size, 
                                                   config.hidden_size)
    self.r_embeddings = self.initialize_embeddings('r_vocab', r_vocab.size,
                                                   config.hidden_size)

    self.p_triples = tf.placeholder(tf.int32, shape=[None, 2], 
                                    name='positive_triples')
    self.n_triples = tf.placeholder(tf.int32, shape=[None, 2],
                                    name='negative_triples')

    self.pt_indices = tf.placeholder(tf.int32, shape=[None],
                                    name='pt_indices')
    self.nt_indices = tf.placeholder(tf.int32, shape=[None],
                                    name='nt_indices')
    ## Loss and Update
    with tf.name_scope("loss"):
      span_outputs = encoder.link_outputs
      with tf.name_scope('positives'):
        self.positives = self.inference(span_outputs, self.p_triples,
                                        self.pt_indices)
      with tf.name_scope('negatives'):
        self.negatives = self.inference(span_outputs, self.n_triples, 
                                        self.nt_indices)
      self.loss = self.cross_entropy(self.positives, self.negatives)



  def inference(self, span_repls, triples, batch_indices):
    #relations, objects = tf.unstack(triples, axis=2)
    relations, objects = tf.unstack(triples, axis=1)
    relations = self.activation(tf.nn.embedding_lookup(self.r_embeddings, relations))
    objects = self.activation(tf.nn.embedding_lookup(self.o_embeddings, objects))

    part_sbj = tf.dynamic_partition(span_repls, 
                                    tf.range(tf.shape(span_repls)[0]), 
                                    self.max_batch_size)
    part_rel = tf.dynamic_partition(relations, batch_indices, 
                                    self.max_batch_size)
    part_obj = tf.dynamic_partition(objects, batch_indices, 
                                    self.max_batch_size)

    scores = []
    for sbj, rel, obj in zip(part_sbj, part_rel, part_obj):
      score = self.scoring_function(sbj, rel, obj)
      scores.append(score)
    return scores

  def cross_entropy(self, positives, negatives):
    positives = tf.concat(positives, axis=0)
    negatives = tf.concat(negatives, axis=0)

    # calculate cross-entropy by hand.
    with tf.name_scope('cross_entropy'):
      ce1 = -tf.log(tf.maximum(positives, tf.constant(1e-6)))
      ce2 = -tf.log(tf.maximum(1 - negatives, tf.constant(1e-6)))
      c_ent = tf.reduce_mean(tf.concat([ce1, ce2], 0))
    return c_ent

  def get_input_feed(self, batch):
    input_feed = {}
    p_triples = batch['p_triples'] 
    n_triples = batch['n_triples']

    PAD_TRIPLE = (0, 0)
    def padding_triples(triples):
      max_num_pt = max([len(t) for t in triples])
      padded = [([1.0] * len(t) + [0.0] * (max_num_pt - len(t)),
                 list(t) + [PAD_TRIPLE] * (max_num_pt - len(t))) for t in triples]
      return map(list, zip(*padded)) # weights, padded_triples

    def flatten_triples(triples):
      res = common.flatten([[(i, x) for x in t] for i, t in enumerate(triples)])
      return map(list, zip(*res)) # in-batch indices, triples 

    def fake_triples(batch_size):
      res = [([0.0], [PAD_TRIPLE]) for i in xrange(batch_size)]
      weights, triples = map(list, zip(*res))
      return weights, triples

    pt_indices, p_triples = flatten_triples(p_triples)
    input_feed[self.p_triples] = np.array(p_triples)
    input_feed[self.pt_indices] = np.array(pt_indices)

    if n_triples:
      n_triples = [common.flatten(t) for t in n_triples] # negative triples are divided by the corresponding positive triples.
    else:
      #_, n_triples = fake_triples(len(p_triples))
      _, n_triples = fake_triples(1)
    nt_indices, n_triples = flatten_triples(n_triples)
    input_feed[self.n_triples] = np.array(n_triples)
    input_feed[self.nt_indices] = np.array(nt_indices)
    return input_feed
