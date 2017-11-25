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
#class GraphLinkPrediction(object):
  def __init__(self, sess, config, is_training, encoder, o_vocab, r_vocab,
               activation=tf.nn.tanh):
    self.name = 'graph'
    self.dataset = 'wikiP2D'
    self.sess = sess
    self.encoder = encoder
    self.is_training = is_training
    self.activation = activation
    self.scoring_function = distmult
    self.max_batch_size = config.batch_size # for tf.dynamic_partition
    self.max_sent_length = config.max_a_sent_length

    # Placeholders
    ## The length of sentences is increased by 2, because BOS and EOS are inserted at the start and the end of a sent.

    #n_offset = self.encoder.w_vocab.n_start_offset + self.encoder.w_vocab.n_end_offset
    with tf.name_scope('Placeholder'):
      self.w_sentences = tf.placeholder(tf.int32, name='w_sentences',
                                        shape=[None, None])
                                      #shape=[None, self.max_sent_length+n_offset])
      self.c_sentences = tf.placeholder(tf.int32, name='c_sentences',
                                      shape=[None, None, None])
                                      #shape=[None, self.max_sent_length+n_offset, None])
      self.sentence_length = tf.placeholder(tf.int32, shape=[None], name="sentence_length")

      ## Sentences of each entity are fed as a 2D Tensor (this is because the number of links is different depending on the linked entity),  we need to dynamically sort them.
      self.entity_indices = tf.placeholder(tf.int32, shape=[None], 
                                           name='entity_indices')
      self.link_spans = tf.placeholder(tf.int32, shape=[None, 2], 
                                       name='link_spans')
      
      self.p_triples = tf.placeholder(tf.int32, shape=[None, 2], 
                                      name='positive_triples')
      self.n_triples = tf.placeholder(tf.int32, shape=[None, 2],
                                      name='negative_triples')
      
      self.pt_indices = tf.placeholder(tf.int32, shape=[None],
                                       name='pt_indices')
      self.nt_indices = tf.placeholder(tf.int32, shape=[None],
                                       name='nt_indices')
    ## Embeddings
    with tf.variable_scope('Embeddings'):
      self.o_embeddings = self.initialize_embeddings('object', [o_vocab.size, config.hidden_size])
      self.r_embeddings = self.initialize_embeddings('relation', [r_vocab.size, config.hidden_size])
    ## Define Loss
    #with tf.name_scope("loss"):
    outputs, state = encoder.encode([self.w_sentences, self.c_sentences], 
                                    self.sentence_length)
    span_outputs = encoder.extract_span(outputs, self.link_spans,
                                        self.entity_indices,
                                        self.max_batch_size)

    with tf.name_scope('Positives'):
      self.positives = self.inference(span_outputs, self.p_triples,
                                      self.pt_indices)
    with tf.name_scope('Negatives'):
      self.negatives = self.inference(span_outputs, self.n_triples, 
                                        self.nt_indices)

    self.outputs = [self.positives, self.negatives]
    self.losses = self.cross_entropy(self.positives, self.negatives)
    self.loss = tf.reduce_mean(self.losses)

    with tf.name_scope("Summary"):
      self.summary_loss = tf.placeholder(tf.float32, shape=[],
                                         name='graph_loss')
      self.summary_mean_rank = tf.placeholder(tf.float32, shape=[],
                                              name='graph_mean_rank')
      self.summary_mrr = tf.placeholder(tf.float32, shape=[],
                                        name='graph_mrr')
      self.summary_hits_10 = tf.placeholder(tf.float32, shape=[],
                                            name='graph_hits_10')

  def inference(self, span_repls, triples, batch_indices):
    with tf.name_scope('inference'):
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
      c_ent = tf.concat([ce1, ce2], 0)
    return c_ent

  def summarize_results(self, raw_batch, positives, negatives):
    p_triples = raw_batch['p_triples']
    n_triples = raw_batch['n_triples']
    batch_size = len(p_triples)
    if not n_triples:
      n_triples = [[[] for _ in xrange(len(p_triples[b]))] for b in xrange(batch_size)]

    scores = [] 
    for b in xrange(batch_size): # per an article
      scores_by_pt = []
      if len(positives[b]) == 0:
        continue
      n_neg = int(len(negatives[b]) / len(positives[b]))
      negatives_by_p = [negatives[b][i*n_neg:(i+1)*n_neg] for i in xrange(len(positives[b]))]

      for p, ns, pt, nts in zip(positives[b], negatives_by_p, 
                                 p_triples[b], n_triples[b]):
        if p > 0.0: # Remove the results of padding triples.
          _triples = [pt] + nts
          _scores = np.insert(ns, 0, p)
          #scores_by_pt.append((_triples, _scores[:len(_triples)]))
          scores_by_pt.append(_scores[:len(_triples)])
      scores.append(scores_by_pt)
    ranks = [[evaluation.get_rank(scores_by_pt) for scores_by_pt in scores_by_art] for scores_by_art in scores]
    return scores, ranks #[batch_size, p]

  def get_input_feed(self, batch):
    input_feed = {}
    ## Sentences
    if len(batch['c_articles']) != len(batch['w_articles']) or len(batch['link_spans']) != len(batch['w_articles']):
      raise ValueError('The length of \'w_articles\', \'c_articles\', and \'link_spans\' must be equal (must have the same number of entity)')

    if self.encoder.cbase:
      entity_indices, c_sentences = common.flatten_with_idx(batch['c_articles'])
      c_sentences, sentence_length, word_length = self.encoder.c_vocab.padding(
        c_sentences, self.max_sent_length)
      input_feed[self.c_sentences] = np.array(c_sentences)

    if self.encoder.wbase:
      entity_indices, w_sentences = common.flatten_with_idx(batch['w_articles'])
      w_sentences, sentence_length =  self.encoder.w_vocab.padding(
        w_sentences, self.max_sent_length)
      input_feed[self.w_sentences] = np.array(w_sentences)
      input_feed[self.sentence_length] = np.array(sentence_length)

    input_feed[self.entity_indices] = np.array(entity_indices)
    entity_indices, link_spans = common.flatten_with_idx(batch['link_spans'])
    start_offset = self.encoder.w_vocab.n_start_offset
    link_spans = [(s+start_offset, e+start_offset) for (s, e) in link_spans] # Shifted back one position by BOS.
    input_feed[self.link_spans] = np.array(link_spans)

    ## Triples
    PAD_TRIPLE = (0, 0)
    def padding_triples(triples):
      max_num_pt = max([len(t) for t in triples])
      padded = [([1.0] * len(t) + [0.0] * (max_num_pt - len(t)),
                 list(t) + [PAD_TRIPLE] * (max_num_pt - len(t))) for t in triples]
      return map(list, zip(*padded)) # weights, padded_triples

    def fake_triples(batch_size):
      res = [([0.0], [PAD_TRIPLE]) for i in xrange(batch_size)]
      weights, triples = map(list, zip(*res))
      return weights, triples

    pt_indices, p_triples = common.flatten_with_idx(batch['p_triples'])
    input_feed[self.p_triples] = np.array(p_triples)
    input_feed[self.pt_indices] = np.array(pt_indices)

    if batch['n_triples']:
      n_triples = [common.flatten(t) for t in batch['n_triples']] # negative triples are divided by the corresponding positive triples.
    else:
      _, n_triples = fake_triples(1)
    nt_indices, n_triples = common.flatten_with_idx(n_triples)
    input_feed[self.n_triples] = np.array(n_triples)
    input_feed[self.nt_indices] = np.array(nt_indices)
    return input_feed

  def test(self, batches):
    scores = []
    ranks = []
    t = time.time()
    for i, raw_batch in enumerate(batches):
      input_feed = self.get_input_feed(raw_batch)
      outputs = self.sess.run(self.outputs, input_feed)
      positives, negatives = outputs
      _scores, _ranks = self.summarize_results(raw_batch, positives, negatives)
      scores.append(_scores)
      ranks.append(_ranks)
      t = time.time()
      break

    f_ranks = [x[0] for x in common.flatten(common.flatten(ranks))] # batch-loop, article-loop
    mean_rank = sum(f_ranks) / len(f_ranks)
    mrr = evaluation.mrr(f_ranks)
    hits_10 = evaluation.hits_k(f_ranks)

    ## return summary
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
    return summary, (scores, ranks, mrr, hits_10)

  def print_results(self, batches, scores, ranks, output_file=None):
    cnt = 0
    if output_file:
      sys.stdout = output_file

    for batch, score_by_batch, ranks_by_batch in zip(batches, scores, ranks): # per a batch
      for batch_by_art, score_by_art, rank_by_art in zip(self.w2p_dataset.batch2text(batch), score_by_batch, ranks_by_batch): # per an article
        ent_name, wa, ca, pts = batch_by_art
        print common.colored('<%d> : %s' % (cnt, ent_name), 'bold')
        #print common.colored("Article(word):", 'bold')
        #print "\n".join(wa) + '\n'
        print common.colored("Article(char):", 'bold')
        print "\n".join(ca) + '\n'
        print common.colored("Triple, Score, Rank:", 'bold')
        for (r, o), scores, rank in zip(pts, score_by_art, rank_by_art): # per a positive triple
          s = scores[0] # scores = [pos, neg_0, neg_1, ...]
          N = 5
          pos_rank, sorted_idx = rank
          pos_id = self.o_vocab.name2id(o)
          idx2id = [pos_id] + [x for x in xrange(self.o_vocab.size) if x != pos_id] # pos_objectを先頭に持ってきているのでidxを並び替え

          top_n_scores = [scores[idx] for idx in sorted_idx[:N]]
          top_n_objs = [self.o_vocab.id2name(idx2id[x]) 
                        for x in sorted_idx[:N]]
          top_n = ", ".join(["%s:%.3f" % (x, score) for x, score in 
                             zip(top_n_objs, top_n_scores)])
          print "(%s, %s) - %f, %d, [Top-%d Objects]: %s" % (r, o, s, pos_rank, N, top_n) 
        print
        cnt += 1 
    sys.stdout = sys.__stdout__
