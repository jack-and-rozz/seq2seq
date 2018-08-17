# coding: utf-8 
from pprint import pprint
import math, time, sys, copy
import tensorflow as tf
import numpy as np

from core.utils.common import dotDict, recDotDict, recDotDefaultDict, flatten_batch, dbgprint, RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, dbgprint, flatten_recdict
from core.utils.tf_utils import shape, batch_dot, linear, cnn, make_summary
from core.models.base import ModelBase
from core.vocabulary.base import UNK_ID, PAD_ID
from core.models.wikiP2D.coref.coref import CoreferenceResolution
from core.dataset.wikiP2D import WikiP2DRelExDataset as dataset_class

  
class GraphLinkPredictionNoObj(CoreferenceResolution):
  def __init__(self, sess, config, encoder, activation=tf.nn.relu):
    ModelBase.__init__(self, sess, config)
    self.sess = sess
    self.encoder = encoder
    self.vocab = self.encoder.vocab
    self.is_training = encoder.is_training
    self.activation = activation
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    self.embedding_size = config.embedding_size
    self.mention_ratio = config.mention_ratio
    self.ffnn_depth = config.ffnn_depth
    self.ffnn_size = config.ffnn_size
    self.max_mention_width = config.max_mention_width
    self.use_width_feature = config.use_width_feature

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.ph = recDotDefaultDict()
      self.ph.text.word = tf.placeholder(
        tf.int32, name='text.word',
        shape=[None, None]) if self.encoder.wbase else None
      self.ph.text.char = tf.placeholder(
        tf.int32, name='text.char',
        shape=[None, None, None]) if self.encoder.cbase else None

      self.ph.query = tf.placeholder(
        tf.int32, name='query.position', shape=[2]) 
      self.ph.gold_mentions = tf.placeholder(
        tf.int32, name='query.position', shape=[None, 2])  # [num_mentions, 2]
      self.ph.target.subjective = tf.placeholder(
        tf.int32, name='target.subject', shape=[None, self.max_mention_width]) # [max_sequence_len, max_mention_width] 
      self.ph.target.objective = tf.placeholder(
        tf.int32, name='target.object', shape=[None, self.max_mention_width]) # [max_sequence_len, max_mention_width]
      self.sentence_length = tf.count_nonzero(self.ph.text.word, axis=-1)
      self.placeholders = flatten_recdict(self.ph)

    with tf.variable_scope('Embeddings'):
      self.mention_width_emb = self.initialize_embeddings(
        "mention_width", [self.max_mention_width+1, self.embedding_size.width])

    word_repls = encoder.word_encoder.word_encode(self.ph.text.word)
    char_repls = encoder.word_encoder.char_encode(self.ph.text.char)
    text_emb, text_outputs, state = encoder.encode([word_repls, char_repls], 
                                                   self.sentence_length)
    self.adv_outputs = text_outputs # for adversarial MTL, it must have the shape [batch_size]

    with tf.name_scope('get_mentions'):
      flattened_text_emb, flattened_text_outputs, flattened_sentence_indices = self.flatten_doc_to_sent(text_emb, text_outputs, self.sentence_length)
      _, _, _, mention_starts, mention_ends, mention_scores, mention_emb = self.get_mentions(flattened_text_emb, flattened_text_outputs, flattened_sentence_indices)

    with tf.name_scope('get_query_emb'):
      query_starts, query_ends = tf.unstack(tf.expand_dims(self.ph.query, 0), axis=-1)
      query_emb = self.get_mention_emb(
        flattened_text_emb, flattened_text_outputs, query_starts, query_ends)

    with tf.name_scope('get_rel_probabilities'):
      with tf.name_scope('subj_logits'):
        subj_logits = self.inference(query_emb, mention_emb, mention_scores, True,)
      with tf.name_scope('obj_logits'):
        obj_logits = self.inference(query_emb, mention_emb, mention_scores,
                                    False, reuse=True)

    # Concatenated [subjective, objective] relations with each mention.
    with tf.name_scope('prediction'):
      predicted_relations =  tf.concat(
        [tf.expand_dims(tf.argmax(subj_logits, axis=-1), -1), 
         tf.expand_dims(tf.argmax(obj_logits, axis=-1), -1)], axis=-1) # [num_mentions, 2]
      self.predictions = [
        predicted_relations,
        mention_starts,
        mention_ends
      ]
    with tf.name_scope('loss'):
      mention_indices = tf.stack([mention_starts, mention_ends-mention_starts], 
                                 axis=-1) # [num_mentions, 2]
      subj_targets = tf.gather_nd(self.ph.target.subjective, mention_indices) # [num_mentions]
      obj_targets = tf.gather_nd(self.ph.target.objective, mention_indices) # [num_mentions]
      subj_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=subj_logits, labels=subj_targets)
      obj_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=obj_logits, labels=obj_targets)
      self.loss = tf.reduce_mean(tf.concat([subj_losses, obj_losses], axis=-1))
      # self.debug_ops += [
      #   tf.concat([subj_losses, obj_losses], axis=-1),
      #   subj_logits, 
      #   predicted_relations,
      #   subj_targets,
      #   obj_targets,
      # ]

  def inference(self, query_emb, mention_emb, mention_scores, 
                is_query_subjective, reuse=None):
    '''
    Args:
    - query_emb:
    - mention_emb:
    - is_query_subjective: A boolean. If true, this function outputs a distribution of relation label probabilities for a triple (query, rel, mention) across rel, otherwise for (mention, rel, query)
    - reuse: A boolean. The variables of this network should be reused by both query-subjective and query-objective predictions by switching the orders of input representations.
    '''
    with tf.variable_scope('pair_emb', reuse=reuse):
      n_mentions = shape(mention_emb, -2)
      query_emb = tf.tile(query_emb, [n_mentions, 1]) # [n_mentions, emb]
      if is_query_subjective:
        pair_emb = tf.concat([query_emb, mention_emb], -1) # [n_mentions, emb]
      else:
        pair_emb = tf.concat([mention_emb, query_emb], -1) # [n_mentions, emb]

      for i in range(self.ffnn_depth):
        with tf.variable_scope('Forward%d' % i ):
          pair_emb = linear(pair_emb, output_size=self.ffnn_size, 
                            activation=self.activation)
          pair_emb = tf.nn.dropout(pair_emb, keep_prob=self.keep_prob)
      with tf.variable_scope('Output'):
        logits = linear(pair_emb, output_size=self.vocab.rel.size)

      # This penalty enables to make the mention scores to no mentions lower, since learning that a mention does not have a relation promotes this mention_unconfidence_penalty larger.
      mention_unconfidence_penalty = tf.concat([
        -tf.expand_dims(mention_scores, 1),
        #tf.tile(tf.expand_dims(mention_scores, 1), [1, shape(logits, 1)-1])
        tf.zeros([shape(logits, 0), shape(logits, 1) - 1], dtype=tf.float32)
       ], axis=-1)
      return logits + mention_unconfidence_penalty

  def test(self, batches, mode, logger, output_path=None):
    results = []
    used_batches = []
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      outputs = self.sess.run(self.predictions, input_feed)
      try:
        used_batches += flatten_batch(batch)
      except Exception as e:
        pprint(batch)
        print(e)
        exit(1)
      results.append(outputs)

    sys.stdout = open(output_path, 'w') if output_path else sys.stdout

    gold_triples, predicted_triples = dataset_class.formatize_and_print(
      used_batches, results, vocab=self.encoder.vocab)
    precision, recall, f1 = dataset_class.evaluate(gold_triples, predicted_triples)

    sys.stdout = sys.__stdout__
    if output_path:
      sys.stderr.write("Output the testing results to \'{}\' .\n".format(output_path))
    summary_dict = {}
    summary_dict['knowledge/%s/f1' % mode] = f1
    summary_dict['knowledge/%s/precision' % mode] = precision
    summary_dict['knowledge/%s/recall' % mode] = recall
    summary = make_summary(summary_dict)
    return f1, summary


  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training

    # todo: 今はバッチ処理出来ないので強制的にbatch_size == 1
    assert batch.text.word.shape[0] == 1
    assert batch.text.char.shape[0] == 1
    input_feed[self.ph.text.word] = batch.text.word[0]
    input_feed[self.ph.text.char] = batch.text.char[0]

    input_feed[self.ph.query] = batch.query.flat_position[0]
    input_feed[self.ph.gold_mentions] = batch.mentions.flat_position[0]
    input_feed[self.ph.target.subjective] = batch.target.subjective[0]
    input_feed[self.ph.target.objective] = batch.target.objective[0]
    return input_feed

