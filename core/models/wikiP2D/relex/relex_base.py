# coding: utf-8 
from pprint import pprint
import math, time, sys, copy
import tensorflow as tf
import numpy as np
from core.dataset import padding
from core.utils.common import dotDict, recDotDict, recDotDefaultDict, flatten_batch, dbgprint, flatten_recdict
from core.utils.tf_utils import shape, batch_dot, linear, cnn, make_summary
from core.models.base import ModelBase
from core.vocabulary.base import UNK_ID, PAD_ID
from core.models.wikiP2D.coref.coref import CoreferenceResolution
from core.dataset.wikiP2D import WikiP2DRelExDataset as dataset_class

  
class RelationExtraction(CoreferenceResolution):
  def __init__(self, sess, config, encoder, activation=tf.nn.relu):
    ModelBase.__init__(self, sess, config)
    self.sess = sess
    self.encoder = encoder
    self.vocab = self.encoder.vocab
    self.is_training = encoder.is_training
    self.activation = activation
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.max_batch_size = config.batch_size
    self.embedding_size = config.embedding_size
    self.mention_ratio = config.mention_ratio
    self.ffnn_depth = config.ffnn_depth
    self.ffnn_size = config.ffnn_size
    self.max_mention_width = config.max_mention_width
    self.use_width_feature = config.use_width_feature
    self.use_gold_mentions = config.use_gold_mentions
    self.use_predicted_mentions = config.use_predicted_mentions
    self.rel_cnn = config.cnn

    self.reuse = None
    self.ph = ph = self.setup_placeholder()

    with tf.variable_scope('Embeddings'):
      self.mention_width_emb = self.initialize_embeddings(
        "mention_width", [self.max_mention_width+1, self.embedding_size.width])

    with tf.name_scope('encode'):
      sentence_length = tf.count_nonzero(ph.text.word, axis=-1)
      word_repls = self.encoder.word_encoder.word_encode(ph.text.word)
      char_repls = self.encoder.word_encoder.char_encode(ph.text.char)
      text_emb, text_outputs, state = self.encoder.encode(
        [word_repls, char_repls], sentence_length)

    with tf.name_scope('relation_matrix'):
      if config.encode_rel_names:
        rel_w = tf.transpose(self.encode_rel_names(self.vocab)) 
      else:
        rel_w = tf.get_variable(
          "rel_w", [self.ffnn_size, self.vocab.rel.size-1])

      #no_relation_vec = tf.zeros([shape(rel_w, -1)])
      #self.rel_w = tf.concat([tf.expand_dims(no_relation_vec, rel_w], 0)
      self.rel_w = rel_w
      print(self.rel_w)


    #self.adv_outputs = text_outputs # for adversarial MTL, it must have the shape [batch_size]
    self.predictions, self.loss = self.batch_inference(
      text_emb, text_outputs, sentence_length)
    self.debug_ops = [self.predictions[0]]

  def encode_rel_names(self, vocab):
    rel_names = dotDict()
    rel_names.raw = self.vocab.rel.rev_names[1:]
    rel_names.word = [vocab.word.sent2ids(rel_name) for rel_name in rel_names.raw]
    rel_names.char = [vocab.char.sent2ids(rel_name) for rel_name in rel_names.raw]

    rel_names.word = padding(
      rel_names.word, 
      minlen=[max(self.rel_cnn.filter_widths)], 
      maxlen=[0])
    rel_names.char = padding(
      rel_names.char, 
      minlen=[0, max(self.rel_cnn.filter_widths)], 
      maxlen=[0, 0])

    rel_embs = dotDict()
    rel_embs.word = self.encoder.word_encoder.word_encode(rel_names.word)
    rel_embs.char = self.encoder.word_encoder.char_encode(rel_names.char)
    
    with tf.variable_scope('RelWordsComposition'):
      rel_repls = cnn(
        tf.concat([rel_embs.word, rel_embs.char], axis=-1),
        filter_widths=self.rel_cnn.filter_widths, 
        filter_size=int(self.ffnn_size/len(self.rel_cnn.filter_widths)),
      )
    return rel_repls

  def batch_inference(self, text_emb, text_outputs, sentence_length):
    # To handle batched inputs.
    document_length = tf.reduce_sum(sentence_length, axis=-1)
    batch_size = shape(text_emb, 0)
    max_num_mentions = tf.to_int32(tf.floor(tf.to_float(tf.reduce_max(document_length)) * self.mention_ratio))

    def loop_func(idx, relations, mentions, losses):
      r, m, l = self.inference(
        text_emb[idx], text_outputs[idx], sentence_length[idx], 
        self.ph.query[idx], self.ph.mentions[idx], self.ph.num_mentions[idx],
        self.ph.target.subjective[idx], self.ph.target.objective[idx], 
        self.ph.loss_weights_by_label[idx], 
        max_num_mentions=max_num_mentions)
      idx = idx + 1
      relations = tf.concat([relations, tf.expand_dims(r, axis=0)], axis=0)
      mentions = tf.concat([mentions, tf.expand_dims(m, axis=0)], axis=0)
      losses = tf.concat([losses, tf.expand_dims(l, axis=0)], axis=0)
      return idx, relations, mentions, losses

    idx = tf.zeros((), dtype=tf.int32)
    cond = lambda idx, *args : idx < batch_size
    loop_vars = [
      idx, 
      tf.zeros((0, max_num_mentions, 2), dtype=tf.int32),
      tf.zeros((0, max_num_mentions, 2), dtype=tf.int32),
      tf.zeros((0), dtype=tf.float32),
    ]
    _, relations, mentions, losses = tf.while_loop(
      cond, loop_func, loop_vars,
      shape_invariants=[
        idx.get_shape(), 
        tf.TensorShape([None, None, 2]),
        tf.TensorShape([None, None, 2]),
        tf.TensorShape([None]),
      ],
      parallel_iterations=self.max_batch_size,
    )
    predictions = [relations, mentions]
    loss = tf.reduce_mean(losses, axis=-1)
    return predictions, loss

  def setup_placeholder(self):
    ph = recDotDefaultDict()
    ph.text.word = tf.placeholder(
      tf.int32, name='text.word',
      shape=[None, None, None]) if self.encoder.wbase else None # [batch_size, max_num_sent, max_num_word]
    ph.text.char = tf.placeholder(
      tf.int32, name='text.char',
      shape=[None, None, None, None]) if self.encoder.cbase else None # [batch_size, max_num_sent, max_num_word, max_num_char]
    
    ph.query = tf.placeholder(
      tf.int32, name='query', shape=[None, 2]) # [batch_size, 2]
    for k in ['subjective', 'objective']:
      ph.target[k] = tf.placeholder(
        tf.int32, name='target.%s' % k, shape=[None, None, self.max_mention_width]) # [max_sequence_len, max_mention_width] 

    ph.mentions = tf.placeholder(
      tf.int32, name='mentions', shape=[None, None, 2])  # [batch_size, max_num_mentions, 2]
    ph.num_mentions = tf.placeholder(
      tf.int32, name='num_mentions', shape=[None]) 
    ph.loss_weights_by_label = tf.placeholder(
      tf.float32, name='loss_weights_by_label', shape=[None, self.vocab.rel.size])
    return ph


  def inference(self, text_emb, text_outputs, sentence_length,
                query, gold_mentions, num_gold_mentions, 
                subj_targets, obj_targets, 
                loss_weights_by_label,
                max_num_mentions=None):
    '''
    Args:
    - text_emb:
    - text_outputs:
    - sentence_length:
    - query:
    - gold_mentions:
    - num_gold_mentions:
    - subj_targets, obj_targets:  [max_sequence_len, max_mention_width]
    - loss_weights_by_label: [num_relations]
    Return:
    - predicted_relations: [num_mentions, 2 (= subj/obj)]
    - predicted_mentions: [num_mentions, 2 (= start/end)]
    - losses: [num_mentions]
    - max_num_mentions: None or An integer tensor. If not None, the first dimentions of predicted_relations and predicted_mentions are padded up to this value for batching.
    
    '''
      # self.sentence_length = tf.count_nonzero(self.ph.text.word, axis=-1)
      # word_repls = encoder.word_encoder.word_encode(self.ph.text.word)
      # char_repls = encoder.word_encoder.char_encode(self.ph.text.char)
      # text_emb, text_outputs, state = encoder.encode([word_repls, char_repls], 
      #                                                self.sentence_length)

    if self.reuse:
      tf.get_variable_scope().reuse_variables()

    with tf.name_scope('flatten_text'):
      flattened_text_emb, flattened_text_outputs, flattened_sentence_indices = self.flatten_doc_to_sent(text_emb, text_outputs, sentence_length)

    with tf.name_scope('get_query_emb'):
      query_starts, query_ends = tf.unstack(tf.expand_dims(query, 0), axis=-1)
      query_emb = self.get_mention_emb(
        flattened_text_emb, flattened_text_outputs, query_starts, query_ends)

    with tf.name_scope('get_mentions'):
      _, _, _, pred_mention_starts, pred_mention_ends, pred_mention_scores, pred_mention_emb = self.get_mentions(flattened_text_emb, flattened_text_outputs, flattened_sentence_indices)

    # Concatenated [subjective, objective] relations with each mention.
    with tf.name_scope('calc_logits'):
      pred_subj_logits = self.predict_relation(query_emb, pred_mention_emb, 
                                               pred_mention_scores, True)
      pred_obj_logits = self.predict_relation(query_emb, pred_mention_emb, 
                                              pred_mention_scores, False)

    with tf.name_scope('predict_mention_and_relation'):
      predicted_relations = tf.concat([
        tf.expand_dims(tf.argmax(pred_subj_logits, axis=-1), -1),
        tf.zeros([shape(pred_subj_logits, 0), 1], dtype=tf.int64) # no prediction as for obj for now.
        #tf.expand_dims(tf.argmax(pred_obj_logits, axis=-1), -1)
       ], axis=-1) # [num_mentions, 2]
      predicted_relations = tf.cast(predicted_relations, tf.int32)
      predicted_mentions = tf.concat([
        tf.expand_dims(pred_mention_starts, -1),
        tf.expand_dims(pred_mention_ends, -1)
      ], axis=-1) # [num_mentions, 2]

      if max_num_mentions is not None:
        num_pads = max_num_mentions - shape(predicted_relations, 0)
        pad_shape = [[0, num_pads], [0, 0]]
        predicted_relations = tf.pad(predicted_relations, pad_shape)
        predicted_mentions = tf.pad(predicted_mentions, pad_shape)

    with tf.name_scope('merge_logits'):
      mention_starts = []
      mention_ends = []
      subj_logits = []
      obj_logits = []

      if self.use_predicted_mentions:
        mention_starts.append(pred_mention_starts)
        mention_ends.append(pred_mention_ends)
        subj_logits.append(pred_subj_logits)
        obj_logits.append(pred_obj_logits)

      if self.use_gold_mentions:
        gold_mentions = tf.slice(gold_mentions, [0, 0], [num_gold_mentions, 2])
        gold_mentions = tf.reshape(gold_mentions, [shape(gold_mentions, 0), 2])
        gold_mention_starts, gold_mention_ends = tf.unstack(gold_mentions, axis=-1)
        gold_mention_emb = self.get_mention_emb(
          flattened_text_emb, flattened_text_outputs, 
          gold_mention_starts, gold_mention_ends)
        gold_mention_scores = self.get_mention_scores(gold_mention_emb)
        gold_subj_logits = self.predict_relation(query_emb, gold_mention_emb, 
                                                 gold_mention_scores, True)
        gold_obj_logits = self.predict_relation(query_emb, gold_mention_emb, 
                                                gold_mention_scores, False)
        mention_starts.append(gold_mention_starts)
        mention_ends.append(gold_mention_ends)
        subj_logits.append(gold_subj_logits)
        obj_logits.append(gold_obj_logits)

      assert self.use_gold_mentions or self.use_predicted_mentions
      mention_starts = tf.concat(mention_starts, axis=0)
      mention_ends = tf.concat(mention_ends, axis=0)
      subj_logits = tf.concat(subj_logits, axis=0)
      obj_logits = tf.concat(obj_logits, axis=0)

    with tf.name_scope('loss'):
      mention_indices = tf.stack([mention_starts, mention_ends-mention_starts], 
                                 axis=-1) # [num_mentions, 2]

      # Gold mentions longer than self.max_mention_width should be cut.
      mention_indices = tf.clip_by_value(
        mention_indices, 0, shape(subj_targets, -1) - 1) # [num_mentions, 2]

      subj_targets = tf.gather_nd(subj_targets, mention_indices) # [num_mentions]
      obj_targets = tf.gather_nd(obj_targets, mention_indices) # [num_mentions]

      subj_loss_weights = tf.gather(loss_weights_by_label, subj_targets)
      obj_loss_weights = tf.gather(loss_weights_by_label, obj_targets)

      subj_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=subj_logits, labels=subj_targets)
      obj_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=obj_logits, labels=obj_targets)
      #losses = tf.concat([subj_losses, obj_losses], axis=-1) * tf.concat([subj_loss_weights, obj_loss_weights], axis=-1)
      losses = subj_losses * subj_loss_weights
      loss = tf.reduce_mean(losses, axis=-1)
    self.reuse = True
    return predicted_relations, predicted_mentions, loss

  def predict_relation(self, query_emb, mention_emb, mention_scores, 
                       is_query_subjective):
    '''
    Args:
    - query_emb: [emb]
    - mention_emb: [n_mentions, emb]
    - is_query_subjective: A boolean. If true, this function outputs a distribution of relation label probabilities for a triple (query, rel, mention) across rel, otherwise for (mention, rel, query)
    - reuse: A boolean. The variables of this network should be reused by both query-subjective and query-objective predictions by switching the orders of input representations.
    '''
    with tf.variable_scope('pair_emb'):
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
        w = self.rel_w
        b = tf.get_variable('biases', [self.vocab.rel.size - 1])
        x = pair_emb
        logits = tf.nn.xw_plus_b(x, w, b)
        no_relation = tf.zeros([shape(mention_scores, 0), 1], tf.float32)
        logits = tf.concat([no_relation, logits], axis=-1)

        # type A
        mention_unconfidence_penalty = tf.concat([
          no_relation,
          tf.tile(tf.expand_dims(mention_scores, 1), [1, self.vocab.rel.size-1])
        ], axis=-1)

        # type B
        # mention_unconfidence_penalty = tf.concat([
        #   -tf.expand_dims(mention_scores, 1),
        #   #tf.tile(tf.expand_dims(mention_scores, 1), [1, shape(logits, 1)-1])
        #   tf.zeros([shape(logits, 0), self.vocab.rel.size-1], dtype=tf.float32)
        # ], axis=-1)

    tf.get_variable_scope().reuse_variables()
    return logits + mention_unconfidence_penalty

  def test(self, batches, mode, logger, output_path=None):
    results = []
    used_batches = []
    for i, batch in enumerate(batches):
      input_feed = self.get_input_feed(batch, False)
      relations, mentions = self.sess.run(self.predictions, input_feed)
      try:
        used_batches += flatten_batch(batch)
      except Exception as e:
        pprint(batch)
        print(e)
        exit(1)
      for rel, mention in zip(relations.tolist(), mentions.tolist()):
        results.append((rel, mention))

    sys.stdout = open(output_path, 'w') if output_path else sys.stdout

    triples, mentions = dataset_class.formatize_and_print(
      used_batches, results, vocab=self.encoder.vocab)
    triple_precision, triple_recall, triple_f1 = dataset_class.evaluate_triples(triples)
    mention_precision, mention_recall, mention_f1 = dataset_class.evaluate_mentions(mentions)

    sys.stdout = sys.__stdout__
    if output_path:
      sys.stderr.write("Output the testing results to \'{}\' .\n".format(output_path))
    summary_dict = {}
    summary_dict['relex/%s/triple/f1' % mode] = triple_f1
    summary_dict['relex/%s/triple/precision' % mode] = triple_precision
    summary_dict['relex/%s/triple/recall' % mode] = triple_recall
    summary_dict['relex/%s/mention/f1' % mode] = mention_f1
    summary_dict['relex/%s/mention/precision' % mode] = mention_precision
    summary_dict['relex/%s/mention/recall' % mode] = mention_recall
    summary = make_summary(summary_dict)
    return triple_f1, summary


  def get_input_feed(self, batch, is_training):
    input_feed = {}
    input_feed[self.is_training] = is_training

    # todo: 今はバッチ処理出来ないので強制的にbatch_size == 1
    # assert batch.text.word.shape[0] == 1
    # assert batch.text.char.shape[0] == 1

    input_feed[self.ph.text.word] = batch.text.word
    input_feed[self.ph.text.char] = batch.text.char

    input_feed[self.ph.query] = batch.query.flat_position
    for k in ['subjective', 'objective']:
      input_feed[self.ph.target[k]] = batch.target[k]
      #input_feed[self.ph.loss_weights_by_label[k]] = batch.loss_weights_by_label[k]
    input_feed[self.ph.mentions] = batch.mentions.flat_position
    input_feed[self.ph.num_mentions] = batch.num_mentions
    input_feed[self.ph.loss_weights_by_label] = batch.loss_weights_by_label
    return input_feed

