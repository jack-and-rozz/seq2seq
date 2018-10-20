# coding: utf-8 
import operator
import math, time, sys, random, re, math, os
import numpy as np
import tensorflow as tf
import pandas as pd
from pprint import pprint
from collections import OrderedDict, defaultdict
from copy import deepcopy

from core.utils.tf_utils import shape, ffnn, make_summary, initialize_embeddings
from core.utils.common import RED, BLUE, GREEN, YELLOW, MAGENTA, CYAN, WHITE, BOLD, BLACK, UNDERLINE, RESET
from core.utils.common import dbgprint, flatten, dotDict, recDotDefaultDict, flatten_recdict
from core.models.base import ModelBase
from core.models.wikiP2D.coref import coref_ops, conll, metrics
from core.models.wikiP2D.coref import util as coref_util
from core.models.wikiP2D.coref.analysis import print_results
from core.vocabulary.base import _UNK, PAD_ID

from core.models.wikiP2D.desc.desc import DescModelBase

class CorefModelBase(ModelBase):
  pass

class CoreferenceResolution(CorefModelBase):
  def __init__(self, sess, config, manager, activation=tf.nn.relu):
    super().__init__(sess, config)
    self.sess = sess
    self.config = config

    self.scorer_path = config.scorer_path
    self.feature_size = config.f_embedding_size
    self.max_training_sentences = config.max_training_sentences
    self.mention_ratio = config.mention_ratio
    self.max_antecedents = config.max_antecedents
    self.ffnn_depth = config.ffnn_depth
    self.ffnn_size = config.ffnn_size

    self.max_mention_width = config.max_mention_width
    self.use_speaker_feature = config.use_speaker_feature
    self.use_genre_feature = config.use_genre_feature
    self.use_width_feature = config.use_width_feature
    self.use_distance_feature = config.use_distance_feature

    # Encoder
    self.vocab = manager.vocab
    shared_layers = manager.restore_shared_layers()
    self.is_training = shared_layers.is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.encoder = self.setup_encoder(shared_layers.encoder, 
                                      config.use_local_rnn)
    # Placeholders
    self.ph = self.setup_placeholders()

    # Embeddings
    with tf.variable_scope('Embeddings'):
      self.same_speaker_emb = initialize_embeddings(
        'same_speaker', [2, self.feature_size]) # True or False.
      self.genre_emb = initialize_embeddings('genre', [self.vocab.genre.size, self.feature_size])
      self.mention_width_emb = initialize_embeddings("mention_width", [self.max_mention_width, self.feature_size])
      self.mention_distance_emb = initialize_embeddings("mention_distance", [10, self.feature_size])

    word_repls = self.encoder.word_encoder.word_encode(self.ph.text.word)
    char_repls = self.encoder.word_encoder.char_encode(self.ph.text.char)
    text_emb, text_outputs, state = self.encoder.encode(
      [word_repls, char_repls], self.ph.sentence_length) 

    with tf.name_scope('candidates_and_mentions'):
      flattened_text_emb, flattened_text_outputs, flattened_sentence_indices = self.flatten_doc_to_sent(text_emb, text_outputs, self.ph.sentence_length)

      candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, mention_scores, mention_emb = self.get_mentions(
        flattened_text_emb, flattened_text_outputs, flattened_sentence_indices)

      with tf.name_scope('keep_mention_embs'):
        self.pred_mention_emb = self.get_mention_emb(
          flattened_text_emb, flattened_text_outputs, 
          mention_starts, 
          mention_ends, 
          False)

        # Add dummy to gold mentions for the case that an example has no gold mentions and self.get_mention_emb(...) causes an error.
        dummy = tf.constant([0], dtype=tf.int32)
        self.gold_mention_emb = self.get_mention_emb(
          flattened_text_emb, flattened_text_outputs, 
          # self.ph.gold_starts, 
          # self.ph.gold_ends, 
          tf.concat([self.ph.gold_starts, dummy], axis=0),
          tf.concat([self.ph.gold_ends, dummy], axis=0),
          False)

    with tf.name_scope('antecedents'):
      antecedents, antecedent_scores, antecedent_labels = self.get_antecedents(
        mention_scores, mention_starts, mention_ends, mention_emb,
        self.ph.speaker_ids, self.ph.genre, 
        self.ph.gold_starts, self.ph.gold_ends, self.ph.cluster_ids)

    self.outputs = [candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores]

    with tf.name_scope('loss'):
      loss = self.softmax_loss(antecedent_scores, antecedent_labels) # [num_mentions]
      self.loss = tf.reduce_sum(loss) # []

    # for adversarial MTL, it has to be a rank 2 Tensor [batch_size, emb_size]
    with tf.name_scope('AdversarialInputs'):
      self.adv_inputs = tf.reduce_mean(text_outputs, axis=1)

  def setup_placeholders(self):
    with tf.name_scope('Placeholder'):
      ph = recDotDefaultDict()
      # Input document
      ph.text.word = tf.placeholder(
        tf.int32, name='text.word',
        shape=[None, None]) if self.encoder.wbase else None
      ph.text.char = tf.placeholder(
        tf.int32, name='text.char',
        shape=[None, None, None]) if self.encoder.cbase else None

      # TODO: truncate_exampleしたものをw_sentencesにfeedした時、何故かtruncate前の単語数を数えてしまう。とりあえずsentence_lengthを直接feed.

      #self.sentence_length = tf.count_nonzero(ph.text.word, axis=1, dtype=tf.int32)
      ph.sentence_length = tf.placeholder(tf.int32, shape=[None])

      # Clusters 
      ph.gold_starts = tf.placeholder(tf.int32, name='gold_starts', shape=[None])
      ph.gold_ends = tf.placeholder(tf.int32, name='gold_ends', shape=[None])
      ph.cluster_ids = tf.placeholder(tf.int32, name='cluster_ids', shape=[None])
      # Metadata
      ph.speaker_ids = tf.placeholder(tf.int32, shape=[None], 
                                        name="speaker_ids")
      ph.genre = tf.placeholder(tf.int32, shape=[], name="genre")
    return ph

  def define_combination(self, other_models):
    # Combined tests with coref task.
    desc_model = [x for x in other_models
                  if isinstance(x, DescModelBase)]
    return False # DEBUG
    if not desc_model:
      return 

    decoder = desc_model[0].decoder
    pred_mention_desc = decoder.decode_test(self.pred_mention_emb)
    gold_mention_desc = decoder.decode_test(self.gold_mention_emb)
    self.outputs += [pred_mention_desc, gold_mention_desc]

  def flatten_doc_to_sent(self, text_emb, text_outputs, text_len):
    '''
    Flatten 3-ranked tensor [num_sentences, max_num_words, emb] to 2-ranked tensor[num_words, emb]. 
    The current version of this function (and whole coref module) is unable to handle a batched input.
    '''
    with tf.name_scope('ReshapeEncoderOutputs'):
      num_sentences = tf.shape(text_emb)[0]
      max_sentence_length = tf.shape(text_emb)[1]
      text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
      text_len_mask = tf.reshape(text_len_mask, [num_sentences * max_sentence_length])
      sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]

      flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
      flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask) # [num_words, dim(word_emb + encoded_char_emb)]
      flattened_text_outputs = self.flatten_emb_by_sentence(text_outputs, text_len_mask) # [num_words, dim(encoder_output) ]
    return flattened_text_emb, flattened_text_outputs, flattened_sentence_indices

  def get_mentions(self, flattened_text_emb, flattened_text_outputs, 
                   flattened_sentence_indices):

    with tf.name_scope('SpanCandidates'):
      candidate_starts, candidate_ends = coref_ops.spans(
        sentence_indices=flattened_sentence_indices,
        max_width=self.max_mention_width)
      candidate_starts.set_shape([None])
      candidate_ends.set_shape([None])

    with tf.name_scope('Mentions'):
      candidate_mention_emb = self.get_mention_emb(flattened_text_emb, flattened_text_outputs, candidate_starts, candidate_ends, self.use_width_feature) # [num_candidates, emb]

      candidate_mention_scores =  self.get_mention_scores(candidate_mention_emb) # [num_mentions, 1]
      #candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [num_mentions]

      k = tf.to_int32(tf.floor(tf.to_float(tf.shape(flattened_text_outputs)[0]) * self.mention_ratio))
      predicted_mention_indices = coref_ops.extract_mentions(candidate_mention_scores, candidate_starts, candidate_ends, k) # ([k], [k])
      predicted_mention_indices.set_shape([None])

      mention_starts = tf.gather(candidate_starts, predicted_mention_indices) # [num_mentions]
      mention_ends = tf.gather(candidate_ends, predicted_mention_indices) # [num_mentions]
      mention_emb = tf.gather(candidate_mention_emb, predicted_mention_indices) # [num_mentions, emb]
      mention_scores = tf.gather(candidate_mention_scores, predicted_mention_indices) # [num_mentions]

    return candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, mention_scores, mention_emb

  def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends,
                      use_width_feature):
    '''
    text_emb: [num_words, dim(word_emb + char_emb)]
    text_outputs: [num_words, dim(encoder_outputs)]
    mention_starts, mention_ends: [num_mentions]
    '''
    mention_emb, head_scores = self.encoder.get_mention_emb(text_emb, text_outputs, mention_starts, mention_ends)

    # Concatenate mention_width_feature (not shared).
    if use_width_feature:
      with tf.name_scope('mention_width'):
        mention_width = 1 + mention_ends - mention_starts # [num_mentions]
        mention_width_index = mention_width - 1  #[num_mentions]
        mention_width_index = tf.clip_by_value(
          mention_width_index, 0, self.max_mention_width - 1)
        mention_width_emb = tf.gather(self.mention_width_emb, mention_width_index) # [num_mentions, emb]
        mention_width_emb = tf.nn.dropout(mention_width_emb, self.keep_prob)
      mention_emb = tf.concat([mention_emb, mention_width_emb], axis=-1)

    return mention_emb


  def get_mention_scores(self, mention_emb):
    with tf.variable_scope("mention_scores"):
      scores = ffnn(mention_emb, self.ffnn_depth, self.ffnn_size, 1, self.keep_prob) # [num_mentions, 1]
      return tf.reshape(scores, [shape(scores, 0)])

  def get_antecedents(self, mention_scores, mention_starts, mention_ends, 
                      mention_emb, speaker_ids, genre, 
                      gold_starts, gold_ends, cluster_ids):
    genre_emb = tf.nn.embedding_lookup(self.genre_emb, genre)
    mention_speaker_ids = tf.gather(speaker_ids, mention_starts) # [num_mentions]

    with tf.name_scope('Antecedents'):
      antecedents, antecedent_labels, antecedents_len = coref_ops.antecedents(mention_starts, mention_ends, gold_starts, gold_ends, cluster_ids, self.max_antecedents) # ([num_mentions, max_ant], [num_mentions, max_ant + 1], [num_mentions]
      antecedents.set_shape([None, None])
      antecedent_labels.set_shape([None, None])
      antecedents_len.set_shape([None])

      antecedent_scores = self.get_antecedent_scores(
        mention_emb, mention_scores, antecedents, antecedents_len, 
        mention_starts, mention_ends, mention_speaker_ids, genre_emb) # [num_mentions, max_ant + 1]

    return antecedents, antecedent_scores, antecedent_labels

  def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts, mention_ends, mention_speaker_ids, genre_emb):
    num_mentions = shape(mention_emb, 0)
    max_antecedents = shape(antecedents, 1)

    feature_emb_list = []

    if self.use_speaker_feature:
      antecedent_speaker_ids = tf.gather(mention_speaker_ids, antecedents) # [num_mentions, max_ant]
      same_speaker = tf.equal(tf.expand_dims(mention_speaker_ids, 1), antecedent_speaker_ids) # [num_mentions, max_ant]
      speaker_pair_emb = tf.gather(self.same_speaker_emb, tf.to_int32(same_speaker)) # [num_mentions, max_ant, emb]
      feature_emb_list.append(speaker_pair_emb)

    if self.use_genre_feature:
      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [num_mentions, max_antecedents, 1]) # [num_mentions, max_ant, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.use_distance_feature:
      target_indices = tf.range(num_mentions) # [num_mentions]
      mention_distance = tf.expand_dims(target_indices, 1) - antecedents # [num_mentions, max_ant]
      mention_distance_bins = coref_ops.distance_bins(mention_distance) # [num_mentions, max_ant]
      mention_distance_bins.set_shape([None, None])
      mention_distance_emb = tf.gather(self.mention_distance_emb, mention_distance_bins) # [num_mentions, max_ant]
      feature_emb_list.append(mention_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [num_mentions, max_ant, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.keep_prob) # [num_mentions, max_ant, emb]

    antecedent_emb = tf.gather(mention_emb, antecedents) # [num_mentions, max_ant, emb]
    target_emb_tiled = tf.tile(tf.expand_dims(mention_emb, 1), [1, max_antecedents, 1]) # [num_mentions, max_ant, emb]
    similarity_emb = antecedent_emb * target_emb_tiled # [num_mentions, max_ant, emb]
    pair_emb = tf.concat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb], 2) # [num_mentions, max_ant, emb]

    with tf.variable_scope("iteration"):
      with tf.variable_scope("antecedent_scoring"):
        antecedent_scores = ffnn(pair_emb, self.ffnn_depth, self.ffnn_size, 1, self.keep_prob) # [num_mentions, max_ant, 1]
    antecedent_scores = tf.squeeze(antecedent_scores, 2) # [num_mentions, max_ant]

    antecedent_mask = tf.log(tf.sequence_mask(
      antecedents_len, max_antecedents, 
      dtype=tf.float32)) # [num_mentions, max_ant]
    antecedent_scores += antecedent_mask # [num_mentions, max_ant]

    antecedent_scores += tf.expand_dims(mention_scores, 1) + tf.gather(mention_scores, antecedents) # [num_mentions, max_ant]

    no_antecedent = tf.zeros([shape(mention_scores, 0), 1]) # [num_mentions, 1]
    antecedent_scores = tf.concat([no_antecedent, antecedent_scores], 1) # [num_mentions, max_ant + 1]
    return antecedent_scores  # [num_mentions, max_ant + 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [num_mentions, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [num_mentions]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [num_mentions]
    return log_norm - marginalized_gold_scores # [num_mentions]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, text_len_mask) # remove masked elements 

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = list(zip(*mentions))
    else:
      starts, ends = [], []
      #return np.array(starts), np.array(ends)
    return starts, ends

  def get_input_feed(self, batch, is_training):
    input_feed = {}

    ## Texts
    if self.encoder.cbase:
      input_feed[self.ph.text.char] = batch.text.char
    if self.encoder.wbase:
      input_feed[self.ph.text.word] = batch.text.word
    input_feed[self.ph.sentence_length] = batch.sentence_length

    ## Mention spans and their clusters
    gold_mentions = sorted(tuple(m) for m in flatten(batch.clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(batch.clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id
    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    input_feed[self.ph.gold_starts] = np.array(gold_starts)
    input_feed[self.ph.gold_ends] = np.array(gold_ends)
    input_feed[self.ph.cluster_ids] = np.array(cluster_ids)

    ## Metadata
    input_feed[self.is_training] = is_training
    input_feed[self.ph.speaker_ids] = batch.speakers
    input_feed[self.ph.genre] = batch.genre

    if is_training and batch.text.word.shape[0] > self.max_training_sentences:
      return self.truncate_example(input_feed)
    else:
      return input_feed

  def truncate_example(self, input_feed):
    max_training_sentences = self.max_training_sentences
    num_sentences = input_feed[self.ph.text.word].shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    sentence_length = input_feed[self.ph.sentence_length]
    #sentence_length = np.array([len([w for w in sent if w != PAD_ID]) for sent in input_feed[self.ph.text.word]]) # Number of words except PAD in each sentence.

    word_offset = sentence_length[:sentence_offset].sum() # The sum of the number of truncated words except PAD.
    num_words = sentence_length[sentence_offset:sentence_offset + max_training_sentences].sum()

    w_sentences = input_feed[self.ph.text.word][sentence_offset:sentence_offset + max_training_sentences,:]
    c_sentences = input_feed[self.ph.text.char][sentence_offset:sentence_offset + max_training_sentences,:,:]
    sentence_length = sentence_length[sentence_offset:sentence_offset + max_training_sentences]
    speaker_ids = input_feed[self.ph.speaker_ids][word_offset: word_offset + num_words]
    gold_spans = np.logical_and(input_feed[self.ph.gold_ends] >= word_offset, input_feed[self.ph.gold_starts] < word_offset + num_words)
    gold_starts = input_feed[self.ph.gold_starts][gold_spans] - word_offset
    gold_ends = input_feed[self.ph.gold_ends][gold_spans] - word_offset
    cluster_ids = input_feed[self.ph.cluster_ids][gold_spans]


    input_feed[self.ph.text.word] = w_sentences
    input_feed[self.ph.text.char] = c_sentences
    input_feed[self.ph.sentence_length] = sentence_length
    input_feed[self.ph.speaker_ids] = speaker_ids
    input_feed[self.ph.gold_starts] = gold_starts
    input_feed[self.ph.gold_ends] = gold_ends
    input_feed[self.ph.cluster_ids] = cluster_ids
    # DEBUG
    #print ('length of trun w_sentences', len(input_feed[self.ph.text.word]), sum([len([w for w in s if w != PAD_ID]) for s in input_feed[self.ph.text.word]]))
    return input_feed

  ##############################################
  ##           Evaluation
  ##############################################

  def test(self, batches, mode, logger, output_path):
    conll_eval_path = os.path.join(
      self.config.dataset.gold_dir, 
      self.config.dataset['%s_gold' % mode])

    with open(output_path + '.stat', 'w') as f:
      sys.stdout = f
      eval_summary, f1, results = self.evaluate(
        batches, conll_eval_path, mode)
      sys.stdout = sys.__stdout__

    with open(output_path + '.detail', 'w') as f:
      sys.stdout = f
      df = print_results(results, self.vocab.encoder, False)
      sys.stdout = sys.__stdout__

    with open(output_path + '.detail.with_desc', 'w') as f:
      sys.stdout = f
      df = print_results(results, self.vocab.encoder, True)
      sys.stdout = sys.__stdout__

    sys.stderr.write("Output the predicted and gold clusters to \'{}\' .\n".format(output_path))

    return f1, eval_summary

  def evaluate(self, batches, gold_path, mode, official_stdout=False):
    def _k_to_tag(k):
      if k == -3:
        return "oracle" # use only gold spans.
      elif k == -2:
        return "actual" # use mention_spans as a result of pruning candidate_spans.
      elif k == -1:
        return "exact" # use the same number of candidate_spans as the gold_spans.
      elif k == 0:
        return "threshold" # use only candidate_spans with a score greater than 0.
      else:
        return "{}%".format(k)
    #mention_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 0, 10, 15, 20, 25, 30, 40, 50] }
    mention_evaluators = { k:coref_util.RetrievalEvaluator() for k in [-3, -2, -1, 0] }

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    results = OrderedDict()

    for example_num, example in enumerate(batches):
      input_feed = self.get_input_feed(example, False)
      gold_starts = input_feed[self.ph.gold_starts]
      gold_ends = input_feed[self.ph.gold_ends]

      ######
      # debug
      # flattened_text_emb, mention_starts, mention_ends, gold_starts, gold_ends = self.sess.run(self.debug_ops, input_feed)
      # dbgprint(str(example_num) + ':')
      # print('text_shape', flattened_text_emb.shape)
      # print('pred_mentions', np.concatenate([np.expand_dims(mention_starts, -1), 
      #                                        np.expand_dims(mention_ends, -1)], 
      #                                       axis=-1))
      # print('gold_mentions', np.concatenate([np.expand_dims(gold_starts, -1), 
      #                                        np.expand_dims(gold_ends, -1)], 
      #                                       axis=-1))
      # print()
      ######

      outputs = self.sess.run(self.outputs, input_feed)
      candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores = outputs[:7]

      self.evaluate_mentions(candidate_starts, candidate_ends, mention_starts, mention_ends, candidate_mention_scores, gold_starts, gold_ends, example, mention_evaluators)
      predicted_antecedents = self.get_predicted_antecedents(antecedents, antecedent_scores)
      coref_predictions[example.doc_key] = self.evaluate_coref(mention_starts, mention_ends, predicted_antecedents, example.clusters, coref_evaluator)

      results[example.doc_key] = dotDict({
        'raw_text': example.text.raw,
        'speakers': example.speakers,
        'extracted_mentions': [(begin, end) for begin, end in zip(mention_starts, mention_ends)],
        'predicted_antecedents': predicted_antecedents
      })
      if len(outputs) > 7:
        mention_descs = {}
        
        pred_mention_desc = [self.vocab.decoder.word.ids2tokens(s) for s in outputs[7][:, 0, :]]
        gold_mention_desc = [self.vocab.decoder.word.ids2tokens(s) for s in outputs[8][:, 0, :]]
        for s, e, desc in zip(mention_starts, mention_ends, pred_mention_desc):
          mention_descs[(s, e)] = desc
        for s, e, desc in zip(gold_starts, gold_ends, pred_mention_desc):
          mention_descs[(s, e)] = desc

        results[example.doc_key].mention_descs = mention_descs
      else:
        results[example.doc_key].mention_descs = []
    summary_dict = {}

    for k, evaluator in sorted(list(mention_evaluators.items()), key=operator.itemgetter(0)):
      tags = ["mention/{} @ {}".format(t, _k_to_tag(k)) for t in ("R", "P", "F")]
      results_to_print = []
      for t, v in zip(tags, evaluator.metrics()):
        results_to_print.append("{:<10}: {:.2f}".format(t, v))
        summary_dict["coref/%s/" % mode + t] = v
      print(", ".join(results_to_print))

    conll_results = conll.evaluate_conll(gold_path, coref_predictions, self.scorer_path, official_stdout)
    val_types = ('p', 'r', 'f')
    for metric in conll_results:
      for val_type in val_types:
        summary_dict["coref/%s/%s/%s" % (mode, metric, val_type)] = conll_results[metric][val_type]
      print ("%s (%s) : %s" % (
        metric, 
        ", ".join(val_types), 
        " ".join(["%.2f" % x for x in conll_results[metric].values()])
      ))

    average_f1 = sum(conll_res["f"] for conll_res in list(conll_results.values())) / len(conll_results)
    summary_dict["coref/%s/Average F1 (conll)" % mode] = average_f1
    print("Average F1 (conll): {:.2f}%".format(average_f1))

    p,r,f = coref_evaluator.get_prf()
    summary_dict["coref/%s/Average F1 (py)" % mode] = f
    print("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["coref/%s/Average precision (py)" % mode] = p
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["coref/%s/Average recall (py)" % mode] = r
    print("Average recall (py): {:.2f}%".format(r * 100))

    aligned_results = coref_evaluator.get_aligned_results()
    for doc_key, aligned in zip(results, aligned_results):
      results[doc_key]['aligned_results'] = aligned

    average_f1 = sum([values['f'] for metric, values in conll_results.items()]) / len(conll_results)
    return make_summary(summary_dict), average_f1, results
    #return util.make_summary(summary_dict), average_f1, results

  def evaluate_mentions(self, candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores, gold_starts, gold_ends, example, evaluators):
    #text_length = sum(len(s) for s in example["w_sentences"])
    text_length = sum(len(s) for s in example.text.word)
    gold_spans = set(zip(gold_starts, gold_ends))

    if len(candidate_starts) > 0:
      sorted_starts, sorted_ends, _ = list(zip(*sorted(zip(candidate_starts, candidate_ends, mention_scores), key=operator.itemgetter(2), reverse=True)))
    else:
      sorted_starts = []
      sorted_ends = []

    for k, evaluator in list(evaluators.items()):
      if k == -3: #oracle
        predicted_spans = set(zip(candidate_starts, candidate_ends)) & gold_spans
      else:
        if k == -2: # actual
          predicted_starts = mention_starts
          predicted_ends = mention_ends
        elif k == 0: # threshold
          is_predicted = mention_scores > 0
          predicted_starts = candidate_starts[is_predicted]
          predicted_ends = candidate_ends[is_predicted]
        else:
          if k == -1: #exact
            num_predictions = len(gold_spans)
          else:
            num_predictions = math.floor((k * text_length) / 100)
          predicted_starts = sorted_starts[:num_predictions]
          predicted_ends = sorted_ends[:num_predictions]
        predicted_spans = set(zip(predicted_starts, predicted_ends))
      evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)

  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, mention_starts, mention_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(mention_starts[predicted_index]), int(mention_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        # Assign a new cluster-id if 'predicted_antecedent' belongs to none of the existing clusters.
        predicted_cluster = len(predicted_clusters) 
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(mention_starts[i]), int(mention_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster
    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in list(mention_to_predicted.items())}
    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, mention_starts, mention_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc
    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters


