# coding: utf-8 
import operator
import math, time, sys
import tensorflow as tf
from core.utils import common, tf_utils
from core.models.base import ModelBase
from core.models.wikiP2D.coref import coref_ops, conll, metrics
import numpy as np
from pprint import pprint

class CoreferenceResolution(ModelBase):
  def __init__(self, sess, config, is_training, encoder, 
               speaker_vocab, genre_vocab,
               activation=tf.nn.tanh):
    self.name = 'coref'
    self.dataset = 'coref'
    self.sess = sess
    self.encoder = encoder
    self.activation = activation

    self.is_training = is_training
    self.in_keep_prob = config.in_keep_prob if is_training else 1.0
    self.out_keep_prob = config.out_keep_prob if is_training else 1.0
    self.hidden_size = config.hidden_size
    self.feature_size = config.f_embedding_size
    self.max_mention_width = config.max_mention_width
    self.mention_ratio = config.mention_ratio
    self.max_antecedents = config.max_antecedents
    self.use_features = config.use_features
    self.use_metadata = config.use_metadata
    self.model_heads = config.model_heads
    self.ffnn_depth = config.num_layers
    self.ffnn_size = config.hidden_size

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.w_sentences = tf.placeholder(tf.int32, name='w_sentences',
                                        shape=[None, None])
      self.c_sentences = tf.placeholder(tf.int32, name='c_sentences',
                                        shape=[None, None, None])
      self.sentence_length = tf.placeholder(tf.int32, shape=[None], 
                                            name="sentence_length")
      self.speaker_ids = tf.placeholder(tf.int32, shape=[None], 
                                        name="speaker_ids")
      self.genre = tf.placeholder(tf.int32, shape=[], name="genre")
      self.gold_starts = tf.placeholder(tf.int32, name='gold_starts', shape=[None])
      self.gold_ends = tf.placeholder(tf.int32, name='gold_ends', shape=[None])
      self.cluster_ids = tf.placeholder(tf.int32, name='cluster_ids', shape=[None])

    # Embeddings
    with tf.variable_scope('Embeddings'):
      self.same_speaker_emb = self.initialize_embeddings(
        'same_speaker', [2, self.feature_size]) # True or False.
      self.genre_emb = self.initialize_embeddings(
        'genre', [genre_vocab.size, self.feature_size])
      self.mention_width_emb = self.initialize_embeddings(
        "mention_width", [self.max_mention_width, self.feature_size])
      self.mention_distance_emb = tf.get_variable("mention_distance", [10, self.feature_size])
    outputs, state = encoder.encode([self.w_sentences, self.c_sentences], 
                                    self.sentence_length)
    self.predictions, self.loss = self.get_predictions_and_loss(outputs, self.sentence_length, self.speaker_ids, self.genre, self.gold_starts, self.gold_ends, self.cluster_ids)

    self.outputs = [self.predictions]

    with tf.name_scope("Summary"):
      self.summary_loss = tf.placeholder(tf.float32, shape=[],
                                         name='coref_loss')


  def get_predictions_and_loss(self, text_emb, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids):
    with tf.name_scope('ReshapeEncoderOutputs'):
      num_sentences = tf.shape(text_emb)[0]
      max_sentence_length = tf.shape(text_emb)[1]

      text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
      text_len_mask = tf.reshape(text_len_mask, [num_sentences * max_sentence_length])
      genre_emb = tf.nn.embedding_lookup(self.genre_emb, genre)
      sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]

      flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
      flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask) # [num_words]

    with tf.name_scope('SpanCandidates'):
      candidate_starts, candidate_ends = coref_ops.spans(
        sentence_indices=flattened_sentence_indices,
        max_width=self.max_mention_width)
      candidate_starts.set_shape([None])
      candidate_ends.set_shape([None])

    with tf.name_scope('Mentions'):
      text_outputs = flattened_text_emb
      candidate_mention_emb = self.get_mention_emb(flattened_text_emb, text_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]

      
      candidate_mention_scores =  self.get_mention_scores(candidate_mention_emb) # [num_mentions, 1]
      candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [num_mentions]

      k = tf.to_int32(tf.floor(tf.to_float(tf.shape(text_outputs)[0]) * self.mention_ratio))
      predicted_mention_indices = coref_ops.extract_mentions(candidate_mention_scores, candidate_starts, candidate_ends, k) # ([k], [k])
      predicted_mention_indices.set_shape([None])

      mention_starts = tf.gather(candidate_starts, predicted_mention_indices) # [num_mentions]
      mention_ends = tf.gather(candidate_ends, predicted_mention_indices) # [num_mentions]
      mention_emb = tf.gather(candidate_mention_emb, predicted_mention_indices) # [num_mentions, emb]
      mention_scores = tf.gather(candidate_mention_scores, predicted_mention_indices) # [num_mentions]

    #mention_start_emb = tf.gather(text_outputs, mention_starts) # [num_mentions, emb]
    #mention_end_emb = tf.gather(text_outputs, mention_ends) # [num_mentions, emb]
      mention_speaker_ids = tf.gather(speaker_ids, mention_starts) # [num_mentions]

    with tf.name_scope('Antecedents'):
      max_antecedents = self.max_antecedents
      antecedents, antecedent_labels, antecedents_len = coref_ops.antecedents(mention_starts, mention_ends, gold_starts, gold_ends, cluster_ids, max_antecedents) # ([num_mentions, max_ant], [num_mentions, max_ant + 1], [num_mentions]
      antecedents.set_shape([None, None])
      antecedent_labels.set_shape([None, None])
      antecedents_len.set_shape([None])
      """
      antecedent_labels : i番目のスパンから見た前max_ant個のスパンが自身の先行詞かどうか(goldのcluster_idとの完全マッチでのみ判断しているのでtest時は全てfalse)
      """
      antecedent_scores = self.get_antecedent_scores(mention_emb, mention_scores, antecedents, antecedents_len, mention_starts, mention_ends, mention_speaker_ids, genre_emb) # [num_mentions, max_ant + 1]
      loss = self.softmax_loss(antecedent_scores, antecedent_labels) # [num_mentions]
      loss = tf.reduce_sum(loss) # []

    return [candidate_starts, candidate_ends, candidate_mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores], loss

  def get_mention_emb(self, text_emb, text_outputs, mention_starts, mention_ends):
    with tf.variable_scope('get_mention_emb'):
      mention_emb_list = []
      mention_start_emb = tf.gather(text_outputs, mention_starts) #[num_mentions, emb]
      mention_emb_list.append(mention_start_emb)

      mention_end_emb = tf.gather(text_outputs, mention_ends) #[num_mentions, emb]
      mention_emb_list.append(mention_end_emb)
      mention_width = 1 + mention_ends - mention_starts # [num_mentions]
      if self.use_features:
        mention_width_index = mention_width - 1  #[num_mentions]
        mention_width_emb = tf.gather(self.mention_width_emb, mention_width_index) # [num_mentions, emb]
        mention_width_emb = tf.nn.dropout(mention_width_emb, self.out_keep_prob)
        mention_emb_list.append(mention_width_emb)

      if self.model_heads:
        mention_indices = tf.expand_dims(tf.range(self.max_mention_width), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
        mention_indices = tf.minimum(tf_utils.shape(text_outputs, 0) - 1, mention_indices) # [num_mentions, max_mention_width]
        mention_text_emb = tf.gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]
        self.head_scores = tf_utils.projection(text_outputs, 1) # [num_words, 1]
        mention_head_scores = tf.gather(self.head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
        mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, self.max_mention_width, dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]
        mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), dim=1) # [num_mentions, max_mention_width, 1]
        mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
        mention_emb_list.append(mention_head_emb)

        mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]
    return mention_emb

  def get_mention_scores(self, mention_emb):
    with tf.variable_scope("mention_scores"):
      return tf_utils.ffnn(mention_emb, self.ffnn_depth, self.ffnn_size, 1, self.out_keep_prob) # [num_mentions, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [num_mentions, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [num_mentions]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [num_mentions]
    return log_norm - marginalized_gold_scores # [num_mentions]

  def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts, mention_ends, mention_speaker_ids, genre_emb):
    num_mentions = tf_utils.shape(mention_emb, 0)
    max_antecedents = tf_utils.shape(antecedents, 1)

    feature_emb_list = []

    if self.use_metadata:
      antecedent_speaker_ids = tf.gather(mention_speaker_ids, antecedents) # [num_mentions, max_ant]
      same_speaker = tf.equal(tf.expand_dims(mention_speaker_ids, 1), antecedent_speaker_ids) # [num_mentions, max_ant]
      #speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.feature_size]), tf.to_int32(same_speaker)) # [num_mentions, max_ant, emb]
      speaker_pair_emb = tf.gather(self.same_speaker_emb, tf.to_int32(same_speaker)) # [num_mentions, max_ant, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [num_mentions, max_antecedents, 1]) # [num_mentions, max_ant, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.use_features:
      target_indices = tf.range(num_mentions) # [num_mentions]
      mention_distance = tf.expand_dims(target_indices, 1) - antecedents # [num_mentions, max_ant]
      mention_distance_bins = coref_ops.distance_bins(mention_distance) # [num_mentions, max_ant]
      mention_distance_bins.set_shape([None, None])
      mention_distance_emb = tf.gather(self.mention_distance_emb, mention_distance_bins) # [num_mentions, max_ant]
      feature_emb_list.append(mention_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [num_mentions, max_ant, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.out_keep_prob) # [num_mentions, max_ant, emb]

    antecedent_emb = tf.gather(mention_emb, antecedents) # [num_mentions, max_ant, emb]
    target_emb_tiled = tf.tile(tf.expand_dims(mention_emb, 1), [1, max_antecedents, 1]) # [num_mentions, max_ant, emb]
    similarity_emb = antecedent_emb * target_emb_tiled # [num_mentions, max_ant, emb]

    pair_emb = tf.concat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb], 2) # [num_mentions, max_ant, emb]

    with tf.variable_scope("iteration"):
      with tf.variable_scope("antecedent_scoring"):
        antecedent_scores = tf_utils.ffnn(pair_emb, self.ffnn_depth, self.ffnn_size, 1, self.out_keep_prob) # [num_mentions, max_ant, 1]
    antecedent_scores = tf.squeeze(antecedent_scores, 2) # [num_mentions, max_ant]

    antecedent_mask = tf.log(tf.sequence_mask(antecedents_len, max_antecedents, dtype=tf.float32)) # [num_mentions, max_ant]
    antecedent_scores += antecedent_mask # [num_mentions, max_ant]

    antecedent_scores += tf.expand_dims(mention_scores, 1) + tf.gather(mention_scores, antecedents) # [num_mentions, max_ant]
    antecedent_scores = tf.concat([tf.zeros([tf_utils.shape(mention_scores, 0), 1]), antecedent_scores], 1) # [num_mentions, max_ant + 1]
    return antecedent_scores  # [num_mentions, max_ant + 1]


  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, tf_utils.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, text_len_mask)

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
      #return np.array(starts), np.array(ends)
    return starts, ends

  def get_input_feed(self, batch):
    input_feed = {}
    clusters = batch["clusters"]
    c_sentences = batch['c_sentences']
    w_sentences = batch['w_sentences']
    speaker_ids = common.flatten(batch['speakers'])
    genre = batch["genre"]
    if self.encoder.cbase:
      c_sentences, sentence_length, word_length = self.encoder.c_vocab.padding(c_sentences)
      input_feed[self.c_sentences] = np.array(c_sentences)

    if self.encoder.wbase:
      w_sentences, sentence_length = self.encoder.w_vocab.padding(w_sentences)
      input_feed[self.w_sentences] = np.array(w_sentences)
      input_feed[self.sentence_length] = np.array(sentence_length)

    gold_mentions = sorted(tuple(m) for m in common.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id
    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    # TODO: Add offsets to spans if BOS and EOS are added to input sentences.
    input_feed[self.gold_starts] = np.array(gold_starts)
    input_feed[self.gold_ends] = np.array(gold_ends)
    #input_feed[self.gold_spans] = np.array(gold_mentions)
    input_feed[self.cluster_ids] = np.array(cluster_ids)
    input_feed[self.speaker_ids] = np.array(speaker_ids)
    input_feed[self.genre] = np.array(genre)

    return input_feed

  ##############################################
  ##           Evaluation
  ##############################################

  def test(self, batches, conll_eval_path, official_stdout=False):
    def _k_to_tag(k):
      if k == -3:
        return "oracle"
      elif k == -2:
        return "actual"
      elif k == -1:
        return "exact"
      elif k == 0:
        return "threshold"
      else:
        return "{}%".format(k)
    mention_evaluators = { k:common.RetrievalEvaluator() for k in [-3, -2, -1, 0, 10, 15, 20, 25, 30, 40, 50] }

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    for example_num, example in enumerate(batches):
      feed_dict = self.get_input_feed(example)
      gold_starts = feed_dict[self.gold_starts]
      gold_ends = feed_dict[self.gold_ends]
      candidate_starts, candidate_ends, mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores = self.sess.run(self.predictions, feed_dict=feed_dict)
      self.evaluate_mentions(candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores, gold_starts, gold_ends, example, mention_evaluators)
      predicted_antecedents = self.get_predicted_antecedents(antecedents, antecedent_scores)

      coref_predictions[example["doc_key"]] = self.evaluate_coref(mention_starts, mention_ends, predicted_antecedents, example["clusters"], coref_evaluator)
      if example_num % 10 == 0:
        #print "Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data))
        #print "Evaluated {} examples.".format(example_num + 1)
        pass
    summary_dict = {}
    for k, evaluator in sorted(mention_evaluators.items(), key=operator.itemgetter(0)):
      tags = ["{} @ {}".format(t, _k_to_tag(k)) for t in ("R", "P", "F")]
      results_to_print = []
      for t, v in zip(tags, evaluator.metrics()):
        results_to_print.append("{:<10}: {:.2f}".format(t, v))
        summary_dict[t] = v
      print ", ".join(results_to_print)

    conll_results = conll.evaluate_conll(conll_eval_path, coref_predictions, official_stdout)
    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    summary_dict["Average F1 (conll)"] = average_f1
    print "Average F1 (conll): {:.2f}%".format(average_f1)

    p,r,f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print "Average F1 (py): {:.2f}%".format(f * 100)
    summary_dict["Average precision (py)"] = p
    print "Average precision (py): {:.2f}%".format(p * 100)
    summary_dict["Average recall (py)"] = r
    print "Average recall (py): {:.2f}%".format(r * 100)

    return tf_utils.make_summary(summary_dict), average_f1

  def evaluate_mentions(self, candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores, gold_starts, gold_ends, example, evaluators):
    text_length = sum(len(s) for s in example["w_sentences"])
    gold_spans = set(zip(gold_starts, gold_ends))

    if len(candidate_starts) > 0:
      sorted_starts, sorted_ends, _ = zip(*sorted(zip(candidate_starts, candidate_ends, mention_scores), key=operator.itemgetter(2), reverse=True))
    else:
      sorted_starts = []
      sorted_ends = []

    for k, evaluator in evaluators.items():
      if k == -3:
        predicted_spans = set(zip(candidate_starts, candidate_ends)) & gold_spans
      else:
        if k == -2:
          predicted_starts = mention_starts
          predicted_ends = mention_ends
        elif k == 0:
          is_predicted = mention_scores > 0
          predicted_starts = candidate_starts[is_predicted]
          predicted_ends = candidate_ends[is_predicted]
        else:
          if k == -1:
            num_predictions = len(gold_spans)
          else:
            num_predictions = (k * text_length) / 100
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
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(mention_starts[i]), int(mention_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

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
