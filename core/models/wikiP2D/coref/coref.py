# coding: utf-8 
import math, time, sys
import tensorflow as tf
from core.utils import common, tf_utils
from core.models.base import ModelBase

import numpy as np


class CorefModel(ModelBase):
  def __init__(self, config, encoder, activation=tf.nn.tanh):
    self.name = 'coref'
    self.hidden_size = config.hidden_size
    self.encoder = encoder
    self.activation = activation
    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)


  def get_predictions_and_loss(self, word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
    self.dropout = 1 - (tf.to_float(is_training) * self.config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * self.config["lexical_dropout_rate"])

    num_sentences = tf.shape(word_emb)[0]
    max_sentence_length = tf.shape(word_emb)[1]

    text_emb_list = [word_emb]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      text_emb_list.append(aggregated_char_emb)

    text_emb = tf.concat(text_emb_list, 2)
    text_emb = tf.nn.dropout(text_emb, self.lexical_dropout)

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
    text_len_mask = tf.reshape(text_len_mask, [num_sentences * max_sentence_length])

    text_outputs = self.encode_sentences(text_emb, text_len, text_len_mask)
    text_outputs = tf.nn.dropout(text_outputs, self.dropout)

    genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]

    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
    flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask) # [num_words]

    candidate_starts, candidate_ends = coref_ops.spans(
      sentence_indices=flattened_sentence_indices,
      max_width=self.max_mention_width)
    candidate_starts.set_shape([None])
    candidate_ends.set_shape([None])

    candidate_mention_emb = self.get_mention_emb(flattened_text_emb, text_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]
    candidate_mention_scores =  self.get_mention_scores(candidate_mention_emb) # [num_mentions, 1]
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [num_mentions]

    k = tf.to_int32(tf.floor(tf.to_float(tf.shape(text_outputs)[0]) * self.config["mention_ratio"]))
    predicted_mention_indices = coref_ops.extract_mentions(candidate_mention_scores, candidate_starts, candidate_ends, k) # ([k], [k])
    predicted_mention_indices.set_shape([None])

    mention_starts = tf.gather(candidate_starts, predicted_mention_indices) # [num_mentions]
    mention_ends = tf.gather(candidate_ends, predicted_mention_indices) # [num_mentions]
    mention_emb = tf.gather(candidate_mention_emb, predicted_mention_indices) # [num_mentions, emb]
    mention_scores = tf.gather(candidate_mention_scores, predicted_mention_indices) # [num_mentions]

    #mention_start_emb = tf.gather(text_outputs, mention_starts) # [num_mentions, emb]
    #mention_end_emb = tf.gather(text_outputs, mention_ends) # [num_mentions, emb]
    mention_speaker_ids = tf.gather(speaker_ids, mention_starts) # [num_mentions]

    max_antecedents = self.config["max_antecedents"]
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
    mention_emb_list = []

    mention_start_emb = tf.gather(text_outputs, mention_starts) # [num_mentions, emb]
    mention_emb_list.append(mention_start_emb)

    mention_end_emb = tf.gather(text_outputs, mention_ends) # [num_mentions, emb]
    mention_emb_list.append(mention_end_emb)
    mention_width = 1 + mention_ends - mention_starts # [num_mentions]
    if self.config["use_features"]:
      mention_width_index = mention_width - 1 # [num_mentions]
      mention_width_emb = tf.gather(tf.get_variable("mention_width_embeddings", [self.config["max_mention_width"], self.config["feature_size"]]), mention_width_index) # [num_mentions, emb]
      mention_width_emb = tf.nn.dropout(mention_width_emb, self.dropout)
      mention_emb_list.append(mention_width_emb)

    if self.config["model_heads"]:
      mention_indices = tf.expand_dims(tf.range(self.config["max_mention_width"]), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
      mention_indices = tf.minimum(util.shape(text_outputs, 0) - 1, mention_indices) # [num_mentions, max_mention_width]
      mention_text_emb = tf.gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]
      self.head_scores = util.projection(text_outputs, 1) # [num_words, 1]
      mention_head_scores = tf.gather(self.head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
      mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, self.config["max_mention_width"], dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]
      mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), dim=1) # [num_mentions, max_mention_width, 1]
      mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
      mention_emb_list.append(mention_head_emb)

    mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]
    return mention_emb

  def get_mention_scores(self, mention_emb):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(mention_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [num_mentions, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [num_mentions, max_ant + 1]
    print gold_scores
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [num_mentions]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [num_mentions]
    return log_norm - marginalized_gold_scores # [num_mentions]

  def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts, mention_ends, mention_speaker_ids, genre_emb):
    num_mentions = util.shape(mention_emb, 0)
    max_antecedents = util.shape(antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      antecedent_speaker_ids = tf.gather(mention_speaker_ids, antecedents) # [num_mentions, max_ant]
      same_speaker = tf.equal(tf.expand_dims(mention_speaker_ids, 1), antecedent_speaker_ids) # [num_mentions, max_ant]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [num_mentions, max_ant, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [num_mentions, max_antecedents, 1]) # [num_mentions, max_ant, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      target_indices = tf.range(num_mentions) # [num_mentions]
      mention_distance = tf.expand_dims(target_indices, 1) - antecedents # [num_mentions, max_ant]
      mention_distance_bins = coref_ops.distance_bins(mention_distance) # [num_mentions, max_ant]
      mention_distance_bins.set_shape([None, None])
      mention_distance_emb = tf.gather(tf.get_variable("mention_distance_emb", [10, self.config["feature_size"]]), mention_distance_bins) # [num_mentions, max_ant]
      feature_emb_list.append(mention_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [num_mentions, max_ant, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [num_mentions, max_ant, emb]

    antecedent_emb = tf.gather(mention_emb, antecedents) # [num_mentions, max_ant, emb]
    target_emb_tiled = tf.tile(tf.expand_dims(mention_emb, 1), [1, max_antecedents, 1]) # [num_mentions, max_ant, emb]
    similarity_emb = antecedent_emb * target_emb_tiled # [num_mentions, max_ant, emb]

    pair_emb = tf.concat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb], 2) # [num_mentions, max_ant, emb]

    with tf.variable_scope("iteration"):
      with tf.variable_scope("antecedent_scoring"):
        antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [num_mentions, max_ant, 1]
    antecedent_scores = tf.squeeze(antecedent_scores, 2) # [num_mentions, max_ant]

    antecedent_mask = tf.log(tf.sequence_mask(antecedents_len, max_antecedents, dtype=tf.float32)) # [num_mentions, max_ant]
    antecedent_scores += antecedent_mask # [num_mentions, max_ant]

    antecedent_scores += tf.expand_dims(mention_scores, 1) + tf.gather(mention_scores, antecedents) # [num_mentions, max_ant]
    antecedent_scores = tf.concat([tf.zeros([util.shape(mention_scores, 0), 1]), antecedent_scores], 1) # [num_mentions, max_ant + 1]
    return antecedent_scores  # [num_mentions, max_ant + 1]


  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, text_len_mask)

