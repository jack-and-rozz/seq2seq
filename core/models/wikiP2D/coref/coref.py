# coding: utf-8 
import operator
import math, time, sys, random, re, math
import numpy as np
import tensorflow as tf
import pandas as pd
from pprint import pprint

from core.utils import tf_utils
from core.utils.common import RED, BLUE, GREEN, YELLOW, MAGENTA, CYAN, WHITE, BOLD, BLACK, UNDERLINE, RESET, flatten, dotDict
from core.models.base import ModelBase
from core.models.wikiP2D.coref import coref_ops, conll, metrics, util
from core.vocabulary.base import _UNK, PAD_ID
from collections import OrderedDict, defaultdict
from copy import deepcopy

def get_statistics(results_by_mention_groups):
  def is_pronoun(mention, raw_text):
    begin, end  = mention
    mention_str = ' '.join(raw_text[begin:end+1])
    # mention: a string.
    pronoun_list = set([
      'he', 'his', 'him', 'she', 'her', 'hers', 
      'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 
      'we', 'our', 'us', 'ours', 
      'it', 'its', 'they', 'their', 'them', 'theirs', 
      'this', 'that', 'these', 'those', 
    ])
    if mention_str.lower() in pronoun_list:
      return True
    return False

  def update(statistics, category, mention_group, raw_text):
    n_pronoun = len([m for m in mention_group if is_pronoun(m, raw_text)])
    statistics[category]['pronoun'] += n_pronoun
    statistics[category]['others'] += len(mention_group) - n_pronoun
    statistics['all']['pronoun'] += n_pronoun
    statistics['all']['others'] += len(mention_group) - n_pronoun

  statistics = OrderedDict()
  for i, (mention_groups, raw_text) in enumerate(results_by_mention_groups):
    if i == 0:
      # Initialize statistics
      for key in list(mention_groups.keys()) + ['all']:
        statistics[key] = OrderedDict([
          ('pronoun', 0),
          ('others', 0),
        ])

    for category, mention_group in mention_groups.items():
      update(statistics, category, mention_group, raw_text)
  return statistics

def print_colored_text(raw_text, aligned, extracted_mentions, 
                       predicted_antecedents, speakers):
  """
  Speakers: A list of speaker-id
  """
  
  decorated_text = deepcopy(raw_text)
  all_linked_mentions_in_gold = flatten([gold_cluster for gold_cluster, _ in aligned])
  all_linked_mentions_in_pred = flatten([pred_cluster for _, pred_cluster in aligned])

  #all_anaphora_in_pred = [source_mention for source_mention, target_mention_idx in zip(extracted_mentions, predicted_antecedents) if target_mention_idx >= 0]
  all_extracted_mentions_in_pred = extracted_mentions

  all_predicted_links = {source_mention:extracted_mentions[target_mention_idx] for source_mention, target_mention_idx in zip(extracted_mentions, predicted_antecedents) if target_mention_idx >= 0} 
  mention_to_gold_cluster_id = defaultdict(lambda: -1)
  for i, (gold_cluster, _) in enumerate(aligned):
    for gm in gold_cluster:
      mention_to_gold_cluster_id[gm] = i

  # Goldに存在するメンションについて
  ## Root のみ
  success_root = []        # 先行詞を持たず，predictionの対応するクラスタにも自身が存在 (blue)
  failure_root = []        # 後続のanaphoraが自身へのリンクを失敗した（リンクを張らなかった or 異なるクラスタからのリンクを張った)  (magenta)

  ## Anaphoric mention のみ
  success_linked = []      # メンションとその先行詞がGoldでも同一クラスタ (blue)
  failure_unlinked = []    # 先行詞の検出に失敗 (green)

  # Both
  failure_linked = []      # 誤ったクラスタにリンクされたメンション (red)
  failure_unextracted = [] # 抽出の段階で省かれたメンション (cyan)

  # Goldに存在しないメンションについて
  failure_extracted = []   # goldに存在しないpredのmention (ignored)
  failure_irregular_mention = []   # goldに存在せず，linkまでされたpredのmention (black)
  others = []

  for gold_cluster, pred_cluster in aligned:
    for gm in gold_cluster:
      if gm not in all_extracted_mentions_in_pred: # Extracted as a mention or not
        failure_unextracted.append(gm)
        continue

      if gm == gold_cluster[0]: # The mention is root in gold
        if gm not in all_predicted_links: # The root mention has no link
          if gm in pred_cluster: 
            success_root.append(gm) # Successfully linked from subsequent anaphora
          else:
            failure_root.append(gm) # Wrongly, or never linked from subsequent anaphora
        else: # Wrong linking from root
          failure_linked.append(gm)
      else: # The mention is anaphora in gold
        if gm in all_predicted_links: # The anaphora has an antecedent or not
          if mention_to_gold_cluster_id[gm] == mention_to_gold_cluster_id[all_predicted_links[gm]]: # The anaphora has a link to an antecedent in the same cluster
            success_linked.append(gm)
          else:
            failure_linked.append(gm)
        else: 
          failure_unlinked.append(gm)

  for pm in all_extracted_mentions_in_pred:
    if pm not in all_linked_mentions_in_gold:
      failure_extracted.append(pm) # Wrongly extracted mentions (not a failure)
      if pm in all_linked_mentions_in_pred:
        failure_irregular_mention.append(pm) # Wrongly extracted and linked mentions

  # Color words and spaces between the colored words as well for natural visualization in order from shorter ones. (the most inside color has a higher priority, e.g. "RED BLUE word RESET" = "BLUE word RESET")
  spaces = [" " for x in range(len(decorated_text))] 
  def _color(decorated_text, spaces, mentions_and_colors):
    for (begin, end), color in mentions_and_colors:
      for x in range(begin, end+1):
        decorated_text[x] = color + decorated_text[x] + RESET
        spaces[x] = color + spaces[x] + RESET if x < end else spaces[x]
    return decorated_text, spaces


  mentions_and_colors = [
    (success_root, BLUE+UNDERLINE),
    (failure_root, MAGENTA+UNDERLINE),
    (success_linked, BLUE+UNDERLINE),
    (failure_linked, RED+UNDERLINE),
    (failure_unlinked, GREEN+UNDERLINE),
    (failure_unextracted, CYAN+UNDERLINE),
    #(failure_extracted, MAGENTA),
    (failure_irregular_mention, BOLD+UNDERLINE),
    (others, YELLOW)
  ]
  
  mentions_and_colors = flatten([[(mention, color) for mention in mentions] for mentions, color in mentions_and_colors])
  mentions_and_colors = sorted(mentions_and_colors, key=lambda x: (x[0][1]-x[0][0]))
  _color(decorated_text, spaces, mentions_and_colors)

  # Enclose each mention with brackets in the order from the shorter ones. (In order to put shorter mentions inside among overlapping ones)
  decorated_punctuated_text = deepcopy(decorated_text)
  gold_cluster_with_id = flatten([[(i, begin, end) for begin, end in gold_cluster] for i, (gold_cluster, _) in enumerate(aligned)])
  gold_cluster_with_id = sorted(gold_cluster_with_id, key=lambda x: (x[2]-x[1]))
  for cluster_id, begin, end in gold_cluster_with_id:
    decorated_punctuated_text[begin] = '[' + decorated_punctuated_text[begin]
    decorated_punctuated_text[end] = decorated_punctuated_text[end] + BOLD + ']%02d' % cluster_id + RESET

  # Concatenate spaces (to add underlines to the spaces between the words included in a mention as well)
  decorated_text = [d+s for d, s in zip(decorated_text, spaces)]
  decorated_punctuated_text = [d+s for d, s in zip(decorated_punctuated_text, spaces)]

  # Add speakers info
  current_speaker = speakers[0]
  decorated_punctuated_text[0] = 'Speaker%d : ' % current_speaker + decorated_punctuated_text[0]
  for i, speaker_id in enumerate(speakers):
    if speaker_id != current_speaker:
      current_speaker = speaker_id
      decorated_punctuated_text[i-1] += '\n'
      decorated_punctuated_text[i] = 'Speaker%d : ' % current_speaker + decorated_punctuated_text[i]

  print("".join(decorated_punctuated_text) + '\n')

  mention_groups = OrderedDict([
    ('Success', success_root + success_linked),
    ('Wrong or No Anaphora (Root)', failure_root),
    ('Wrong Antecedent', failure_linked), 
    ('No Antecedent (Anaphora)', failure_unlinked),
    ('Unextracted', failure_unextracted),
    ('Irregular_mention', failure_irregular_mention),
  ])
  return decorated_text, mention_groups


class CoreferenceResolution(ModelBase):
  def __init__(self, sess, config, encoder, vocab,
               activation=tf.nn.tanh):
    super(CoreferenceResolution, self).__init__(sess, config)
    self.dataset = config.dataset.name
    self.sess = sess
    self.config = config
    self.encoder = encoder
    self.vocab = vocab
    self.activation = activation
    self.is_training = encoder.is_training
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate
    self.feature_size = config.f_embedding_size
    self.max_mention_width = config.max_mention_width
    self.max_training_sentences = config.max_training_sentences
    self.mention_ratio = config.mention_ratio
    self.max_antecedents = config.max_antecedents
    self.use_features = config.use_features
    self.use_metadata = config.use_metadata
    self.model_heads = config.model_heads
    self.ffnn_depth = config.ffnn_depth
    self.ffnn_size = config.ffnn_size

    # Placeholders
    with tf.name_scope('Placeholder'):
      self.text_ph = dotDict()
      self.text_ph.word = tf.placeholder(
        tf.int32, name='text.word',
        shape=[None, None]) if self.encoder.wbase else None
      self.text_ph.char = tf.placeholder(
        tf.int32, name='text.char',
        shape=[None, None, None]) if self.encoder.cbase else None
      # TODO: truncate_exampleしたものをw_sentencesにfeedした時、何故かtruncate前の単語数を数えてしまう。とりあえずsentence_lengthを直接feedすれば問題なし？
      #self.sentence_length = tf.count_nonzero(self.text_ph.word,
      #                                        axis=1, dtype=tf.int32)
      self.sentence_length = tf.placeholder(tf.int32, shape=[None])
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
      self.genre_emb = self.initialize_embeddings('genre', [self.vocab.genre.size, self.feature_size])
      self.mention_width_emb = self.initialize_embeddings("mention_width", [self.max_mention_width, self.feature_size])
      self.mention_distance_emb = self.initialize_embeddings("mention_distance", [10, self.feature_size])
    text_emb, text_outputs, state = encoder.encode([self.text_ph.word, self.text_ph.char], self.sentence_length)
    self.encoder_outputs = text_outputs # for adversarial MTL

    with tf.name_scope('predictions_and_loss'):
      self.predictions, self.loss = self.get_predictions_and_loss(text_emb, text_outputs, self.sentence_length, self.speaker_ids, self.genre, self.gold_starts, self.gold_ends, self.cluster_ids)

    self.outputs = [self.predictions]

    with tf.name_scope("Summary"):
      self.summary_loss = tf.placeholder(tf.float32, shape=[],
                                         name='coref_loss')


  def get_predictions_and_loss(self, text_emb, text_outputs, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids):
    with tf.name_scope('ReshapeEncoderOutputs'):
      num_sentences = tf.shape(text_emb)[0]
      max_sentence_length = tf.shape(text_emb)[1]
      text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
      text_len_mask = tf.reshape(text_len_mask, [num_sentences * max_sentence_length])
      genre_emb = tf.nn.embedding_lookup(self.genre_emb, genre)
      sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]

      flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
      flattened_text_emb = self.flatten_emb_by_sentence(text_emb, text_len_mask) # [num_words]
      text_outputs = self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    with tf.name_scope('SpanCandidates'):
      candidate_starts, candidate_ends = coref_ops.spans(
        sentence_indices=flattened_sentence_indices,
        max_width=self.max_mention_width)
      candidate_starts.set_shape([None])
      candidate_ends.set_shape([None])

    with tf.name_scope('Mentions'):
      # TODO: text_embとtext_outputsは別物
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
        mention_width_emb = tf.nn.dropout(mention_width_emb, self.keep_prob)
        mention_emb_list.append(mention_width_emb)

      if self.model_heads:
        mention_indices = tf.expand_dims(tf.range(self.max_mention_width), 0) + tf.expand_dims(mention_starts, 1) # [num_mentions, max_mention_width]
        mention_indices = tf.minimum(util.shape(text_outputs, 0) - 1, mention_indices) # [num_mentions, max_mention_width]
        mention_text_emb = tf.gather(text_emb, mention_indices) # [num_mentions, max_mention_width, emb]
        self.head_scores = util.projection(text_outputs, 1) # [num_words, 1]
        mention_head_scores = tf.gather(self.head_scores, mention_indices) # [num_mentions, max_mention_width, 1]
        mention_mask = tf.expand_dims(tf.sequence_mask(mention_width, self.max_mention_width, dtype=tf.float32), 2) # [num_mentions, max_mention_width, 1]
        mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask), dim=1) # [num_mentions, max_mention_width, 1]
        mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 1) # [num_mentions, emb]
        mention_emb_list.append(mention_head_emb)

      mention_emb = tf.concat(mention_emb_list, 1) # [num_mentions, emb]
    return mention_emb

  def get_mention_scores(self, mention_emb):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(mention_emb, self.ffnn_depth, self.ffnn_size, 1, self.keep_prob) # [num_mentions, 1]

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [num_mentions, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [num_mentions]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [num_mentions]
    return log_norm - marginalized_gold_scores # [num_mentions]

  def get_antecedent_scores(self, mention_emb, mention_scores, antecedents, antecedents_len, mention_starts, mention_ends, mention_speaker_ids, genre_emb):
    num_mentions = util.shape(mention_emb, 0)
    max_antecedents = util.shape(antecedents, 1)

    feature_emb_list = []

    if self.use_metadata:
      antecedent_speaker_ids = tf.gather(mention_speaker_ids, antecedents) # [num_mentions, max_ant]
      same_speaker = tf.equal(tf.expand_dims(mention_speaker_ids, 1), antecedent_speaker_ids) # [num_mentions, max_ant]
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
    feature_emb = tf.nn.dropout(feature_emb, self.keep_prob) # [num_mentions, max_ant, emb]

    antecedent_emb = tf.gather(mention_emb, antecedents) # [num_mentions, max_ant, emb]
    target_emb_tiled = tf.tile(tf.expand_dims(mention_emb, 1), [1, max_antecedents, 1]) # [num_mentions, max_ant, emb]
    similarity_emb = antecedent_emb * target_emb_tiled # [num_mentions, max_ant, emb]

    pair_emb = tf.concat([target_emb_tiled, antecedent_emb, similarity_emb, feature_emb], 2) # [num_mentions, max_ant, emb]

    with tf.variable_scope("iteration"):
      with tf.variable_scope("antecedent_scoring"):
        antecedent_scores = util.ffnn(pair_emb, self.ffnn_depth, self.ffnn_size, 1, self.keep_prob) # [num_mentions, max_ant, 1]
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

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = list(zip(*mentions))
    else:
      starts, ends = [], []
      #return np.array(starts), np.array(ends)
    return starts, ends

  def get_input_feed(self, batch, is_training):
    input_feed = {}
    clusters = batch["clusters"]
    c_sentences = np.array(batch['c_sentences'])
    w_sentences = np.array(batch['w_sentences'])
    speaker_ids = batch['speakers'] #flatten(batch['speakers'])
    genre = batch["genre"]
    ## Texts
    if self.encoder.cbase:
      input_feed[self.text_ph.char] = c_sentences

    if self.encoder.wbase:
      input_feed[self.text_ph.word] = w_sentences

    input_feed[self.sentence_length] = np.array([len([w for w in s if w != PAD_ID]) for s in w_sentences])
    ## Mention spans and their clusters
    gold_mentions = sorted(tuple(m) for m in flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id
    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    input_feed[self.gold_starts] = np.array(gold_starts)
    input_feed[self.gold_ends] = np.array(gold_ends)
    input_feed[self.cluster_ids] = np.array(cluster_ids)

    ## Metadata
    input_feed[self.is_training] = is_training
    if self.use_metadata:
      input_feed[self.speaker_ids] = np.array(speaker_ids)
      input_feed[self.genre] = np.array(genre)

    ######### INPUT DEBUG
    # if not is_training:
    #   with open('feed_dict.txt', 'w') as f:
    #     sys.stdout = f
    #     for k, v in input_feed.items():
    #       if re.search('w_sentences', k.name):
    #         print k.name
    #         print self.sess.run(tf.nn.embedding_lookup(self.encoder.w_embeddings, self.text_ph.word), input_feed)
    #       else:
    #         print k.name
    #         print v
    #     w = '_UNK'
    #     w_id = self.encoder.vocab.word.token2id(w)
    #     print w, w_id
    #     print self.sess.run(tf.nn.embedding_lookup(self.encoder.w_embeddings, tf.constant(w_id)))
    #     w = 'this'
    #     w_id = self.encoder.vocab.word.token2id(w)
    #     print w, w_id
    #     print self.sess.run(tf.nn.embedding_lookup(self.encoder.w_embeddings, tf.constant(w_id)))
    #     sys.stdout = sys.__stdout__
    #   exit(1)
    #########

    if is_training and len(w_sentences) > self.max_training_sentences:
      return self.truncate_example(input_feed)
    else:
      return input_feed

  def truncate_example(self, input_feed):
    # DEBUG
    # print ('length of w_sentences', len(input_feed[self.text_ph.word]), sum([len([w for w in s if w != PAD_ID]) for s in input_feed[self.text_ph.word]]))
    max_training_sentences = self.max_training_sentences
    num_sentences = input_feed[self.text_ph.word].shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)

    sentence_length = np.array([len([w for w in sent if w != PAD_ID]) for sent in input_feed[self.text_ph.word]]) # Number of words except PAD in each sentence.

    word_offset = sentence_length[:sentence_offset].sum() # The sum of the number of truncated words except PAD.
    num_words = sentence_length[sentence_offset:sentence_offset + max_training_sentences].sum()

    w_sentences = input_feed[self.text_ph.word][sentence_offset:sentence_offset + max_training_sentences,:]
    c_sentences = input_feed[self.text_ph.char][sentence_offset:sentence_offset + max_training_sentences,:,:]
    sentence_length = sentence_length[sentence_offset:sentence_offset + max_training_sentences]
    speaker_ids = input_feed[self.speaker_ids][word_offset: word_offset + num_words]
    gold_spans = np.logical_and(input_feed[self.gold_ends] >= word_offset, input_feed[self.gold_starts] < word_offset + num_words)
    gold_starts = input_feed[self.gold_starts][gold_spans] - word_offset
    gold_ends = input_feed[self.gold_ends][gold_spans] - word_offset
    cluster_ids = input_feed[self.cluster_ids][gold_spans]


    input_feed[self.text_ph.word] = w_sentences
    input_feed[self.text_ph.char] = c_sentences
    input_feed[self.sentence_length] = sentence_length
    input_feed[self.speaker_ids] = speaker_ids
    input_feed[self.gold_starts] = gold_starts
    input_feed[self.gold_ends] = gold_ends
    input_feed[self.cluster_ids] = cluster_ids
    # DEBUG
    #print ('length of trun w_sentences', len(input_feed[self.text_ph.word]), sum([len([w for w in s if w != PAD_ID]) for s in input_feed[self.text_ph.word]]))
    return input_feed

  ##############################################
  ##           Evaluation
  ##############################################

  def test(self, batches, gold_path, mode, official_stdout=False):
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
    mention_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 0] }

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    results = OrderedDict()

    for example_num, example in enumerate(batches):
      feed_dict = self.get_input_feed(example, False)
      gold_starts = feed_dict[self.gold_starts]
      gold_ends = feed_dict[self.gold_ends]
      candidate_starts, candidate_ends, mention_scores, mention_starts, mention_ends, antecedents, antecedent_scores = self.sess.run(self.predictions, feed_dict=feed_dict)
      self.evaluate_mentions(candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores, gold_starts, gold_ends, example, mention_evaluators)
      predicted_antecedents = self.get_predicted_antecedents(antecedents, antecedent_scores)
      coref_predictions[example["doc_key"]] = self.evaluate_coref(mention_starts, mention_ends, predicted_antecedents, example["clusters"], coref_evaluator)

      results[example['doc_key']] = dotDict({
        'raw_text': example['raw_text'],
        'speakers': example['speakers'],
        'extracted_mentions': [(begin, end) for begin, end in zip(mention_starts, mention_ends)],
        'predicted_antecedents': predicted_antecedents
      })

    summary_dict = {}
    for k, evaluator in sorted(list(mention_evaluators.items()), key=operator.itemgetter(0)):
      tags = ["mention/{} @ {}".format(t, _k_to_tag(k)) for t in ("R", "P", "F")]
      results_to_print = []
      for t, v in zip(tags, evaluator.metrics()):
        results_to_print.append("{:<10}: {:.2f}".format(t, v))
        summary_dict["coref/%s/" % mode + t] = v
      print(", ".join(results_to_print))

    conll_results = conll.evaluate_conll(gold_path, coref_predictions, official_stdout)
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

    return util.make_summary(summary_dict), [values['f'] for metric, values in conll_results.items()], results
    #return util.make_summary(summary_dict), average_f1, results

  def evaluate_mentions(self, candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores, gold_starts, gold_ends, example, evaluators):
    text_length = sum(len(s) for s in example["w_sentences"])
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

  def print_results(self, results):
    '''
    Args:
       - results: An Ordereddict keyed by 'doc_key', whose elements are a dictionary that has the keys, 'raw_text' and 'aligned_results'
    '''

    results_by_mention_groups = []
    for i, (doc_key, result) in enumerate(results.items()):
      print("===%03d===\t%s" % (i, doc_key))
      raw_text = flatten(result['raw_text'])
      extracted_mentions = result['extracted_mentions']
      predicted_antecedents = result['predicted_antecedents']
      aligned = result['aligned_results']
      speakers = result['speakers']
      
      print('<text>')
      decorated_text, mention_groups = print_colored_text(
        raw_text, aligned, extracted_mentions, predicted_antecedents, speakers)
      results_by_mention_groups.append([mention_groups, raw_text])
      print('<cluster>')

      for j, (gold_cluster, predicted_cluster) in enumerate(aligned):
        g = ["".join(decorated_text[s:e+1]) + str((s, e)) for (s,e) in gold_cluster]
        p = ["".join(decorated_text[s:e+1]) + str((s, e)) for (s,e) in predicted_cluster]
        print("%03d-G%02d  " % (i, j) , ', '.join(g))
        print("%03d-P%02d  " % (i, j) , ', '.join(p))
      print('')

    statistics = get_statistics(results_by_mention_groups)
    header = ['Category'] + list(statistics['all'].keys())
    n_mentions = sum(statistics['all'].values())
    data = [[category] + ['%.2f' % (100.0 * n / n_mentions) for n in cnt_by_pos.values()] for category, cnt_by_pos in statistics.items() if category != 'all']
    df = pd.DataFrame(data, columns=header)

    print ('<Mention group statistics>')
    print(df.ix[:, header])
