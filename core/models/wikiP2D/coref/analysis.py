# coding: utf-8 
import operator
import math, time, sys, random, re, math
import numpy as np
import pandas as pd
from pprint import pprint

from core.utils.common import RED, BLUE, GREEN, YELLOW, MAGENTA, CYAN, WHITE, BOLD, BLACK, UNDERLINE, RESET, flatten, dotDict
from core.models.base import ModelBase
from core.models.wikiP2D.coref import coref_ops, conll, metrics, util
from core.vocabulary.base import _UNK, PAD_ID
from collections import OrderedDict, defaultdict
from copy import deepcopy

PRONOUNS = set([
  'he', 'his', 'him', 'she', 'her', 'hers', 
  'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 
  'we', 'our', 'us', 'ours', 
  'it', 'its', 'they', 'their', 'them', 'theirs', 
  'this', 'that', 'these', 'those', 
])

def get_statistics(results_by_mention_groups, word_vocab):
  def is_pronoun(mention, raw_text):
    begin, end  = mention
    mention_str = ' '.join(raw_text[begin:end+1])
    # mention: a string.
    if mention_str.lower() in PRONOUNS:
      return True
    return False

  def partially_unk(mention, raw_text):
    begin, end = mention
    unk_list = set([word_vocab.is_unk(w) for w in raw_text[begin:end+1]])
    if True in unk_list and False in unk_list:
      return True
    return False

  def all_unk(mention, raw_text):
    begin, end  = mention
    for w in raw_text[begin:end+1]:
      if not word_vocab.is_unk(w):
        return False
    return True

  def update(statistics, category, mention_group, raw_text):
    n_pronoun = len([m for m in mention_group if is_pronoun(m, raw_text)])
    n_partially_unk = len([m for m in mention_group if partially_unk(m, raw_text)])
    n_all_unk = len([m for m in mention_group if all_unk(m, raw_text)])

    # All pronouns are supposed to be in vocabulary.
    statistics[category]['pronoun'] += n_pronoun
    statistics[category]['all known'] += len(mention_group) - n_pronoun - n_partially_unk - n_all_unk
    statistics[category]['partially unk'] += n_partially_unk
    statistics[category]['all unk'] += n_all_unk
    statistics[category]['overall'] += len(mention_group)


  statistics = OrderedDict()
  for i, (mention_groups, raw_text) in enumerate(results_by_mention_groups):
    if i == 0:
      # Initialize statistics
      for key in list(mention_groups.keys()): #+ ['all']:
        statistics[key] = OrderedDict([
          ('pronoun', 0),
          ('all known', 0),
          ('partially unk', 0),
          ('all unk', 0),
          ('overall', 0),
        ])

    for category, mention_group in mention_groups.items():
      update(statistics, category, mention_group, raw_text)
  return statistics

def print_colored_text(raw_text, aligned, extracted_mentions, 
                       predicted_antecedents, speakers, word_vocab):
  """
  Speakers: A list of speaker-id
  """

  decorated_text = deepcopy(raw_text)
  if word_vocab:
    unknowns = [(i, i) for i, w in enumerate(raw_text) if word_vocab.is_unk(w)]
  else:
    unknowns = []

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
    (success_root, BLUE),
    (failure_root, MAGENTA),
    (success_linked, BLUE),
    (failure_linked, RED),
    (failure_unlinked, GREEN),
    (failure_unextracted, CYAN),
    (failure_irregular_mention, BOLD),
    # (success_root, BLUE+UNDERLINE),
    # (failure_root, MAGENTA+UNDERLINE),
    # (success_linked, BLUE+UNDERLINE),
    # (failure_linked, RED+UNDERLINE),
    # (failure_unlinked, GREEN+UNDERLINE),
    # (failure_unextracted, CYAN+UNDERLINE),
    # #(failure_extracted, MAGENTA),
    # (failure_irregular_mention, BOLD+UNDERLINE),
    (others, YELLOW),
    (unknowns, UNDERLINE)
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
    ('Success', success_root+ success_linked),
    #('Success (Root)', success_root),
    #('Success (Anaphora)', success_linked),
    ('Wrong or No Anaphora (Root)', failure_root),
    ('Wrong Antecedent', failure_linked), 
    ('No Antecedent (Anaphora)', failure_unlinked),
    ('Not Extracted', failure_unextracted),
    ('Only in Prediction', failure_irregular_mention),
  ])
  return decorated_text, mention_groups

