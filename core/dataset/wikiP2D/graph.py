#coding: utf-8
from pprint import pprint
import os, re, sys, random, copy, time
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from core.models.wikiP2D.category.evaluation import decorate_text

from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict, read_jsonlines
from core.utils.common import RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, MAGENTA, CYAN, colored

from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, VocabularyWithEmbedding, FeatureVocab
from core.vocabulary.wikiP2D import WikiP2DRelVocabulary 

from core.dataset.base import DatasetBase, padding
from core.dataset.wikiP2D import mask_span, _WikiP2DDataset

class _WikiP2DGraphDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, properties, mask_link):
    super().__init__(config, filename, vocab, max_rows)
    self.properties = properties
    self.mask_link = mask_link

  def preprocess(self, article):
    raw_text = [s.split() for s in article.text]
    num_words = [len(s) for s in raw_text]
    links = {}

    # Convert a list of sentneces to a flattened sequence of words.
    for qid, link in article.link.items():
      (sent_id, (begin, end)) = link
      flatten_begin = begin + sum(num_words[:sent_id])
      flatten_end = end + sum(num_words[:sent_id])
      assert flatten_begin >= 0 and flatten_end >= 0
      links[qid] = (flatten_begin, flatten_end)
    article.link = links
    article.text = flatten(raw_text)
    article.desc = article.desc.split()
    return article

  def article2entries(self, article):
    if not (article.text and article.positive_triple and article.negative_triple):
      return []

    def qid2position(qid, article):
      assert qid in article.link
      begin, end = article.link[qid]
      entity =  recDotDefaultDict()
      entity.raw  = article.text[begin:end+1] 
      entity.position = (begin, end)
    return entity

    def triple2entry(triple, article, label):
      entry = recDotDefaultDict()
      entry.qid = article.qid

      subj_qid, rel_pid, obj_qid = triple
      rel = self.properties[rel_pid].name.split()
      entry.rel.raw = rel  # 1D tensor of str. 
      entry.rel.word = self.vocab.word.sent2ids(rel) # 1D tensor of int.
      entry.rel.char = self.vocab.char.sent2ids(rel) # 2D tensor of int.
      entry.subj = qid2position(subj_qid, article) # (begin, end)
      entry.obj = qid2position(obj_qid, article)# (begin, end)
      entry.label = label # 1 or 0.

      entry.text.raw = article.text
      raw_text = article.text
      if self.mask_link:
        raw_text = mask_span(raw_text, entry.subj.position)
        raw_text = mask_span(raw_text, entry.obj.position)
      entry.text.word = self.vocab.word.sent2ids(raw_text)
      entry.text.char = self.vocab.char.sent2ids(raw_text)
      
      return entry

    positive = triple2entry(article.positive_triple, article, 1)
    negative = triple2entry(article.negative_triple, article, 0)
    return [positive, negative]

  def padding(self, batch):
    batch.text.word = padding(
       batch.text.word, 
       minlen=[self.config.minlen.word],
       maxlen=[self.config.maxlen.word])

    batch.text.char = padding(
      batch.text.char, 
      minlen=[self.config.minlen.word, self.config.minlen.char],
      maxlen=[self.config.maxlen.word, self.config.maxlen.char])

    cnn_max_filter_width = 3
    batch.rel.word = padding(batch.rel.word, 
                                minlen=[cnn_max_filter_width], 
                                maxlen=[None])
    batch.rel.char = padding(batch.rel.char, 
                             minlen=[3, self.config.minlen.char], 
                             maxlen=[None, self.config.maxlen.char])
    return batch

class WikiP2DGraphDataset(DatasetBase):
  '''
  A class which contains train, valid, testing datasets.
  '''
  dataset_class =  _WikiP2DGraphDataset
  def __init__(self, config, vocab, mask_link_in_test=True):
    self.vocab = vocab
    properties_path = os.path.join(config.source_dir, config.prop_data)
    self.properties = OrderedDict([(d['qid'], d) 
                                   for d in read_jsonlines(properties_path)])
    self.vocab.rel = WikiP2DRelVocabulary(self.properties.values(), 
                                          start_vocab=[_UNK])
    self.train = self.dataset_class(config, config.filename.train, vocab,
                                    config.max_rows.train,
                                    self.properties, config.mask_link)

    self.valid = self.dataset_class(config, config.filename.valid, vocab, 
                                    config.max_rows.valid,
                                    self.properties, mask_link_in_test)
    self.test = self.dataset_class(config, config.filename.test, vocab, 
                                    config.max_rows.test,
                                   self.properties, mask_link_in_test)

