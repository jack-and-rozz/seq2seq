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
from core.dataset.wikiP2D import mask_span

class _WikiP2DDescDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows):
    super().__init__(config, filename, vocab, max_rows)
    self.max_contexts = config.max_contexts

  def preprocess(self, article):
    return article

  def article2entries(self, article):
    if not (article.desc and article.contexts):
      return []

    entry = recDotDefaultDict()
    desc = article.desc.split()
    entry.title.raw = article.title
    entry.desc.raw = desc
    entry.desc.word = self.vocab.word.sent2ids(desc)
    entry.contexts.link = []
    entry.contexts.raw = []
    entry.contexts.word = []
    entry.contexts.char = []
    for context, link in article.contexts[:self.max_contexts]:
      entry.link.append(link)
      context = context.split()
      entry.contexts.raw.append(context)
      if self.mask_link:
        context = mask_span(context, link)
      entry.contexts.word.append(self.vocab.word.sent2ids(context))
      entry.contexts.char.append(self.vocab.char.sent2ids(context))
    return [entry]

  def padding(self, batch):
    '''
    batch.desc.word: [batch_size, max_words]
    batch.contexts.word: [batch_size, max_contexts, max_words]
    batch.contexts.char: [batch_size, max_contexts, max_words, max_chars]
    '''
    batch.contexts.char = padding(
      batch.contexts.char,
      minlen=[None, self.config.minlen.word, self.config.minlen.char],
      maxlen=[None, self.config.maxlen.word, self.config.maxlen.char])

    batch.contexts.word = padding(
      batch.contexts.word, 
      minlen=[None, self.config.minlen.word],
      maxlen=[None, self.config.maxlen.word])

    batch.desc.word = padding(
      batch.desc.word, 
      minlen=[self.config.minlen.word],
      maxlen=[self.config.maxlen.word])

    return batch

class WikiP2DDescDataset(DatasetBase):
  
  def __init__(self, config, vocab):
    self.vocab = vocab
    dataset_class =  _WikiP2DDescDataset
    self.train = self.dataset_class(config, config.filename.train, vocab, 
                                    config.max_rows.train)
    self.valid = self.dataset_class(config, config.filename.valid, vocab,
                                    config.max_rows.valid)
    self.test = self.dataset_class(config, config.filename.test, vocab,
                                   config.max_rows.test)

