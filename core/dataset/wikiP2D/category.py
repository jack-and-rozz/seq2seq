#coding: utf-8
from pprint import pprint
import os, re, sys, random, copy, time
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict, read_jsonlines
from core.utils.common import RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, MAGENTA, CYAN, colored

from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, VocabularyWithEmbedding, FeatureVocab
from core.dataset.base import DatasetBase, padding
from core.dataset.wikiP2D import mask_span, _WikiP2DDataset
from core.models.wikiP2D.category.evaluation import decorate_text

class _WikiP2DCategoryDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, mask_link):
    super().__init__(config, filename, vocab, max_rows)
    self.max_contexts = config.max_contexts
    self.mask_link = mask_link
    self.iterations_per_epoch = int(config.iterations_per_epoch)
    self.data_by_category = None

  def preprocess(self, article):
    return article

  def article2entries(self, article):
    if not (article.category and article.contexts):
      return []

    entry = recDotDefaultDict()
    entry.title.raw = article.title
    desc = article.desc.split()
    entry.desc.raw = desc
    entry.desc.word = self.vocab.word.sent2ids(desc)

    entry.category.raw = article.category
    entry.category.label = self.vocab.category.token2id(article.category)
    if entry.category.label == self.vocab.category.token2id(_UNK):
      return []
    entry.contexts.raw = []
    entry.contexts.word = []
    entry.contexts.char = []
    entry.contexts.link = []
    for context, link in article.contexts[:self.max_contexts]:
      context = context.split()
      entry.contexts.raw.append(context)

      if self.mask_link:
        context = mask_span(context, link)
      entry.contexts.word.append(self.vocab.word.sent2ids(context))
      entry.contexts.char.append(self.vocab.char.sent2ids(context))
      entry.contexts.link.append(link)
    return [entry]

  def padding(self, batch):
    '''
    batch.contexts.word: [batch_size, max_contexts, max_words]
    batch.contexts.char: [batch_size, max_contexts, max_words, max_chars]
    batch.link: [batch_size, max_contexts, 2]
    '''
    batch.contexts.char = padding(
      batch.contexts.char,
      minlen=[None, self.config.minlen.word, self.config.minlen.char],
      maxlen=[None, self.config.maxlen.word, self.config.maxlen.char])
    batch.contexts.word = padding(
      batch.contexts.word, 
      minlen=[None, self.config.minlen.word],
      maxlen=[None, self.config.maxlen.word])
    batch.contexts.link = padding(
      batch.contexts.link,
      minlen=[None, 2],
      maxlen=[None, 2])
    batch.desc.word = padding(
      batch.desc.word,
      minlen=[self.config.minlen.word],
      maxlen=[self.config.maxlen.word])
    return batch

  def get_batch(self, batch_size, do_shuffle=False):
    if not self.data:
      self.load_data()

    if do_shuffle:
      random.shuffle(self.data)
      if not self.data_by_category:
        self.data_by_category = defaultdict(list)
        for d in self.data:
          self.data_by_category[d.category.raw].append(d)
      data = [random.choice(random.choice(list(self.data_by_category.values()))) 
              for _ in range(self.iterations_per_epoch * batch_size)]
    else:
      data = self.data

    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      yield batch


class WikiP2DCategoryDataset(DatasetBase):
  dataset_class =  _WikiP2DCategoryDataset
  def __init__(self, config, vocab):
    self.vocab = vocab

    #categories = [l.split()[0] for l in open(os.path.join(config.source_dir, config.category_vocab))][:config.category_size]
    self.vocab.category = VocabularyWithEmbedding(
      config.embeddings_conf, config.category_size, 
      start_vocab=[_UNK],
    )
    self.train = self.dataset_class(config, config.filename.train, vocab, 
                                    config.max_rows.train,
                                    config.mask_link)
    self.valid = self.dataset_class(config, config.filename.valid, vocab, 
                                    config.max_rows.valid, True)
    self.test = self.dataset_class(config, config.filename.test, vocab, 
                                   config.max_rows.test, True)
  
