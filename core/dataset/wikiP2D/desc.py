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

class _WikiP2DDescDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, mask_link):
    super().__init__(config, filename, vocab, max_rows)
    self.max_contexts = config.max_contexts
    self.mask_link = mask_link
    self.iterations_per_epoch = int(config.iterations_per_epoch)

  def preprocess(self, article):
    return article

  def article2entries(self, article):
    example = recDotDefaultDict()
    example.qid = article.qid
    example.title.raw = article.title

    example.desc.raw = self.vocab.decoder.word.tokenizer(article.desc)
    example.desc.word = self.vocab.decoder.word.sent2ids(article.desc)

    example.contexts.raw = []
    example.contexts.word = []
    example.contexts.char = []
    example.contexts.link = []
    #for context, link in article.contexts[:self.max_contexts]:
    for context, link in article.contexts:
      context = context.split()
      example.contexts.raw.append(context)

      if self.mask_link:
        context = mask_span(context, link)
      example.contexts.word.append(self.vocab.encoder.word.sent2ids(context))
      example.contexts.char.append(self.vocab.encoder.char.sent2ids(context))
      example.contexts.link.append(link)
    return [example]
  def sample(self, example, is_random=True):
    assert len(example.contexts.word) == len(example.contexts.char)
    assert len(example.contexts.word) == len(example.contexts.link)
    print('-------------------------------')
    example = copy.deepcopy(recDotDict(example))
    print(len(example.contexts.word))
    if is_random:
      idxs = random.sample(range(0, len(example.contexts.link)), self.max_contexts)
    else:
      idxs = range(0, self.max_contexts) # Use the first 'self.max_contexts' ones.
    example.contexts.raw = [example.contexts.raw[idx] for idx in idxs]
    example.contexts.word = [example.contexts.word[idx] for idx in idxs]
    example.contexts.char = [example.contexts.char[idx] for idx in idxs]
    example.contexts.link = [example.contexts.link[idx] for idx in idxs]
    print(len(example.contexts.word))
    return example

  def padding(self, batch):
    '''
    batch.contexts.word: [batch_size, max_contexts, max_words]
    batch.contexts.char: [batch_size, max_contexts, max_words, max_chars]
    batch.link: [batch_size, max_contexts, 2]
    '''

    # [batch_size, max_num_sent, max_num_word_in_sent]
    batch.contexts.word = padding(
      batch.contexts.word, 
      minlen=[None, self.config.minlen.word],
      maxlen=[None, self.config.maxlen.word])

    # [batch_size, max_num_sent, max_num_word_in_sent, max_num_char_in_word]
    batch.contexts.char = padding(
      batch.contexts.char,
      minlen=[None, self.config.minlen.word, self.config.minlen.char],
      maxlen=[None, self.config.maxlen.word, self.config.maxlen.char])

    # [batch_size, max_num_sent, 2]
    batch.contexts.link = padding(
      batch.contexts.link,
      minlen=[None, 2],
      maxlen=[None, 2])

    # [batch_size, max_num_word_in_sent]
    batch.desc.word = padding(
      batch.desc.word,
      minlen=[self.config.minlen.word],
      maxlen=[self.config.maxlen.word])
    return batch


class WikiP2DDescDataset(DatasetBase):
  dataset_class =  _WikiP2DDescDataset
  def __init__(self, config, vocab):
    self.vocab = vocab
    self.train = self.dataset_class(config, config.filename.train, vocab, 
                                    config.max_rows.train,
                                    config.mask_link)
    self.valid = self.dataset_class(config, config.filename.valid, vocab, 
                                    config.max_rows.valid, 
                                    config.mask_link)
    self.test = self.dataset_class(config, config.filename.test, vocab, 
                                   config.max_rows.test, 
                                   config.mask_link)
  
