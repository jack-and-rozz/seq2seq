#coding: utf-8
import tensorflow as tf
from pprint import pprint
import os, re, sys, random, time
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict, read_jsonlines
from core.utils.common import RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, MAGENTA, CYAN, colored

from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, VocabularyWithEmbedding, FeatureVocab
from core.dataset.base import DatasetBase, padding
from core.dataset.wikiP2D import mask_span, _WikiP2DDataset

class _WikiP2DDescDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, mask_link, is_training):
    super().__init__(config, filename, vocab, max_rows)
    self.max_contexts = config.max_contexts
    self.mask_link = mask_link
    self.iterations_per_epoch = int(config.iterations_per_epoch)
    self.is_training = is_training

    # To use tf.data
    self.batch_size = 100 # TODO: Feed batch_size to args
    self._dataset = None

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
    cnt = 0
    for context, link in article.contexts:
      if cnt >= self.max_contexts:
        break

      if not link[1] >= link[0]:
        continue

      context = context.split()
      if len(context) > self.config.maxlen.word:
        continue

      example.contexts.raw.append(context)
      if self.mask_link:
        context = mask_span(context, link)
      example.contexts.word.append(self.vocab.encoder.word.sent2ids(context))
      example.contexts.char.append(self.vocab.encoder.char.sent2ids(context))
      example.contexts.link.append(link)
      cnt += 1
    if not example.contexts.raw:
      return []
    return [example]

  def sample(self, example, is_random=True):
    return example
    # assert len(example.contexts.word) == len(example.contexts.char)
    # assert len(example.contexts.word) == len(example.contexts.link)
    # print('-------------------------------')
    # example = copy.deepcopy(recDotDict(example))
    # print(len(example.contexts.word))
    # if is_random:
    #   idxs = random.sample(range(0, len(example.contexts.link)), self.max_contexts)
    # else:
    #   idxs = range(0, self.max_contexts) # Use the first 'self.max_contexts' ones.
    # example.contexts.raw = [example.contexts.raw[idx] for idx in idxs]
    # example.contexts.word = [example.contexts.word[idx] for idx in idxs]
    # example.contexts.char = [example.contexts.char[idx] for idx in idxs]
    # example.contexts.link = [example.contexts.link[idx] for idx in idxs]
    # return example

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

  # # (DEBUG) ####################################
  # def get_batch(self, batch_size, do_shuffle=False):
  #   while True:
  #     yield {}


  # def setup_dataset(self):
  #   if not self.data:
  #     self.load_data()

  #   # Create lists containing all examples by keys.
  #   data = recDotDefaultDict()
  #   for d in self.data:
  #     data = batching_dicts(data, d)
  #   data = (data.contexts.word, data.contexts.char, data.contexts.link, data.desc.word)
  #   data = (np.array(d) for d in data)
  #   print([type(d) for d in data])
  #   #exit(1)
  #   output_types = (tf.int32 for _ in data)
  #   dataset = tf.data.Dataset.from_generator(
  #     lambda: zip(data),
  #     output_types=output_types)
  #   if self.is_training:
  #     dataset = dataset.shuffle(10000)
  #   padded_shapes = (
  #     [None, self.config.maxlen.word],
  #     [None,  self.config.maxlen.word, self.config.maxlen.char],
  #     [None, 2],
  #     [self.config.maxlen.word]
  #   )
  #   dataset = dataset.padded_batch(self.batch_size, padded_shapes)
  #   return dataset

  # @property
  # def dataset(self):
  #   if not self._dataset:
  #     self._dataset = self.setup_dataset()
  #   return self._dataset

  # (DEBUG) ####################################

class WikiP2DDescDataset(DatasetBase):
  dataset_class =  _WikiP2DDescDataset
  def __init__(self, config, vocab):
    self.vocab = vocab
    self.train = self.dataset_class(config, config.filename.train, vocab, 
                                    config.max_rows.train,
                                    config.mask_link, True)
    self.valid = self.dataset_class(config, config.filename.valid, vocab, 
                                    config.max_rows.valid, 
                                    config.mask_link, False)
    self.test = self.dataset_class(config, config.filename.test, vocab, 
                                   config.max_rows.test, 
                                   config.mask_link, False)
  
