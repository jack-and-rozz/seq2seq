#coding: utf-8
import os, re, sys, random, copy
import subprocess, itertools
from collections import OrderedDict, defaultdict
from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict, read_jsonlines
from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, VocabularyWithEmbedding, FeatureVocab

def mask_span(raw_text, position, token=_UNK):
  assert type(raw_text) == list
  raw_text = copy.deepcopy(raw_text)
  begin, end = position
  for i in range(begin, end+1):
    raw_text[i] = token
  return raw_text

class _WikiP2DDataset(object):
  '''
  A class for dataset divided to train, dev, test portions.
  The class containing three instances of this class is also defined aside.
  '''
  def __init__(self, config, filename, vocab, max_rows):
    '''
    Args:
    - config:
    - filename:
    - vocab:
    '''
    self.source_path = os.path.join(config.source_dir, filename)
    self.config = config
    self.vocab = vocab
    self.data = [] # Lazy loading.
    self.max_rows = max_rows

  def preprocess(self, article):
    raise NotImplementedError

  def article2entries(self, article):
    '''
    Args:
    - article: An instance of recDotDict.
    Return:
    A list of entry which is an instance of recDotDict.
    '''
    raise NotImplementedError

  def padding(self, batch):
    raise NotImplementedError

  @property
  def size(self):
    if len(self.data) == 0:
      self.load_data()
    return len(self.data)

  def load_data(self):
    sys.stderr.write("Loading wikiP2D dataset from \'%s\'... \n" % self.source_path)
    data = read_jsonlines(self.source_path, max_rows=self.max_rows)
    data = [self.preprocess(d) for d in data]
    self.data = flatten([self.article2entries(d) for d in data])

  def tensorize(self, data):
    batch = recDotDefaultDict()
    for d in data:
      batch = batching_dicts(batch, d) # list of dictionaries to dictionary of lists.
    batch = self.padding(batch)
    return batch

  def get_batch(self, batch_size, do_shuffle=False):
    if not self.data:
      self.load_data()

    if do_shuffle:
      random.shuffle(self.data)
      if hasattr(self, 'iterations_per_epoch') and self.iterations_per_epoch:
        data = self.data[:self.iterations_per_epoch * batch_size]
      else:
        data = self.data
    else:
      data = self.data

    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      yield batch

