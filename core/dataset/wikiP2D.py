#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from core.utils.common import recDotDefaultDict, recDotDict, flatten, batching_dicts, pad_sequences
#from core.utils import visualize
from core.dataset.base import DatasetBase
from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, fill_empty_brackets, fill_zero
from core.vocabulary.wikiP2D import WikiP2DVocabulary, WikiP2DSubjVocabulary, WikiP2DRelVocabulary, WikiP2DObjVocabulary

random.seed(0)

# これやると下の階層でそれぞれ最長が異なるのでうまくいかない。todo.
# def define_length(batch, minlen=None, maxlen=None):
#   # バッチ内で最長のものの長さとconfigの最長のうち小さい方を
#   length = max([len(b) for b in batch]) 
#   if maxlen:
#     length = min(length, maxlen)

#   # バッチ内最長が最短より短かった場合補完
#   if minlen and minlen > length:
#     length = minlen
#   return length

def define_length(batch, minlen=None, maxlen=None):
  if minlen is None:
    minlen = 0

  if maxlen:
    return max(maxlen, minlen)
  else:
    return max([len(b) for b in batch] + [minlen])


def padding_2d(batch, minlen=None, maxlen=None, pad=PAD_ID, pad_type='post'):
  '''
  Args:
  batch: a 2D list. 
  maxlen: an integer.
  Return:
  A 2D tensor of which shape is [batch_size, max_num_word].
  '''
  if type(maxlen) == list:
    maxlen = maxlen[0]
  if type(minlen) == list:
    minlen = minlen[0]

  length_of_this_dim = define_length(batch, minlen, maxlen)
  return np.array([fill_zero(l[:length_of_this_dim], length_of_this_dim) for l in batch])

  # return pad_sequences(
  #   batch, maxlen=maxlen, value=pad,
  #   padding=pad_type, truncating=pad_type)

def padding_3d(batch, minlen=[None, None], maxlen=[None, None]):
  '''
  Args:
  array: a 3D list. 
  maxlen: an list of integer. [max_num_word, max_num_char]
  Return:
  A 3D tensor of which shape is [batch_size, max_num_word, max_num_char].
  '''
  length_of_this_dim = define_length(batch, minlen[0], maxlen[0])
  padded_batch = []
  for l in batch:
    l = fill_empty_brackets(l[:length_of_this_dim], length_of_this_dim)
    # 別々にpadding_2dしたらそれぞれmaxlenが異なってしまうけどどうしよう
    l = padding_2d(l, minlen=minlen[1:], maxlen=maxlen[1:])
    padded_batch.append(l)
  return np.array(padded_batch)

def read_jsonlines(source_path, max_rows=0):
  data = []
  for i, l in enumerate(open(source_path)):
    if max_rows and i >= max_rows:
      break
    d = recDotDict(json.loads(l))
    data.append(d)
  return data

def qid2position(qid, article):
  assert qid in article.link
  begin, end = article.link[qid]
  entity =  recDotDefaultDict()
  entity.raw  = article.text[begin:end+1] 
  entity.position = (begin, end)
  return entity

def span2unk(raw_text, position):
  assert type(raw_text) == list
  raw_text = copy.deepcopy(raw_text)
  begin, end = position
  for i in range(begin, end+1):
    raw_text[i] = _UNK
  return raw_text


class _WikiP2DDataset(object):
  def __init__(self, config, filename, vocab):
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
    self.max_rows = config.max_rows

  def preprocess(self, article):
    def flatten_text_and_link(article):
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

    article = flatten_text_and_link(article)
    return article

  def article2entries(self, article):
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

  def padding(self, batch):

    '''
    TODO: paddingどうする？paddingfifoqueueをちゃんと使ったほうが良いかも. 
    '''
    batch.text.word = padding_2d(
       batch.text.word, 
       minlen=self.config.minlen.word,
       maxlen=self.config.maxlen.word)

    batch.text.char = padding_3d(
      batch.text.char, 
      minlen=[self.config.minlen.word, self.config.minlen.char],
      maxlen=[self.config.maxlen.word, self.config.maxlen.char])

    cnn_max_filter_width = 3
    batch.rel.word = padding_2d(batch.rel.word, 
                                minlen=cnn_max_filter_width, 
                                maxlen=None)
    batch.rel.char = padding_3d(batch.rel.char, 
                                minlen=[3, self.config.minlen.char], 
                                maxlen=[None, self.config.maxlen.char])
    return batch

  def get_batch(self, batch_size, do_shuffle=False):
    if not self.data:
      self.load_data()

    if do_shuffle:
      random.shuffle(self.data)

    for i, b in itertools.groupby(enumerate(self.data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      yield batch


class _WikiP2DGraphDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, properties, mask_link):
    super().__init__(config, filename, vocab)
    self.properties = properties
    self.mask_link = mask_link

  def article2entries(self, article):
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
        raw_text = span2unk(raw_text, entry.subj.position)
        raw_text = span2unk(raw_text, entry.obj.position)
      entry.text.word = self.vocab.word.sent2ids(raw_text)
      entry.text.char = self.vocab.char.sent2ids(raw_text)

      return entry

    positive = triple2entry(article.positive_triple, article, 1)
    negative = triple2entry(article.negative_triple, article, 0)
    return (positive, negative)

class _WikiP2DDescDataset(_WikiP2DDataset):
  def article2entries(self, article):
    entry = recDotDefaultDict()
    entry.text.raw = article.text
    raw_text = article.text
    entry.text.word = self.vocab.word.sent2ids(raw_text)
    entry.text.word = self.vocab.char.sent2ids(raw_text)
    entry.desc.raw = article.desc
    entry.desc.word = self.vocab.word.sent2ids(article.desc)
    return (entry)


class WikiP2DGraphDataset(DatasetBase):
  '''
  A class which contains train, valid, testing datasets.
  '''
  def __init__(self, config, vocab):
    self.vocab = vocab
    properties_path = os.path.join(config.source_dir, config.prop_data)
    self.properties = recDotDict({d['qid']:d for d in read_jsonlines(properties_path)})
    data_class =  _WikiP2DGraphDataset
    self.train = data_class(config, config.train_data, vocab,
                            self.properties, config.mask_link)

    self.valid = data_class(config, config.valid_data, vocab, 
                            self.properties, True)
    self.test = data_class(config, config.test_data, vocab, 
                           self.properties, True)


class WikiP2DDescDataset(DatasetBase):
  
  def __init__(self, config, vocab):
    self.vocab = vocab
    data_class =  _WikiP2DDescDataset
    self.train = data_class(config, config.train_data, vocab)
    self.valid = data_class(config, config.valid_data, vocab)
    self.test = data_class(config, config.test_data, vocab)

