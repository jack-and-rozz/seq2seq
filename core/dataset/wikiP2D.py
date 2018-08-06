#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from core.models.wikiP2D.category.evaluation import decorate_text

from core.utils.common import recDotDefaultDict, recDotDict, flatten, batching_dicts, pad_sequences
#from core.utils import visualize
from core.dataset.base import DatasetBase
from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, fill_empty_brackets, fill_zero, VocabularyWithEmbedding
from core.vocabulary.wikiP2D import WikiP2DVocabulary, WikiP2DSubjVocabulary, WikiP2DRelVocabulary, WikiP2DObjVocabulary

random.seed(0)

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

def padding(batch, minlen, maxlen):
  '''
  Args:
  - batch: A list of tensors with different shapes.
  - minlen, maxlen: A list of integers or None. Each i-th element specifies the minimum (or maximum) size of the tensor in the rank i+1.
    minlen[i] is considered as 0 if it is None, and maxlen[i] is automatically set to be equal to the maximum size of 'batch', the input tensor.
  
  e.g. 
  [[1], [2, 3], [4, 5, 6]] with minlen=[None], maxlen=[None] should be
  [[1, 0, 0], [2, 3, 0], [4, 5, 6]]
  '''
  assert len(minlen) == len(maxlen)
  rank = len(minlen) + 1
  length_of_this_dim = define_length(batch, minlen[0], maxlen[0])
  padded_batch = []
  if rank == 2:
    return padding_2d(batch, minlen=minlen[0], maxlen=maxlen[0])

  for l in batch:
    l = fill_empty_brackets(l[:length_of_this_dim], length_of_this_dim)
    if rank == 3:
      l = padding_2d(l, minlen=minlen[1:], maxlen=maxlen[1:])
    else:
      l = padding(l, minlen=minlen[1:], maxlen=maxlen[1:])

    padded_batch.append(l)
  largest_shapes = [max(n_dims) for n_dims in zip(*[tensor.shape for tensor in padded_batch])]
  target_tensor = np.zeros([len(batch)] + largest_shapes)

  for i, tensor in enumerate(padded_batch):
    pad_lengths = [x - y for x, y in zip(largest_shapes, tensor.shape)]
    pad_shape = [(0, l) for l in pad_lengths] 
    padded_batch[i] = np.pad(tensor, pad_shape, 'constant')
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

def mask_span(raw_text, position, token=_UNK):
  assert type(raw_text) == list
  raw_text = copy.deepcopy(raw_text)
  begin, end = position
  for i in range(begin, end+1):
    raw_text[i] = token
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
    self.data = sorted(self.data, key=lambda d: len(d.contexts.raw))

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
    '''
    TODO: paddingどうする？paddingfifoqueueをちゃんと使ったほうが良いかも. 
    '''
    batch.text.word = padding_2d(
       batch.text.word, 
       minlen=self.config.minlen.word,
       maxlen=self.config.maxlen.word)

    batch.text.char = padding(
      batch.text.char, 
      minlen=[self.config.minlen.word, self.config.minlen.char],
      maxlen=[self.config.maxlen.word, self.config.maxlen.char])

    cnn_max_filter_width = 3
    batch.rel.word = padding_2d(batch.rel.word, 
                                minlen=cnn_max_filter_width, 
                                maxlen=None)
    batch.rel.char = padding(batch.rel.char, 
                             minlen=[3, self.config.minlen.char], 
                             maxlen=[None, self.config.maxlen.char])
    return batch

class _WikiP2DDescDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab):
    super().__init__(config, filename, vocab)
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

class _WikiP2DCategoryDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, mask_link):
    super().__init__(config, filename, vocab)
    self.max_contexts = config.max_contexts
    self.mask_link = mask_link

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
    batch.desc.word = padding_2d(
      batch.desc.word,
      minlen=[self.config.minlen.word],
      maxlen=[self.config.maxlen.word])
    return batch


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

class WikiP2DCategoryDataset(DatasetBase):
  def __init__(self, config, vocab):
    self.vocab = vocab

    #categories = [l.split()[0] for l in open(os.path.join(config.source_dir, config.category_vocab))][:config.category_size]
    self.vocab.category = VocabularyWithEmbedding(
      config.embeddings_conf, config.category_size, 
      start_vocab=[_UNK],
    )
    data_class =  _WikiP2DCategoryDataset
    self.train = data_class(config, config.train_data, vocab, config.mask_link)
    self.valid = data_class(config, config.valid_data, vocab, True)
    self.test = data_class(config, config.test_data, vocab, True)

    # def print_entry(self, entry):
  #   '''
  #   Args:
  #   - entry: An entry in a batch obtained from 'flatten_batch(batch)'.
  #   '''
  #   print('<Title>', entry.title.raw)
  #   print('<Contexts>')
  #   for i in range(len(entry.contexts.raw)):
  #     text = decorate_text(entry.contexts.raw[i], 
  #                          self.vocab,
  #                          link=entry.contexts.link[i],
  #                          word_ids=entry.contexts.word[i])
  #     print(text)
  #   print ('<Category>', self.vocab.category.id2token(entry.category.label))
  #   if entry.desc:
  #     desc = decorate_text(entry.desc.raw,
  #                          self.vocab,
  #                          word_ids=entry.desc.word)
  #    print ('<Desc>', desc)
