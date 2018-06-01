#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from tensorflow.python.platform import gfile
import core.utils.common as common
#from core.utils import visualize
from core.dataset.base import DatasetBase
from core.vocabulary.base import UNK_ID
from core.vocabulary.wikiP2D import WikiP2DVocabulary, WikiP2DSubjVocabulary, WikiP2DRelVocabulary, WikiP2DObjVocabulary

random.seed(0)


def read_jsonlines(source_path, max_rows=0):
  data = []
  for i, l in enumerate(open(source_path)):
    if max_rows and i >= max_rows:
      break
    d = common.recDotDict(json.loads(l))
    data.append(d)
  return data

class _WikiP2DDataset():
  def __init__(self, config, filename, vocab, properties):
    '''
    Args:
    - config:
    - filename:
    - vocab:
    - properties: A dictionary.
    '''
    self.source_path = os.path.join(config.source_dir, filename)
    self.vocab = vocab
    self.properties = properties
    self.data = [] # Lazy loading.
    self.max_rows = None

  def preprocess(self, article):
    entry = common.recDotDefaultDict()
    raw_text = [s.split() for s in article.text]
    entry.text.raw = raw_text
    entry.text.word = [self.vocab.word.sent2ids(s) for s in raw_text]
    entry.text.char = [self.vocab.char.sent2ids(s) for s in raw_text]

    # TODO: とりあえずsubject = title, object = link だけ
    # (title, rel, ?)
    subj_qid, rel_pid, obj_qid = article.question_triple
    raw_title = article.title.split('_')

    subj = raw_title
    entry.subj.raw = subj
    entry.subj.word = self.vocab.word.sent2ids(subj)
    entry.subj.char = self.vocab.char.sent2ids(subj)

    rel = self.properties[rel_pid].name.split()
    entry.rel.raw = rel
    entry.rel.word = self.vocab.word.sent2ids(rel)
    entry.rel.char = self.vocab.char.sent2ids(rel)

    links = article.link
    _, gold_sid, (gold_begin, gold_end) = links[obj_qid]
    entry.obj.raw = raw_text[gold_sid][gold_begin:gold_end+1]
    entry.obj.position = [(sent_id, begin, end) for (_, sent_id, (begin, end)) in links.values()]
    entry.obj.label = [1 if link_qid == obj_qid else 0 for link_qid in links]
    return entry

  def load_data(self):
    sys.stderr.write("Loading wikiP2D dataset from \'%s\'... \n" % self.source_path)
    data = read_jsonlines(self.source_path, max_rows=self.max_rows)
    self.data = [self.preprocess(d) for d in data]

  def tensorize(self, data):
    batch = common.recDotDefaultDict()
    for d in data:
      batch = common.add_entry_to_batch(batch, d)
    return batch

  def padding_2d(array):
    pass

  def padding_3d(array):
    pass

  def padding_4d(array):
    pass

  def padding(self, data):
    data.text.word = padding_3d(data.text.word)
    data.text.word = padding_4d(data.text.word)

    data.subj.word = padding_2d(data.subj.word)
    data.subj.char = padding_3d(data.subj.char)

    data.rel.word = padding_2d(data.rel.word)
    data.rel.char = padding_3d(data.rel.char)

  def get_batch(self, batch_size, do_shuffle=False):
    if not self.data:
      self.load_data()

    if do_shuffle:
      random.shuffle(self.data)

    for i, b in itertools.groupby(enumerate(self.data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      batch = self.padding(batch)
      yield batch

class WikiP2DDataset(DatasetBase):
  '''
  A class which contains train, valid, testing datasets.
  '''
  def __init__(self, config, vocab):
    self.vocab = vocab
    properties_path = os.path.join(config.source_dir, config.prop_data)
    self.properties = common.recDotDict({d['qid']:d for d in read_jsonlines(properties_path)})
    self.train = _WikiP2DDataset(config, config.train_data, vocab, self.properties)
    self.valid = _WikiP2DDataset(config, config.valid_data, vocab, self.properties)
    self.test = _WikiP2DDataset(config, config.test_data, vocab, self.properties)

