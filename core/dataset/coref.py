
#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from tensorflow.python.platform import gfile
import core.utils.common as common
from core.dataset.base import DatasetBase
from core.vocabulary.base import _UNK 


class FeatureVocab(object):
  def __init__(self, all_tokens):
    counter = Counter(all_tokens)
    self.freq = counter.values
    self.rev_vocab = list(counter.keys())
    self.rev_vocab.insert(0, _UNK)
    self.vocab = OrderedDict([(t, i) for i,t in enumerate(self.rev_vocab)])
    self.size = len(self.vocab)

  def __str__(self):
    return str(self.vocab)

  def token2id(self, token):
    return self.vocab.get(token, self.vocab.get(_UNK))

  def sent2ids(self, sent):
    return [self.token2id(t) for t in sent]

  def paragraph2ids(self, para):
    return [self.sent2ids(s) for s in para]

class _CoNLL2012CorefDataset(object):
  def __init__(self, data, w_vocab, c_vocab, genre_vocab):
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.genre_vocab = genre_vocab
    self.data = self.tokenize(data)
    self.size = len(self.data)

  def tokenize(self, data):
    def _tokenize(jsonline):
      ## As what the model wants to know is whether each antecedent-mention pair is spoken by the same speaker, speaker_vocab is created every time only from those who are in a paragraph.
      speakers = common.flatten(jsonline["speakers"])
      speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
      speaker_ids = [speaker_dict[s] for s in speakers]

      record = {
        'raw_text': jsonline['sentences'],
        'w_sentences': [self.w_vocab.sent2ids(l) for l in jsonline['sentences']],
        'c_sentences': [self.c_vocab.sent2ids(l) for l in jsonline['sentences']],
        'clusters': jsonline['clusters'],
        'genre': self.genre_vocab.token2id(jsonline['doc_key'][:2]), # wb
        'doc_key': jsonline['doc_key'], # wb/c2e/00/c2e_0022_0 (not tokenized)
        'speakers': speaker_ids,
      }
      return record
    res = [_tokenize(jsonline) for jsonline in data]
    return res

  def get_batch(self, batch_size, do_shuffle=False, n_batches=1):
    if do_shuffle:
      random.shuffle(self.data)

    for i, b in itertools.groupby(enumerate(self.data), 
                                  lambda x: x[0] // (batch_size*n_batches)):
      raw_batch = [x[1] for x in b] # (id, data) -> data
      batches = []
      for j, b2 in itertools.groupby(enumerate(raw_batch), lambda x: x[0] // (len(raw_batch) // n_batches)):
        b2 = [x[1] for x in b2]
        keys = list(b2[0].keys())

        # TODO: オリジナルのコードではbatch_size == 1を前提としているためとりあえずこうする
        if batch_size > 1:
          batch = {k:[d[k] for d in b2] for k in keys}
        else:
          batch = b2[0]
        #batch = common.dotDict({k:[d[k] for d in b2] for k in keys})
        batches.append(batch)
      if len(batches) == 1:
        batches = batches[0]
      yield batches

class CoNLL2012CorefDataset(DatasetBase):
  def __init__(self, dataset_path, w_vocab, c_vocab):
    '''
    Args:
    - dataset_path : A dictionary which contains pathes of CoNLL datasets.
    - w_vocab :
    - c_vocab :
    '''
    source_dir = dataset_path.source_dir
    train_file = dataset_path.train_data
    valid_file= dataset_path.valid_data
    test_file= dataset_path.test_data
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab

    if not w_vocab and not c_vocab:
      raise ValueError('You have to prepare vocabularies in advance.')

    def load_source(source_dir, fname):
      fpath = os.path.join(source_dir, fname)
      sys.stderr.write("Loading coref dataset from \'%s\'... \n" % fpath)
      return [json.loads(line) for line in open(fpath)]

    # As this dataset is relatively small, we don't need to create processed (symbolized) files.
    self.train = load_source(source_dir, train_file)
    self.valid = load_source(source_dir, valid_file)
    self.test = load_source(source_dir, test_file)

    genre_tokens = [d['doc_key'][:2] for d in self.train] # wb/c2e/00/c2e_0022_0 -> wb
    self.genre_vocab = FeatureVocab(genre_tokens)

    self.train = _CoNLL2012CorefDataset(self.train, w_vocab, c_vocab, self.genre_vocab)
    self.valid = _CoNLL2012CorefDataset(self.valid, w_vocab, c_vocab, self.genre_vocab)
    self.test = _CoNLL2012CorefDataset(self.test, w_vocab, c_vocab, self.genre_vocab)

