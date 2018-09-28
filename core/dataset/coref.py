#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

#from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.python.platform import gfile
from core.dataset.base import DatasetBase, padding
from core.vocabulary.base import _UNK, PAD_ID
from core.utils.common import dotDict, recDotDict, recDotDefaultDict, batching_dicts, flatten, flatten_batch

def load_data(fpath):
  sys.stderr.write("Loading coref dataset from \'%s\'... \n" % fpath)
  return [json.loads(line) for line in open(fpath)]

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
  def __init__(self, data, vocab, genre_vocab):
    self.vocab = vocab.encoder
    self.genre_vocab = genre_vocab
    self.data = self.preprocess(data)
    self.size = len(self.data)

  def preprocess(self, data):
    def _preprocess(jsonline):
      example = recDotDefaultDict()
      example.text.raw = jsonline['sentences']
      example.text.word = [self.vocab.word.sent2ids(l) for l in jsonline['sentences']]
      example.text.char = [self.vocab.char.sent2ids(l) for l in jsonline['sentences']]
      example.sentence_length = np.array([len([w for w in s if w != PAD_ID]) for s in example.text.word])

      example.genre = self.genre_vocab.token2id(jsonline['doc_key'][:2]) 
      example.doc_key = jsonline['doc_key'] # e.g. wb/c2e/00/c2e_0022_0 
      example.clusters = jsonline['clusters']

      # Assign unique IDs to each speaker, and the model checks whether a pair of two mentions is spoken by the same speaker or not.
      speakers = flatten(jsonline["speakers"])
      speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
      speaker_ids = [speaker_dict[s] for s in speakers]

      example.speakers = speaker_ids
      return example
    return [_preprocess(jsonline) for jsonline in data]

  def padding(self, batch):
    # [batch_size, max_num_sent, max_num_word_in_sent]
    batch.text.word = padding(
      batch.text.word,
      minlen = [None, None],
      maxlen = [None, None]) 

    # [batch_size, max_num_sent, max_num_word_in_sent, max_num_char_in_word]
    batch.text.char = padding(
      batch.text.char,
      minlen = [None, None, None],
      maxlen = [None, None, None]) 
    batch.speaker_ids = padding(
      batch.speakers, minlen=[None], maxlen=[None])
    batch.genre = np.array(batch.genre)
    return batch

  def get_batch(self, batch_size, do_shuffle=False):
    '''
    '''
    assert batch_size == 1 # The code by Li et.al cannot handle batched inputs.
    if do_shuffle:
      random.shuffle(self.data)

    for i, b in itertools.groupby(enumerate(self.data), 
                                  lambda x: x[0] // batch_size):
      raw_batch = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(raw_batch)
      batch = flatten_batch(batch)[0] # TODO
      yield batch

  def tensorize(self, data):
    '''
    Args
    - data : A list of dictionary.
    '''
    batch = recDotDefaultDict()
    for d in data:
      batch = batching_dicts(batch, d) # list of dictionaries to dictionary of lists.
    batch = self.padding(batch)
    return batch



class CoNLL2012CorefDataset(DatasetBase):
  def __init__(self, dataset_path, vocab):
    '''
    Args:
    - dataset_path : A dictionary which contains pathes of CoNLL datasets.
    - w_vocab :
    - c_vocab :
    '''
    source_dir = dataset_path.source_dir
    train_path = os.path.join(source_dir, dataset_path.train_data)
    valid_path = os.path.join(source_dir, dataset_path.valid_data)
    test_path = os.path.join(source_dir, dataset_path.test_data)
    self.vocab = vocab

    # As this dataset is relatively small, we don't need to store processed files.
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    test_data = load_data(test_path)

    genre_tokens = [d['doc_key'][:2] for d in train_data] # wb/c2e/00/c2e_0022_0 -> wb
    self.genre_vocab = FeatureVocab(genre_tokens)

    self.train = _CoNLL2012CorefDataset(train_data, vocab, self.genre_vocab)
    self.valid = _CoNLL2012CorefDataset(valid_data, vocab, self.genre_vocab)
    self.test = _CoNLL2012CorefDataset(test_data, vocab, self.genre_vocab)

