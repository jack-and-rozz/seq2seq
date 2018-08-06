#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

#from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.python.platform import gfile
from core.dataset.base import DatasetBase
from core.vocabulary.base import _UNK, PAD_ID, fill_empty_brackets
from core.utils.common import dotDict, flatten, pad_sequences

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
  def __init__(self, data, w_vocab, c_vocab, genre_vocab):
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.genre_vocab = genre_vocab
    self.data = self.symbolize(data)
    self.size = len(self.data)

  def symbolize(self, data):
    def _symbolize(jsonline):
      ## Since what our model wants to know is whether each antecedent-mention pair is spoken by the same speaker, speaker_vocab is created every time only from those who are in the paragraph.
      speakers = flatten(jsonline["speakers"])
      speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
      speaker_ids = [speaker_dict[s] for s in speakers]

      record = {
        'raw_text': jsonline['sentences'],
        'w_sentences': [self.w_vocab.sent2ids(l) for l in jsonline['sentences']],
        'c_sentences': [self.c_vocab.sent2ids(l) for l in jsonline['sentences']],
        'clusters': jsonline['clusters'],
        'genre': self.genre_vocab.token2id(jsonline['doc_key'][:2]), # wb
        'doc_key': jsonline['doc_key'], # wb/c2e/00/c2e_0022_0 (not symbolized)
        'speakers': speaker_ids,
      }
      return record
    res = [_symbolize(jsonline) for jsonline in data]
    return res

  def get_batch(self, batch_size, do_shuffle=False, n_batches=1):
    '''
    '''
    if do_shuffle:
      random.shuffle(self.data)

    for i, b in itertools.groupby(enumerate(self.data), 
                                  lambda x: x[0] // (batch_size*n_batches)):
      raw_batch = [x[1] for x in b] # (id, data) -> data
      batches = []
      for j, b2 in itertools.groupby(enumerate(raw_batch), lambda x: x[0] // (len(raw_batch) // n_batches)):
        batched_raw_data = [x[1] for x in b2]
        batch = self.create_batch(batched_raw_data)
        batches.append(batch)
      if len(batches) == 1:
        batches = batches[0]
      yield batches

  def create_batch(self, data):
    '''
    Args
    - data : A list of dictionary.
    '''
    # TODO: batch_size is supposed to only be 1 in the original code.
    if len(data) > 1:
      raise NotImplementedError
    else:
      data = data[0]
      data = dotDict(data)
      data.w_sentences = self.w_padding(data.w_sentences)
      data.c_sentences = self.c_padding(data.c_sentences)
    return data

  def w_padding(self, inputs, max_sent_len=None):
    _max_sent_len = max([len(sent) for sent in inputs])
    if not max_sent_len or _max_sent_len < max_sent_len:
      max_sent_len = _max_sent_len

    padded_sentences = pad_sequences(
      inputs, maxlen=max_sent_len, 
      padding='post', truncating='post', value=PAD_ID)
    return padded_sentences # [num_sentences, max_sent_len]

  def c_padding(self, inputs, min_word_len=5, 
                max_sent_len=None, max_word_len=None, ):
    '''
    Args:
    - inputs: A list of 2d-array. Each element is a char-based sentence.
              # [num_sentences, num_words, num_chars]
    - min_word_len: The minimum number of characters. This must be equal or higher than CNN filter size used in model's encoder.
    - max_sent_len: The maximum number of words.
    - max_char_len: The maximum number of characters.
    '''
    _max_sent_len = max([len(sent) for sent in inputs])
    if not max_sent_len or _max_sent_len < max_sent_len:
      max_sent_len = _max_sent_len

    _max_word_len = max([max([len(word) for word in sent]) for sent in inputs])
    if not max_word_len or _max_word_len < max_word_len:
      max_word_len = _max_word_len

    # Because of the maximum window width of CNN.
    if max_word_len < min_word_len:
      max_word_len = min_word_len

    padded_sentences = [fill_empty_brackets(sent, max_sent_len) for sent in inputs] 
    padded_sentences = [pad_sequences(sent, maxlen=max_word_len, 
                                      padding='post', truncating='post', 
                                      value=PAD_ID) for sent in padded_sentences]
    return padded_sentences # [num_sentences, max_sent_len, max_word_len]


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

    if 'word' not in vocab and 'char' not in vocab:
      raise ValueError('You have to prepare vocabularies in advance.')


    # As this dataset is relatively small, we don't need to create processed (symbolized) files.
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)
    test_data = load_data(test_path)

    genre_tokens = [d['doc_key'][:2] for d in train_data] # wb/c2e/00/c2e_0022_0 -> wb
    self.genre_vocab = FeatureVocab(genre_tokens)

    self.train = _CoNLL2012CorefDataset(train_data, vocab.word, vocab.char, self.genre_vocab)
    self.valid = _CoNLL2012CorefDataset(valid_data, vocab.word, vocab.char, self.genre_vocab)
    self.test = _CoNLL2012CorefDataset(test_data, vocab.word, vocab.char, self.genre_vocab)

