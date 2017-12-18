#coding: utf-8
from __future__ import absolute_import

import os, re, sys, random, copy
import commands, itertools
import MeCab
import numpy as np

from core.utils import common
from tensorflow.python.platform import gfile
from core.utils.dataset.base import TranslationDataset

####################################
##     Translation
####################################

class ASPECDataset(TranslationDataset):
  def __init__(self, source_dir, processed_dir, filename, s_vocab, t_vocab,
               max_sequence_length=None, max_rows=None):
    self.tokenizer = s_vocab.tokenizer
    self.s_vocab = s_vocab
    self.t_vocab = t_vocab

    s_data, s_source_path, _ = self.initialize_data(source_dir, processed_dir, 
                                                    filename, s_vocab,
                                                    max_rows=max_rows)
    t_data, t_source_path, _ = self.initialize_data(source_dir, processed_dir, 
                                                    filename, t_vocab,
                                                    max_rows=max_rows)
    self.s_source_path = s_source_path
    self.t_source_path = t_source_path

    self.data = [(i, s, t) for i,(s,t) in enumerate(zip(s_data, t_data))]
    if max_sequence_length:
      self.data = [(i, s, t) for (i, s, t) in self.data
                   if len(s) <= max_sequence_length and 
                   len(t) <= max_sequence_length - 2]
    self.size = len(self.data)
    self.max_sequence_length = [max([len(s)for (_, s, t) in self.data]),
                                max([len(t)for (_, s, t) in self.data])+2]
    self.max_sequence_length = max(self.max_sequence_length)

  def initialize_data(self, source_dir, processed_dir, filename, vocab, 
                      max_rows=None):
    source_path = os.path.join(source_dir, filename) + '.%s' % vocab.suffix
    processed_path = os.path.join(processed_dir, filename) + '.%s.Wids%d' % (vocab.suffix, vocab.size)
    data = []
    if gfile.Exists(processed_path):
      for i, l in enumerate(open(processed_path, 'r')):
        if max_rows and i >= max_rows:
          break
        data.append([int(x) for x in l.replace('\n', '').split()])
    else:
      for i, l in enumerate(open(source_path, 'r')):
        if i % 100000 == 0:
          print("  processing line %d" % i)
        data.append(vocab.to_ids(self.tokenizer(l)))
      with open(processed_path, 'w') as f:
        f.write('\n'.join([' '.join([str(x) for x in l]) for l in data]) + '\n')
    return data, source_path, processed_path

  def stat(self):
    print ("Corpus Statistics")
    lens = [len(d[1]) for d in self.data]
    lent = [len(d[2]) for d in self.data]
    print ('len-Source: (min, max, ave) = (%d, %d, %.2f)' % (min(lens), max(lens), sum(lens)/len(lens)))
    print ('len-Target: (min, max, ave) = (%d, %d, %.2f)' % (min(lent), max(lent), sum(lent)/len(lent)))

  def get_batch(self, batch_size, do_shuffle=False, n_batches=1):
    # get 'n_batches' batches each of which has 'batch_size' records.
    data = self.data
    if do_shuffle:
      data = copy.deepcopy(data)
      random.shuffle(data)
    # Extract n_batches * batch_size lines from data
    for i, d in itertools.groupby(enumerate(data), lambda x: x[0] // (batch_size*n_batches)):
      raw_batch = [x[1] for x in d]
      batch = [[x[1] for x in d2] for j, d2 in itertools.groupby(enumerate(raw_batch), lambda x: x[0] // (len(raw_batch) // n_batches))]
      # Yield 'n_batches' batches which have 'batch_size' lines
      yield batch if n_batches > 1 else batch[0]
