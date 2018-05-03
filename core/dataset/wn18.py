#coding: utf-8


import os, re, sys, random, copy, subprocess, itertools

from core.utils import common
from core.utils.dataset.base import DatasetBase

####################################
##     Wordnet (wn18)
####################################

class WordNetDataset(DatasetBase):
  def __init__(self, source_dir, processed_dir, filename, 
               s_vocab, r_vocab, max_rows=None):
    self.s_vocab = s_vocab
    self.r_vocab = r_vocab
    self.source_path = os.path.join(source_dir, filename)
    self.processed_path = os.path.join(processed_dir, filename)+ '.bin'
    self.max_rows = max_rows

    self.data = self.initialize_data(
      self.source_path, self.processed_path)
    if max_rows:
      self.data = self.data[:max_rows]
    self.size = len(self.data)

  def initialize_data(self, source_path, processed_path, max_rows):
    def process(source_path):
      def process_line(l):
        s1, r, s2 = l.replace('\n', '').split('\t')
        return (self.s_vocab.to_id(s1),
                self.r_vocab.to_id(r),
                self.s_vocab.to_id(s2))

      data = [process_line(l) for l in open(source_path)]
      return data
    return common.load_or_create(processed_path, process, source_path, max_rows)

  def get_train_batch(self, batch_size, 
                      do_shuffle=False, n_batches=1, negative_sampling_rate=0.0):
    data = copy.deepcopy(self.data) if do_shuffle or negative_sampling_rate > 0 else self.data
    if do_shuffle:
      random.shuffle(data)
    # Extract n_batches * batch_size lines from data
    for i, b in itertools.groupby(enumerate(data), lambda x: x[0] // (batch_size*n_batches)):

      raw_batch = [x[1] for x in b] # (id, data) -> data
      # Yield 'n_batches' batches which have 'batch_size' lines.
      batch = [[x[1] for x in d2] for j, d2 in itertools.groupby(enumerate(raw_batch), lambda x: x[0] // (len(raw_batch) // n_batches))]

      neg_batch = [self.negative_sample(batch[i], negative_sampling_rate) for i in range(n_batches)]

      if n_batches == 1:
        batch = batch[0]
        neg_batch = neg_batch[0]
      yield (batch, neg_batch)

  def get_test_batch(self, batch_size, filtered=False):
    data = self.data
    for i, l in enumerate(data):
      subj, rel, obj = l
      subj_replaced = [(i, rel, obj) for i in range(self.s_vocab.size)]
      obj_replaced = [(subj, rel, i) for i in range(self.s_vocab.size)]
      # Put the correct triple at the beginning.
      subj_replaced.remove((subj, rel, obj))
      obj_replaced.remove((subj, rel, obj))
      subj_replaced.insert(0, (subj, rel, obj))
      obj_replaced.insert(0, (subj, rel, obj))
      subj_replaced = common.batching(subj_replaced, batch_size)
      obj_replaced = common.batching(obj_replaced, batch_size)
      yield subj_replaced, obj_replaced

  def negative_sample(self, batch, ns_rate):
    batch_size = int(len(batch) * ns_rate)
    if batch_size == 0:
      return None
    def _random_sample(batch_size):
      neg_batch = []
      for i in range(batch_size):
        subj, obj = random.sample(range(1, self.s_vocab.size), 2)
        rel = random.sample(range(1, self.r_vocab.size), 1)[0]
        #batch.append((0.0, (s1, r, s2)))
        neg_batch.append((subj, rel, obj))
      return neg_batch

    def _close_negative_sample(batch):
      neg_batch = []
      for subj, rel, _ in batch:
        obj = random.sample(range(1, self.s_vocab.size), 1)[0]
        neg_batch.append((subj, rel, obj))
      return neg_batch
    #return _random_sample(batch_size)
    return _close_negative_sample(batch)
