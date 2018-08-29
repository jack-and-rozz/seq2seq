#coding: utf-8
import numpy as np
import os, re, sys, random, copy, time, json
from core.vocabulary.base import PAD_ID

class DatasetBase(object):
  @property
  def size(self):
    train_size = self.train.size if hasattr(self.train, 'size') else 0
    valid_size = self.valid.size if hasattr(self.valid, 'size') else 0
    test_size = self.test.size if hasattr(self.test, 'size') else 0
    return train_size, valid_size, test_size

# Functions for padding.

def fill_empty_brackets(sequence, max_len):
  """
  - sequence: A 1D list of list.
  """
  return sequence + [[] for _ in range(max_len - len(sequence))]

def fill_zero(sequence, length): # 最長系列が短すぎたときに0埋め
  '''
  Make the length of a sequence at least 'length' by truncating of filling 0.
  Args:
  sequence: A 1D list of integer.
  length: an integer.
  '''
  if len(sequence) < length:
    return sequence + [0 for _ in range(length - len(sequence))]
  else:
    return sequence


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

def padding(batch, minlen, maxlen, pad=PAD_ID):
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
  padded_batch = []

  length_of_this_dim = define_length(batch, minlen[0], maxlen[0])
  if rank == 2:
    return padding_2d(batch, minlen=minlen[0], maxlen=maxlen[0], pad=pad)

  for l in batch:
    l = fill_empty_brackets(l[:length_of_this_dim], length_of_this_dim)
    if rank == 3:
      l = padding_2d(l, minlen=minlen[1:], maxlen=maxlen[1:], pad=pad)
    else:
      l = padding(l, minlen=minlen[1:], maxlen=maxlen[1:], pad=pad)

    padded_batch.append(l)
  largest_shapes = [max(n_dims) for n_dims in zip(*[tensor.shape for tensor in padded_batch])]
  target_tensor = np.zeros([len(batch)] + largest_shapes)

  for i, tensor in enumerate(padded_batch):
    pad_lengths = [x - y for x, y in zip(largest_shapes, tensor.shape)]
    pad_shape = [(0, l) for l in pad_lengths] 
    padded_batch[i] = np.pad(tensor, pad_shape, 'constant')
  return np.array(padded_batch)
