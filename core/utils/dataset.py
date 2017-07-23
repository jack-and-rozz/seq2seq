#coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os, re, sys, random, copy
import commands, itertools
import MeCab
import numpy as np
import mojimoji
from tensorflow.python.platform import gfile

from core.utils import common
from core.utils.vocabulary import PAD_ID, GO_ID, EOS_ID, UNK_ID
from core.utils.vocabulary import WordNetSynsetVocabulary, WordNetRelationVocabulary


#_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def format_zen_han(l):
  l = l.decode('utf-8') if type(l) == str else l
  l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
  l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
  l = l.encode('utf-8')
  return l

def separate_numbers(sent):
  def addspace(m):
    return ' ' + m.group(0) + ' '
  return re.sub(_DIGIT_RE, addspace, sent).replace('  ', ' ')

def space_tokenizer(sent, do_format_zen_han=True, do_separate_numbers=True):
  if do_format_zen_han:
    sent = format_zen_han(sent)
  if do_separate_numbers:
    sent = separate_numbers(sent)
  return sent.replace('\n', '').split()

def padding_and_format(data, max_sequence_length, use_sequence_length=True):
  '''
  Caution:  if both do_reverse and use_sequence_length are True at the same time, many PAD_IDs and only a small part of a sentence are read.
  '''
  do_reverse = not use_sequence_length
  batch_size = len(data)
  encoder_size, decoder_size = max_sequence_length, max_sequence_length
  encoder_inputs, decoder_inputs, encoder_sequence_length = [], [], []
  for _, encoder_input, decoder_input in data:
    encoder_sequence_length.append(len(encoder_input))
    # Encoder inputs are padded and then reversed if do_reverse=True.
    encoder_pad = [PAD_ID for _ in xrange((encoder_size - len(encoder_input)))] 
    encoder_input = encoder_input + encoder_pad
    if do_reverse:
      encoder_input = list(reversed(encoder_input))
    encoder_inputs.append(encoder_input)

    # Decoder inputs get an extra "GO" and "EOS" symbol, and are padded then.
    decoder_pad_size = decoder_size - len(decoder_input) - 2
    decoder_inputs.append([GO_ID] + decoder_input + [EOS_ID] +
                          [PAD_ID] * decoder_pad_size)

  # Now we create batch-major vectors from the data selected above.
  batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

  # Batch encoder inputs are just re-indexed encoder_inputs.
  for length_idx in xrange(encoder_size):
    batch_encoder_inputs.append(
      np.array([encoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))

  # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
  for length_idx in xrange(decoder_size):
    batch_decoder_inputs.append(
      np.array([decoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Create target_weights to be 0 for targets that are padding.
    batch_weight = np.ones(batch_size, dtype=np.float32)
    for batch_idx in xrange(batch_size):
      # We set weight to 0 if the corresponding target is a PAD symbol.
      # The corresponding target is decoder_input shifted by 1 forward.
      if length_idx < decoder_size - 1:
        target = decoder_inputs[batch_idx][length_idx + 1]
      if length_idx == decoder_size - 1 or target == PAD_ID:
        batch_weight[batch_idx] = 0.0
    batch_weights.append(batch_weight)
  if not use_sequence_length:
    encoder_sequence_length = None 
  batch = common.dotDict({
    'encoder_inputs' : batch_encoder_inputs,
    'decoder_inputs' : batch_decoder_inputs,
    'target_weights' : batch_weights,
    'sequence_length' : encoder_sequence_length,
    'batch_size' : batch_size,
  })
  return batch


class ASPECDataset(object):
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


####################################

class DatasetBase(object):
  pass

class WordNetDataset(DatasetBase):
  def __init__(self, source_dir, processed_dir, filename, 
               s_vocab, r_vocab, max_rows=None):
    self.s_vocab = s_vocab
    self.r_vocab = r_vocab
    self.source_path = os.path.join(source_dir, filename)
    self.processed_path = os.path.join(processed_dir, filename)+ '.bin'
    self.max_rows = max_rows

    self.data = self.initialize_data(
      self.source_path, self.processed_path, max_rows)
    if max_rows:
      self.data = self.data[:max_rows]
    self.size = len(self.data)
    
  def initialize_data(self, source_path, processed_path, max_rows):
    def process(source_path, max_rows):
      def process_line(l):
        s1, r, s2 = l.replace('\n', '').split('\t')
        #return [1.0, (self.s_vocab.to_id(s1),
        #              self.r_vocab.to_id(r),
        #              self.s_vocab.to_id(s2))]
        return (self.s_vocab.to_id(s1),
                self.r_vocab.to_id(r),
                self.s_vocab.to_id(s2))

      data = [process_line(l) for i, l in enumerate(open(source_path)) if not max_rows or i < max_rows]
      return data
    return common.load_or_create(processed_path, process, source_path, max_rows)

  def get_batch(self, batch_size, 
                do_shuffle=False, n_batches=1, negative_sampling_rate=0.0):
    data = copy.deepcopy(self.data) if do_shuffle or negative_sampling_rate > 0 else self.data

    #if negative_sampling_rate > 0:
    #  data += self.negative_sample(int(len(self.data)*negative_sampling_rate))
    if do_shuffle:
      random.shuffle(data)
    # Extract n_batches * batch_size lines from data
    for i, b in itertools.groupby(enumerate(data), lambda x: x[0] // (batch_size*n_batches)):

      raw_batch = [x[1] for x in b] # (id, data) -> data
      # Yield 'n_batches' batches which have 'batch_size' lines
      batch = [[x[1] for x in d2] for j, d2 in itertools.groupby(enumerate(raw_batch), lambda x: x[0] // (len(raw_batch) // n_batches))]

      neg_batch = [self.negative_sample(batch[i], negative_sampling_rate) for i in xrange(n_batches)]

      if n_batches == 1:
        batch = batch[0]
        neg_batch = neg_batch[0]
      yield (batch, neg_batch)

  def negative_sample(self, batch, ns_rate):
    batch_size = int(len(batch) * ns_rate)
    if batch_size == 0:
      return None
    def _random_sample(batch_size):
      neg_batch = []
      for i in xrange(batch_size):
        subj, obj = random.sample(xrange(1, self.s_vocab.size), 2)
        rel = random.sample(xrange(1, self.r_vocab.size), 1)[0]
        #batch.append((0.0, (s1, r, s2)))
        neg_batch.append((subj, rel, obj))
      return neg_batch

    def _close_negative_sample(batch):
      neg_batch = []
      for subj, rel, _ in batch:
        obj = random.sample(xrange(1, self.s_vocab.size), 1)[0]
        neg_batch.append((subj, rel, obj))
      return neg_batch
    #return _random_sample(batch_size)
    return _close_negative_sample(batch)

