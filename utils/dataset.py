#coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import gzip, os, re, tarfile, json, sys, collections, types, random, copy
import commands, itertools
import MeCab
import numpy as np
import mojimoji
from tensorflow.python.platform import gfile
from utils import common


_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

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


class Vocabulary(object):
  #def __init__(self, source_path, target_path, vocab_size):
  def __init__(self, source_dir, target_dir, vocab_file, lang, vocab_size):
    source_path = os.path.join(source_dir, vocab_file) + '.' + lang
    target_path = os.path.join(target_dir, vocab_file) + '.%s.Wvocab%d' %(lang, vocab_size)
    self.tokenizer = space_tokenizer
    self.normalize_digits = False
    self.create_vocabulary(source_path, target_path, vocab_size)
    self.vocab, self.rev_vocab = self.load_vocabulary(target_path)
    self.lang = lang
    self.size = vocab_size

  def get(self, token):
    if not self.normalize_digits:
      return self.vocab.get(token, UNK_ID)
    else:
      return self.vocab.get(re.sub(_DIGIT_RE, "0", token), UNK_ID)

  def to_tokens(self, ids):
    return [self.rev_vocab[_id] for _id in ids]
  def to_ids(self, tokens):
    return [self.get(w) for w in tokens]

  def load_vocabulary(self, vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab = [l.split('\t')[0] for l in f]
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

  def create_vocabulary(self, data_path, vocabulary_path, 
                        max_vocabulary_size):
    vocab = collections.defaultdict(int)
    counter = 0
    if not gfile.Exists(vocabulary_path):
      print("Creating vocabulary \"%s\" " % (vocabulary_path))
      for line in gfile.GFile(data_path, mode="r"):
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = self.tokenizer(line)
        for w in tokens:
          vocab[w] += 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      for w in _START_VOCAB:
        vocab[w] = 0
      n_unknown = sum([vocab[w] for w in vocab_list[max_vocabulary_size:]])
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      vocab[_UNK] = n_unknown
      vocab[_EOS] = counter
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write("%s\t%d\n" % (w, vocab[w]))


class ASPECDataset(object):
  def __init__(self, source_dir, target_dir, filename, s_vocab, t_vocab,
               max_sequence_length=None, max_rows=None):
    self.tokenizer = space_tokenizer
    self.s_vocab = s_vocab
    self.t_vocab = t_vocab

    s_data = self.initialize_data(source_dir, target_dir, filename, s_vocab,
                                  max_rows=max_rows)
    t_data = self.initialize_data(source_dir, target_dir, filename, t_vocab,
                                  max_rows=max_rows)

    #self.data = sorted([(i, s, t) for i,(s,t) in enumerate(zip(s_data, t_data))],key=lambda x: len(x[1]))
    self.data = [(i, s, t) for i,(s,t) in enumerate(zip(s_data, t_data))]
    if max_sequence_length:
      self.data = [(i, s, t) for (i, s, t) in self.data
                   if len(s) <= max_sequence_length and 
                   len(t) <= max_sequence_length - 2]
    self.size = len(self.data)
    self.largest_bucket = [max([len(s)for (_, s, t) in self.data]),
                           max([len(t)for (_, s, t) in self.data])+2]

  def initialize_data(self, source_dir, target_dir, filename, vocab, 
                      max_rows=None):
    source_path = os.path.join(source_dir, filename) + '.%s' % vocab.lang
    processed_path = os.path.join(target_dir, filename) + '.%s.Wids%d' % (vocab.lang, vocab.size)
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
    return data

  def stat(self):
    print ("Corpus Statistics")
    lens = [len(d[1]) for d in self.data]
    lent = [len(d[2]) for d in self.data]
    print ('len-Source: (min, max, ave) = (%d, %d, %.2f)' % (min(lens), max(lens), sum(lens)/len(lens)))
    print ('len-Target: (min, max, ave) = (%d, %d, %.2f)' % (min(lent), max(lent), sum(lent)/len(lent)))


  def get_batch(self, batch_size, do_shuffle=False, n_batches=1):
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

