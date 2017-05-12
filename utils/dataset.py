#coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import gzip, os, re, tarfile, json, sys, collections, types, commands
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


def space_tokenizer(sent, normalize_digits=False):
  sent = format_zen_han(sent)
  sent = separate_numbers(sent)
  return sent.replace('\n', '').split()

class Vocabulary(object):
  #def __init__(self, source_path, target_path, vocab_size):
  def __init__(self, source_dir, target_dir, vocab_file, lang, vocab_size):
    source_path = os.path.join(source_dir, vocab_file) + '.' + lang
    target_path = os.path.join(target_dir, vocab_file) + '.%s.vocab%d' %(lang, vocab_size)
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
  def __init__(self, source_dir, target_dir, filename, s_vocab, t_vocab):
    self.tokenizer = space_tokenizer
    self.s_vocab = s_vocab
    self.t_vocab = t_vocab
    s_data = self.initialize_data(source_dir, target_dir, filename, s_vocab)
    t_data = self.initialize_data(source_dir, target_dir, filename, t_vocab)
    self.data = sorted([(i, s, t) for i,(s,t) in enumerate(zip(s_data, t_data))], 
                       key=lambda x: len(x[1]))
    #self.data = [(i, s, t) for i,(s,t) in enumerate(zip(s_data, t_data))]
    print "Corpus Statistics"
    lens = [len(d[1]) for d in self.data]
    lent = [len(d[2]) for d in self.data]
    print 'Source: (min, max, ave) = (%d, %d, %.2f)' % (min(lens), max(lens), sum(lens)/len(lens))
    print 'Target: (min, max, ave) = (%d, %d, %.2f)' % (min(lent), max(lent), sum(lent)/len(lent))
    
    print "Corpus Example"
    for i, s, t in self.data[:3]:
      print '<%d>' %i
      print " ".join(self.s_vocab.to_tokens(s))
      print " ".join(self.t_vocab.to_tokens(t))
     

  def get_batch(self, batch_size):
    pass

  #@common.timewatch()
  def initialize_data(self, source_dir, target_dir, filename, vocab):
    source_path = os.path.join(source_dir, filename) + '.%s' % vocab.lang
    processed_path = os.path.join(target_dir, filename) + '.%s.Wids%d' % (vocab.lang, vocab.size)
    data = []
    if gfile.Exists(processed_path):
      for l in open(processed_path, 'r'):
        data.append([int(x) for x in l.replace('\n', '').split()])
    else:
      for l in open(source_path, 'r'):
        data.append(vocab.to_ids(self.tokenizer(l)))
      with open(processed_path, 'w') as f:
        f.write('\n'.join([' '.join([str(x) for x in l]) for l in data]))
    return data