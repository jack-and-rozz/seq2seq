#coding: utf-8
import collections, os, time, re, sys
from tensorflow.python.platform import gfile
import core.utils.common as common
import numpy as np

_PAD = "_PAD"
_BOS = "_BOS"
_EOS = "_EOS"
_UNK = "_UNK"

ERROR_ID = -1
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

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

def space_tokenizer(sent, do_format_zen_han=True, 
                    do_separate_numbers=True):
  if do_format_zen_han:
    sent = format_zen_han(sent)
  if do_separate_numbers:
    sent = separate_numbers(sent)
  return sent.replace('\n', '').split()


class VocabularyBase(object):
  def __init__(self):
    self.vocab = None
    self.rev_vocab = None

  def load_vocab(self, vocab_path):
    raise NotImplementedError

  def save_vocab(self, vocab_with_freq, vocab_path):
    raise NotImplementedError

  def id2token(self, _id):
    if _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
    elif _id in set([PAD_ID, EOS_ID, BOS_ID]):
      return ''
    else:
      return self.rev_vocab[_id]


class VocabularyWithEmbedding(VocabularyBase):
  def __init__(self, emb_files, source_dir="dataset/embeddings"):
    '''
    All pretrained embeddings must be under the source_dir.'
    '''
    embedding_path = [os.path.join(source_dir, f) for f in emb_files]
    self.embeddings = [PretrainedEmbeddings(p) for p in embedding_path]
    for e in self.embeddings:
      print e
  def to_tokens(self, ids)
  def load(self, embedding_path, embedding_format='txt'):
    '''
    Load pretrained vocabularies.
    '''
    sys.stderr.write("Loading word embeddings from {}...\n".format(embedding_path))
    skip_first = embedding_format == "vec"
    embedding_dict = None
    with open(embedding_path) as f:
      for i, line in enumerate(f.readlines()):
        if skip_first and i == 0:
          continue
        splits = line.split()
        word = splits[0]
        vector = splits[1:]

        if not embedding_dict:
          embedding_size = len(vector)
          default_embedding = np.zeros(embedding_size)
          embedding_dict = collections.defaultdict(lambda:default_embedding)

        assert len(splits) == embedding_size + 1
        embedding = np.array([float(s) for s in vector])
        embedding_dict[word] = embedding
      sys.stderr.write("Done loading word embeddings.\n")
    return embedding_dict


