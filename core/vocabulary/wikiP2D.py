#coding: utf-8
import os, time, re, sys
from collections import defaultdict, OrderedDict, Counter
import core.utils.common as common
from core.vocabulary.base import ERROR_ID, PAD_ID, BOS_ID, EOS_ID, UNK_ID, _PAD, _BOS, _EOS, _UNK 
from core.vocabulary.base import VocabularyBase

_DIGIT_RE = re.compile(r"\d")

# todo: process special_words (e.g. @entity0)
def word_tokenizer(special_words=None, lowercase=False, normalize_digits=False):
  def _tokenizer(sent):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    return sent.replace('\n', '').split()
  return _tokenizer

def char_tokenizer(special_words=None, lowercase=False, normalize_digits=False):
  def _tokenizer(sent):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    def word2chars(word):
      if not special_words or word not in special_words:
        return [c.encode('utf-8') for c in word.decode('utf-8')]
      return word
    words = sent.replace('\n', '').split()
    chars = [word2chars(w) for w in words]
    return chars
  return _tokenizer

class WikiP2DVocabulary(VocabularyBase):
  def __init__(self, sentences, vocab_path, vocab_size,
               cbase=False, lowercase=False, special_words=None,
               normalize_digits=False):
    if cbase:
      self.tokenizer = char_tokenizer(special_words=special_words, 
                                      lowercase=lowercase,
                                      normalize_digits=normalize_digits) 
    else: 
      self.tokenizer = word_tokenizer(special_words=special_words, 
                                      lowercase=lowercase,
                                      normalize_digits=normalize_digits) 

    self.cbase = cbase
    self.vocab, self.rev_vocab = self.init_vocab(sentences, vocab_path, vocab_size)
    self.size = len(self.vocab)

  def init_vocab(self, sentences, vocab_path, vocab_size):
    if os.path.exists(vocab_path):
      return self.load_vocab(vocab_path)
    elif not sentences:
      raise ValueError('This vocabulary does not exist and no sentences were passed when initializing.')

    START_VOCAB = [(_PAD, 0), (_BOS, 0), (_EOS, 0), (_UNK, 0) ]
    tokenized = common.flatten([self.tokenizer(s) for s in sentences])
    if isinstance(tokenized[0], list):
      tokenized = common.flatten(tokenized)
    tokens = Counter(tokenized)
    tokens = sorted([(k, f) for (k, f) in tokens.items()], key=lambda x: -x[1])

    rev_vocab = [k for k, _ in START_VOCAB + tokens[:(vocab_size - len(START_VOCAB))]]
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})

    START_VOCAB[UNK_ID] = (_UNK, sum([f for _, f in tokens[vocab_size:]]))
    START_VOCAB[BOS_ID] = (_BOS, len(sentences))
    START_VOCAB[EOS_ID] = (_EOS, len(sentences))
    vocab_with_freq = START_VOCAB + tokens[:(vocab_size - len(START_VOCAB))]
    self.save_vocab(vocab_with_freq, vocab_path)
    return vocab, rev_vocab

  def load_vocab(self, vocab_path):
    sys.stderr.write('Loading vocabulary from \'%s\' ...\n' % vocab_path)
    rev_vocab = [l.split('\t')[0] for l in open(vocab_path, 'r')]
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    return vocab, rev_vocab

  def save_vocab(self, vocab_with_freq, vocab_path):
    with open(vocab_path, 'w') as f:
      f.write('\n'.join(["%s\t%d"% tuple(x) for x in vocab_with_freq]) + '\n')

  def sent2ids(self, sentence):
    tokens = self.tokenizer(sentence)
    if self.cbase:
      res = [[self.vocab.get(char, UNK_ID) for char in word] for word in tokens]
      res.append([EOS_ID])
      #res.insert(0, [BOS_ID]) # todo link_spanをどうする？
    else:
      res = [self.vocab.get(word, UNK_ID) for word in tokens]
      res.append(EOS_ID)
      #res.insert(0, BOS_ID)
    return res
  def id2token(self, _id):
    if _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
    else:
      return self.rev_vocab[_id]

  def ids2tokens(self, ids):
    if self.cbase:
      if type(ids[0]) == int:
        sent = "".join([self.rev_vocab[char_id] for char_id in ids])
      else:
        sent = " ".join(["".join([self.rev_vocab[char_id] for char_id in word]) for word in ids])
    else:
      sent = " ".join([self.rev_vocab[word_id] for word_id in ids])
    return sent

class WikiP2DRelVocabulary(WikiP2DVocabulary):
  def __init__(self, data, vocab_path, vocab_size=None):
    '''
    data : Ordereddict[Pid] = {'name': str, 'freq': int, 'aka': set, 'desc': str}
    '''
    if os.path.exists(vocab_path) or not data:
      self.vocab, self.rev_vocab = self.load_vocab(vocab_path)
    else:
      vocab_with_freq = sorted([(k,v['freq'])for k, v in data.items()],
                               key=lambda x:-x[1])
      self.rev_vocab = [k for k,_ in vocab_with_freq]
      if vocab_size:
        self.rev_vocab = self.rev_vocab[:vocab_size]
      self.vocab = OrderedDict({t:i for i,t in enumerate(self.rev_vocab)})
      self.save_vocab(vocab_with_freq, vocab_path)
    self.size = len(self.vocab)

  def to_ids(self, token):
    return self.vocab.get(token, ERROR_ID)

  def to_tokens(self, _id):
    return self.rev_vocab[_id]

class WikiP2DObjVocabulary(WikiP2DRelVocabulary):
  pass


