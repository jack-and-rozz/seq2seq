#coding: utf-8
import os, time, re, sys
from collections import defaultdict, OrderedDict, Counter
import core.utils.common as common
from core.vocabulary.base import ERROR_ID, PAD_ID, GO_ID, EOS_ID, UNK_ID, _PAD, _GO, _EOS, _UNK 
from core.vocabulary.base import format_zen_han, VocabularyBase

_DIGIT_RE = re.compile(r"\d")

# todo: process special_words (e.g. @entity0)
def word_tokenizer(special_words=[], lowercase=False, normalize_digits=False):
  def _tokenizer(sent):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    return sent.replace('\n', '').split()
  return _tokenizer

def char_tokenizer(special_words=[], lowercase=False, normalize_digits=False):
  def _tokenizer(sent):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    def word2chars(word):
      if word not in special_words:
        return [c.encode('utf-8') for c in word.decode('utf-8')]
      return word
    words = sent.replace('\n', '').split()
    chars = [word2chars(w) for w in words]
    return chars
  return _tokenizer

class WikiP2DVocabulary(VocabularyBase):
  def __init__(self, sentences, vocab_path, vocab_size,
               cbase=False, lowercase=False, special_words=[],
               normalize_digits=False):
    lowercase=True
    normalize_digits=True
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
    self.vocab_size = len(self.vocab)

  def init_vocab(self, sentences, vocab_path, vocab_size):
    if os.path.exists(vocab_path):
      return self.load_vocab(vocab_path)
    elif not sentences:
      raise ValueError('This vocabulary does not exist and no sentences were passed when initializing.')

    #START_VOCAB = [_PAD, _GO, _EOS, _UNK ]
    START_VOCAB = [(_PAD, 0), (_GO, 0), (_EOS, 0), (_UNK, 0) ]
    tokenized = common.flatten([self.tokenizer(s) for s in sentences])
    if isinstance(tokenized[0], list):
      tokenized = common.flatten(tokenized)
    tokens = Counter(tokenized)
    tokens = sorted([(k, f) for (k, f) in tokens.items()], key=lambda x: -x[1])[:(vocab_size - len(START_VOCAB))]
    rev_vocab = START_VOCAB + tokens #[x[0] for x in tokens]
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    self.save_vocab(rev_vocab, vocab_path)
    return vocab, rev_vocab

  def load_vocab(self, vocab_path):
    sys.stderr.write('Loading vocabulary from \'%s\' ...\n' % vocab_path)
    rev_vocab = [(l.split('\t')[0], int(l.split('\t')[1])) for l in open(vocab_path, 'r')]
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    return vocab, rev_vocab

  def save_vocab(self, rev_vocab, vocab_path):
    with open(vocab_path, 'w') as f:
      f.write('\n'.join(["%s\t%d"% tuple(x) for x in rev_vocab]) + '\n')

  def sent2ids(self, sentence):
    tokens = self.tokenizer(sentence)
    if self.cbase:
      res = [[self.vocab.get(char, UNK_ID) for char in word] for word in tokens]
      res.append([EOS_ID])
      res.insert(0, [GO_ID])
    else:
      res = [self.vocab.get(word, UNK_ID) for word in tokens]
      res.append(EOS_ID)
      res.insert(0, GO_ID)
    return res

  def ids2sent(self, ids):
    # Todo
    return ids

class WikiP2DRelVocabulary(WikiP2DVocabulary):
  def __init__(self, data, vocab_path, vocab_size=None):
    '''
    data : Ordereddict[Pid] = {'name': str, 'freq': int, 'aka': set, 'desc': str}
    '''
    if os.path.exists(vocab_path) or not data:
      self.vocab, self.rev_vocab = self.load_vocab(vocab_path)
    else:
      self.rev_vocab = sorted([(k,v['freq'])for k, v in data.items()],
                              key=lambda x:-x[1])
      if vocab_size:
        self.rev_vocab = self.rev_vocab[:vocab_size]
      self.vocab = OrderedDict({t:i for i,t in enumerate(self.rev_vocab)})
      self.save_vocab(self.rev_vocab, vocab_path)
    self.size = len(self.vocab)

  def to_ids(self, token):
    return self.vocab.get(token, ERROR_ID)

  def to_tokens(self, _id):
    return self.rev_vocab[_id]

class WikiP2DObjVocabulary(WikiP2DRelVocabulary):
  pass


