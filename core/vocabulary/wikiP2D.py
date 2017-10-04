#coding: utf-8
import os, time, re, sys
from collections import defaultdict, OrderedDict, Counter
import core.utils.common as common
from core.vocabulary.base import ERROR_ID, PAD_ID, BOS_ID, EOS_ID, UNK_ID, _PAD, _BOS, _EOS, _UNK 
from core.vocabulary.base import VocabularyBase

_DIGIT_RE = re.compile(r"\d")

# TODO: process parentheses "()[]{}"
def word_tokenizer(lowercase=False, normalize_digits=False):
  def _tokenizer(sent):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    return sent.replace('\n', '').split()
  return _tokenizer

def char_tokenizer(special_words=set([_PAD, _BOS, _UNK, _EOS]), 
                   lowercase=False, normalize_digits=False):
  def _tokenizer(sent):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    def word2chars(word):
      if not special_words or word not in special_words:
        return [c.encode('utf-8') for c in word.decode('utf-8')]
      return [word]
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
      self.tokenizer = word_tokenizer(lowercase=lowercase,
                                      normalize_digits=normalize_digits) 

    self.cbase = cbase
    self.vocab, self.rev_vocab = self.init_vocab(sentences, vocab_path, vocab_size)
    self.size = len(self.vocab)

    # Number of additonal tokens (e.g. EOS) when articles are padded.
    # self.n_additional_paddings = 1

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
    else:
      res = [self.vocab.get(word, UNK_ID) for word in tokens]
    return res

  def id2token(self, _id):
    if _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
    elif _id in set([PAD_ID, EOS_ID, BOS_ID]):
      return ''
    else:
      return self.rev_vocab[_id]

  def ids2tokens(self, ids, link_span=None):
    if self.cbase:
      sent_tokens = ["".join([self.id2token(char_id) for char_id in word]) 
                     for word in ids]
    else:
      sent_tokens = [self.id2token(word_id) for word_id in ids]
    if link_span:
      for i in xrange(link_span[0], link_span[1]+1):
        sent_tokens[i] = common.colored(sent_tokens[i], 'bold')
      sent_tokens = [w for w in sent_tokens if w]
    return " ".join(sent_tokens)

  def padding(self, sentences, max_sentence_length=None, max_word_length=None):
    '''
    '''
    if not max_sentence_length:
      max_sentence_length = max([len(s) for s in sentences])
    if not max_word_length and self.cbase:
      max_word_length = max([max([len(w) for w in s]) for s in sentences])

    def wsent_padding(sentences, max_s_length):
      def w_pad(sent):
        padded_s = [BOS_ID] + sent[:max_s_length] + [EOS_ID]
        size = len(padded_s)
        padded_s += [PAD_ID] * (max_s_length+2 - size)
        return padded_s, size
      res = [w_pad(s) for s in sentences]
      return map(list, zip(*res))

    def csent_padding(sentences, max_s_length, max_w_length):
      def c_pad(w):
        padded_w = w[:max_w_length] 
        size = len(padded_w)
        padded_w += [PAD_ID] * (max_w_length - size)
        return padded_w, size
      def s_pad(s):
        s = s[:max_s_length]
        padded_s, word_lengthes = map(list, zip(*[c_pad(w) for w in s]))
        padded_s.insert(0, [BOS_ID] + [PAD_ID] * (max_w_length-1))
        padded_s.append([EOS_ID] + [PAD_ID] * (max_w_length-1))
        sentence_length = len(padded_s)
        padded_s += [[PAD_ID] * max_w_length] * (max_s_length+2 - sentence_length)
        word_lengthes.insert(0, 1)
        word_lengthes.extend([1]+[0] * (max_s_length+2 - sentence_length))
        return padded_s, sentence_length, word_lengthes
      res = [s_pad(s) for s in sentences]
      return map(list, zip(*res))

    sentence_lengthes =  [] # [batch_size]
    word_lengthes = [] # [batch_size, max_sentence_length]
    if self.cbase:
      return csent_padding(sentences, max_sentence_length, max_word_length)
    else:
      return wsent_padding(sentences, max_sentence_length)

class WikiP2DRelVocabulary(WikiP2DVocabulary):
  def __init__(self, data, vocab_path, vocab_size=None):
    '''
    data : Ordereddict[Pid] = {'name': str, 'freq': int, 'aka': set, 'desc': str}
    '''
    self.data = data
    if os.path.exists(vocab_path) or not data:
      self.vocab, self.rev_vocab = self.load_vocab(vocab_path)
    else:
      vocab_with_freq = sorted([(k, v['freq'])for k, v in data.items()],
                               key=lambda x:-x[1])
      self.rev_vocab = [k for k,_ in vocab_with_freq]
      if vocab_size:
        self.rev_vocab = self.rev_vocab[:vocab_size]
      self.vocab = OrderedDict({t:i for i,t in enumerate(self.rev_vocab)})
      self.save_vocab(vocab_with_freq, vocab_path)
    self.size = len(self.vocab)
    self.names = OrderedDict([(self.id2name(_id), _id) for _id in xrange(len(self.data))])

  def token2id(self, token):
    return self.vocab.get(token, ERROR_ID)

  def name2id(self, name):
    return self.names.get(name, ERROR_ID)

  def id2token(self, _id):
    return self.rev_vocab[_id]

  def id2name(self, _id):
    return self.data[self.id2token(_id)]['name']

  def save_vocab(self, vocab_with_freq, vocab_path):
    with open(vocab_path, 'w') as f:
      txt = ["%s\t%d\t%s" % (v, freq, self.data[v]['name']) 
             for (v, freq) in vocab_with_freq]
      f.write('\n'.join(txt) + '\n')


class WikiP2DObjVocabulary(WikiP2DRelVocabulary):
  pass

class WikiP2DSubjVocabulary(WikiP2DRelVocabulary):
  pass


