#coding: utf-8
from pprint import pprint
import os, time, re, sys
from collections import defaultdict, OrderedDict, Counter
import core.utils.common as common
from core.vocabulary.base import ERROR_ID, PAD_ID, BOS_ID, EOS_ID, UNK_ID, _PAD, _BOS, _EOS, _UNK, word_tokenizer, char_tokenizer
from core.vocabulary.base import WordVocabularyBase, CharVocabularyBase, VocabularyBase

class WikiP2DVocabularyBase(VocabularyBase):
  def __init__(self, sentences, vocab_path, vocab_size,
               cbase=False, lowercase=False, special_words=None,
               normalize_digits=False, add_bos=False, add_eos=False):
    super(WikiP2DVocabularyBase, self).__init__(add_bos=add_bos, add_eos=add_eos)


class WikiP2DVocabulary(VocabularyBase):
  def __init__(self, sentences, vocab_path, vocab_size,
               cbase=False, lowercase=False, special_words=None,
               normalize_digits=False, add_bos=False, add_eos=False):
    self.cbase = cbase
    if self.cbase:
      self.tokenizer = char_tokenizer(special_words=special_words, 
                                      lowercase=lowercase,
                                      normalize_digits=normalize_digits) 
    else: 
      self.tokenizer = word_tokenizer(lowercase=lowercase,
                                      normalize_digits=normalize_digits) 

    self.vocab, self.rev_vocab = self.init_vocab(sentences, vocab_path, vocab_size)
    self.start_offset = [BOS_ID] if add_bos else []
    self.end_offset = [EOS_ID] if add_eos else []
    self.n_start_offset = len(self.start_offset)
    self.n_end_offset = len(self.end_offset)

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
    restored_data = START_VOCAB + tokens[:(vocab_size - len(START_VOCAB))]
    self.save_vocab(restored_data, vocab_path)
    return vocab, rev_vocab

  def load_vocab(self, vocab_path):
    sys.stderr.write('Loading vocabulary from \'%s\' ...\n' % vocab_path)
    rev_vocab = [l.split('\t')[0] for l in open(vocab_path, 'r')]
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    return vocab, rev_vocab

  def save_vocab(self, restored_data, vocab_path):
    with open(vocab_path, 'w') as f:
      f.write('\n'.join(["%s\t%d"% tuple(x) for x in restored_data]) + '\n')

  def id2token(self, _id):
    if _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
    elif _id in set([PAD_ID, EOS_ID, BOS_ID]):
      return ''
    else:
      return self.rev_vocab[_id]

  def token2id(self, token):
    return self.vocab.get(token, UNK_ID)

  def sent2ids(self, sentence):
    if type(sentence) == list:
      sentence = " ".join(sentence)
    tokens = self.tokenizer(sentence) 
    if self.cbase:
      res = [[self.token2id(char) for char in word] for word in tokens]
    else:
      res = [self.token2id(word) for word in tokens]
    return res

  def ids2tokens(self, ids, link_span=None):
    if self.cbase:
      sent_tokens = ["".join([self.id2token(char_id) for char_id in word]) 
                     for word in ids]
    else:
      sent_tokens = [self.id2token(word_id) for word_id in ids]
    if link_span:
      for i in xrange(link_span[0], link_span[1]+1):
        sent_tokens[i] = common.colored(sent_tokens[i], 'link')
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
        padded_s = self.start_offset + sent[:max_s_length] + self.end_offset
        size = len(padded_s)
        padded_s += [PAD_ID] * (max_s_length + self.n_start_offset + self.n_end_offset - size)
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
        if self.start_offset:
          padded_s.insert(0, self.start_offset + [PAD_ID] * (max_w_length - len(self.start_offset)))
          word_lengthes.insert(0, 1)
        if self.end_offset:
          padded_s.append(self.end_offset + [PAD_ID] * (max_w_length-len(self.end_offset)))
          word_lengthes.extend([1]+[0] * (max_s_length + self.n_start_offset + self.n_end_offset - sentence_length))
        sentence_length = len(padded_s)
        padded_s += [[PAD_ID] * max_w_length] * (max_s_length + self.n_start_offset + self.n_end_offset - sentence_length)
        return padded_s, sentence_length, word_lengthes
      res = [s_pad(s) for s in sentences]
      return map(list, zip(*res))
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
      self.vocab, self.rev_vocab, self.rev_names = self.load_vocab(vocab_path)
    else:
      restored_data = sorted([(k, v['freq'], v['name']) for k, v in data.items()], key=lambda x:-x[1])
      self.rev_vocab = [x[0] for x in restored_data]
      self.rev_names = [x[2] for x in restored_data]
      if vocab_size:
        self.rev_vocab = self.rev_vocab[:vocab_size]
        self.rev_names = self.rev_names[:vocab_size]
      self.vocab = OrderedDict({t:i for i,t in enumerate(self.rev_vocab)})
      self.save_vocab(restored_data, vocab_path)
    self.names = OrderedDict([(self.id2name(_id), _id) for _id in xrange(len(self.rev_names))])

  def token2id(self, token):
    return self.vocab.get(token, ERROR_ID)

  def name2id(self, name):
    return self.names.get(name, ERROR_ID)

  def id2token(self, _id):
    return self.rev_vocab[_id]

  def id2name(self, _id):
    return self.rev_names[_id] #self.data[self.id2token(_id)]['name']

  def save_vocab(self, restored_data, vocab_path):
    with open(vocab_path, 'w') as f:
      txt = ["%s\t%d\t%s" % x for x in restored_data]
      f.write('\n'.join(txt) + '\n')

  def load_vocab(self, vocab_path):
    sys.stderr.write('Loading vocabulary from \'%s\' ...\n' % vocab_path)
    data = [l.replace('\n', '').split('\t') for l in open(vocab_path, 'r')]
    rev_vocab = [x[0] for x in data]
    vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    rev_names = [x[2] for x in data]
    return vocab, rev_vocab, rev_names

class WikiP2DObjVocabulary(WikiP2DRelVocabulary):
  pass

class WikiP2DSubjVocabulary(WikiP2DRelVocabulary):
  pass


