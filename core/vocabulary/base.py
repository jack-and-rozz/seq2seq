#coding: utf-8
import collections, os, time, re, sys
from tensorflow.python.platform import gfile
from orderedset import OrderedSet
import core.utils.common as common
import numpy as np
#from sklearn.preprocessing import normalize

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
  import mojimoji
  l = l.decode('utf-8') if type(l) == str else l
  l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
  l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
  l = l.encode('utf-8')
  return l

# def separate_numbers(sent):
#   def addspace(m):
#     return ' ' + m.group(0) + ' '
#   return re.sub(_DIGIT_RE, addspace, sent).replace('  ', ' ')

# def space_tokenizer(sent, do_format_zen_han=True, 
#                     do_separate_numbers=True):
#   if do_format_zen_han:
#     sent = format_zen_han(sent)
#   if do_separate_numbers:
#     sent = separate_numbers(sent)
#   return sent.replace('\n', '').split()

def word_tokenizer(lowercase=False, normalize_digits=False):
  '''
  Args:
     - flatten: Not to be used (used only in char_tokenizer)
  '''
  def _tokenizer(sent, flatten=None): # Arg 'flatten' is not used in this func. 
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    return sent.replace('\n', '').split()
  return _tokenizer

def char_tokenizer(special_words=set([_PAD, _BOS, _UNK, _EOS]), 
                   lowercase=False, normalize_digits=False):
  def _tokenizer(sent, flatten=False):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    def word2chars(word):
      if not special_words or word not in special_words:
        if not type(word) == unicode:
          word = word.decode('utf-8')
        return [c.encode('utf-8') for c in word]
      return [word]
    words = sent.replace('\n', '').split()
    chars = [word2chars(w) for w in words]
    if flatten:
      chars = common.flatten(chars)
    return chars
  return _tokenizer


class VocabularyBase(object):
  def __init__(self, add_bos=False, add_eos=False):
    self.vocab = None
    self.rev_vocab = None
    self.name = None
    self.start_offset = [BOS_ID] if add_bos else []
    self.end_offset = [EOS_ID] if add_eos else []
    self.n_start_offset = len(self.start_offset)
    self.n_end_offset = len(self.end_offset)

  def load_vocab(self, vocab_path):
    raise NotImplementedError

  def save_vocab(self, vocab_with_freq, vocab_path):
    raise NotImplementedError

  def padding(self, sentences, max_sentence_length=None, max_word_length=None):
    raise NotImplementedError

  @property
  def size(self):
    return len(self.vocab)


class WordVocabularyBase(VocabularyBase):
  def id2token(self, _id):
    if _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
    elif _id in set([PAD_ID, EOS_ID, BOS_ID]):
      return None
    else:
      return self.rev_vocab[_id]

  def ids2tokens(self, ids, link_span=None):
    '''
    ids: a list of word-ids.
    link_span : a tuple of the indices between the start and the end of a link.
    '''
    def _ids2tokens(ids, link_span):
      sent_tokens = [self.id2token(word_id) for word_id in ids]
      if link_span:
        for i in xrange(link_span[0], link_span[1]+1):
          sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      sent_tokens = [w for w in sent_tokens if w]
      return " ".join(sent_tokens)
    return _ids2tokens(ids, link_span)

  def token2id(self, token):
    return self.vocab.get(token, UNK_ID)

  def sent2ids(self, sentence):
    if type(sentence) == list:
      sentence = " ".join(sentence)
    tokens = self.tokenizer(sentence) 
    res = [self.token2id(word) for word in tokens]
    return res

  def padding(self, sentences, max_sentence_length=None):
    if not max_sentence_length:
      max_sentence_length = max([len(s) for s in sentences])

    def _padding(sent):
      padded_s = self.start_offset + sent[:max_sentence_length] + self.end_offset
      size = len(padded_s)
      padded_s += [PAD_ID] * (max_sentence_length + self.n_start_offset + self.n_end_offset - size)
      return padded_s, size
    res = [_padding(s) for s in sentences]
    return map(list, zip(*res))


class CharVocabularyBase(VocabularyBase):
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
    res = [[self.token2id(char) for char in word] for word in tokens]
    return res

  def ids2tokens(self, ids, link_span=None):
    sent_tokens = ["".join([self.id2token(char_id) for char_id in word]) 
                   for word in ids]
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
    if not max_word_length:
      max_word_length = max([max([len(w) for w in s]) for s in sentences])

    def _padding(sentences, max_s_length, max_w_length):
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
    return _padding(sentences, max_sentence_length, max_word_length)


class VocabularyWithEmbedding(WordVocabularyBase):
  def __init__(self, emb_configs, source_dir="dataset/embeddings",
               lowercase=False, normalize_digits=False, 
               normalize_embedding=True,
               add_bos=False, add_eos=False):
    '''
    All pretrained embeddings must be under the source_dir.'
    This class can merge two or more pretrained embeddings by concatenating both.
    For OOV word, this returns zero vector.
    '''
    super(VocabularyWithEmbedding, self).__init__(add_bos=add_bos, add_eos=add_eos)
    self.name = "_".join([c['path'].split('.')[0] for c in emb_configs])
    self.tokenizer = word_tokenizer(lowercase=lowercase,
                                    normalize_digits=normalize_digits)
    self.normalize_embedding = normalize_embedding
    #embedding_path = [os.path.join(source_dir, c['path']) for c in emb_configs]
    self.vocab, self.rev_vocab, self.embeddings = self.init_vocab(
      emb_configs, source_dir)

  def init_vocab(self, emb_configs, source_dir):
    '''
    Todo: Separately implement the two class, that of using only one type of embedding and using multi-embeddings.
    '''
    pretrained = [self.load(os.path.join(source_dir, c['path']), c['format']) 
                  for c in emb_configs]
    rev_vocab = common.flatten([e.keys() for e in pretrained])
    START_VOCAB = [_PAD, _BOS, _EOS, _UNK]
    rev_vocab = OrderedSet(START_VOCAB + [self.tokenizer(w, flatten=True)[0] 
                                          for w in rev_vocab])
    
    #vocab = collections.OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i

    w = 'this'
    w_id = vocab[w]
    print w + '(original)', w_id 
    print np.array(common.flatten([emb[w] for emb in pretrained]))
    
    # Merge pretrained embeddings and allocate zero vectors to START_VOCAB.
    if self.normalize_embedding:
      # Normalize the pretrained embeddings for each of the embedding types.
      embeddings = [common.flatten([common.normalize_vector(emb[w]) for emb in pretrained]) for w in vocab]
    else:
      embeddings = [common.flatten([emb[w] for emb in pretrained]) for w in vocab]
    embeddings = np.array(embeddings)
    return vocab, rev_vocab, embeddings

  def load(self, embedding_path, embedding_format):
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
          #default_embedding = np.zeros(embedding_size)
          default_embedding = [0.0 for _ in xrange(embedding_size)]
          embedding_dict = collections.defaultdict(lambda:default_embedding)

        assert len(splits) == embedding_size + 1
        embedding = [float(s) for s in vector]
        #embedding = np.array([float(s) for s in vector])
        embedding_dict[word] = embedding
      sys.stderr.write("Done loading word embeddings.\n")
    return embedding_dict


class PredefinedVocab(VocabularyBase):
  def init_vocab(self, embedding_path):
    rev_vocab = self.load(embedding_path)
    START_VOCAB = [_PAD, _BOS, _EOS, _UNK]
    rev_vocab = OrderedSet(START_VOCAB + [self.tokenizer(w, flatten=True)[0] 
                                          for w in rev_vocab])

    vocab = collections.OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    return vocab, rev_vocab

  def load(self, embedding_path, embedding_format='txt'):
    sys.stderr.write("Loading predefined vocab from {}...\n".format(embedding_path))
    with open(embedding_path) as f:
      rev_vocab = [l.split()[0] for l in f]
      sys.stderr.write("Done loading the vocabulary.\n")
    return rev_vocab

  

class PredefinedCharVocab(CharVocabularyBase, PredefinedVocab):
  def __init__(self, embedding_path,
               lowercase=False, normalize_digits=False, 
               add_bos=False, add_eos=False):
    super(PredefinedCharVocab, self).__init__(add_bos=add_bos, add_eos=add_eos)
    self.tokenizer = char_tokenizer(lowercase=lowercase,
                                    normalize_digits=normalize_digits)
    self.vocab, self.rev_vocab = self.init_vocab(embedding_path)
