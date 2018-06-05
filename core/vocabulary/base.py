#coding: utf-8
import collections, os, time, re, sys, math
from tensorflow.python.platform import gfile
from orderedset import OrderedSet
import core.utils.common as common
import numpy as np
#from sklearn.preprocessing import normalize

_PAD = "_PAD"
_BOS = "_BOS"
#_EOS = "_EOS"
_UNK = "_UNK"

START_VOCAB = [_PAD, _BOS, _UNK]
PAD_ID = START_VOCAB.index(_PAD) # PAD_ID must be 0 for sequence length counting.
BOS_ID = START_VOCAB.index(_BOS)
#EOS_ID = START_VOCAB.index(_EOS)
UNK_ID = START_VOCAB.index(_UNK)

_DIGIT_RE = re.compile(r"\d")

def format_zen_han(l):
  import mojimoji
  l = l.decode('utf-8') if type(l) == str else l
  l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
  l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
  l = l.encode('utf-8')
  return l

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

def char_tokenizer(special_words=START_VOCAB, 
                   lowercase=False, normalize_digits=False):
  def _tokenizer(sent, flatten=False):
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
    def word2chars(word):
      return [c for c in word]
    words = sent.replace('\n', '').split()
    chars = [word2chars(w) for w in words]
    if flatten:
      chars = common.flatten(chars)
    return chars
  return _tokenizer

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



def word_sent_padding(inputs, max_sent_len=None):
  _max_sent_len = max([len(sent) for sent in inputs])
  if not max_sent_len or _max_sent_len < max_sent_len:
    max_sent_len = _max_sent_len

  padded_sentences = tf.keras.preprocessing.sequence.pad_sequences(
    inputs, maxlen=max_sent_len, padding='post', truncating='post', value=PAD_ID)
  return padded_sentences # [batch_size, max_sent_len]

def char_sent_padding(inputs, 
                      max_sent_len=None, max_word_len=None):

  _max_sent_len = max([len(sent) for sent in inputs])
  if not max_sent_len or _max_sent_len < max_sent_len:
    max_sent_len = _max_sent_len

  _max_word_len = max([max([len(word) for word in sent]) for sent in inputs])
  if not max_word_len or _max_word_len < max_word_len:
    max_word_len = _max_word_len

  # Because of the maximum window width of CNN.
  if max_word_len < 5:
    max_word_len = 5

  padded_sentences = [fill_empty_brackets(sent, max_sent_len) for sent in inputs] 
  padded_sentences = [tf.keras.preprocessing.sequence.pad_sequences(sent, maxlen=max_word_len, padding='post', truncating='post', value=PAD_ID) for sent in inputs]
  return padded_sentences # [batch_size, max_sent_len, max_word_len]

def random_embedding_generator(embedding_size):
  return lambda: np.random.uniform(-math.sqrt(3), math.sqrt(3), 
                                   size=embedding_size)
def zero_embedding_generator(embedding_size):
  return lambda: np.array([0.0 for _ in range(embedding_size)])


class VocabularyBase(object):
  def __init__(self):
    self.vocab = None
    self.rev_vocab = None
    self.name = None

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
    if type(_id) not in [int, np.int32, np.int64] or _id < 0 or _id > len(self.rev_vocab):
      sys.stderr.write(type(_id))
      raise ValueError('Token ID must be an integer between 0 and %d (ID=%d)' % (len(self.rev_vocab), _id))
    elif _id in set([PAD_ID, BOS_ID]):
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
        for i in range(link_span[0], link_span[1]+1):
          sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      sent_tokens = [w for w in sent_tokens if w]
      return " ".join(sent_tokens)
    return _ids2tokens(ids, link_span)

  def token2id(self, token):
    return self.vocab.get(token, UNK_ID)

  def sent2ids(self, sentence, word_dropout=0.0):
    if type(sentence) == list:
      assert type(sentence[0]) == str
      sentence = " ".join(sentence)
    tokens = self.tokenizer(sentence) 
    if word_dropout:
      res = [self.token2id(word) if np.random.rand() <= word_dropout else UNK_ID
             for word in tokens]
    else:
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
    return list(map(list, list(zip(*res))))


class CharVocabularyBase(VocabularyBase):
  def id2token(self, _id):
    if _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
    elif _id in set([PAD_ID, BOS_ID]):
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
      for i in range(link_span[0], link_span[1]+1):
        sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      sent_tokens = [w for w in sent_tokens if w]
    return " ".join(sent_tokens)

class VocabularyWithEmbedding(WordVocabularyBase):
  def __init__(self, emb_configs, vocab_size,
               lowercase=False, normalize_digits=False, 
               normalize_embedding=True):
    '''
    All pretrained embeddings must be under the source_dir.'
    This class can merge two or more pretrained embeddings by concatenating both.
    For OOV word, this returns zero vector.
    '''
    super(VocabularyWithEmbedding, self).__init__()
    self.start_vocab = START_VOCAB
    self.name = "_".join([c['path'].split('.')[0] for c in emb_configs])
    self.tokenizer = word_tokenizer(lowercase=lowercase,
                                    normalize_digits=normalize_digits)
    self.normalize_embedding = normalize_embedding
    self.vocab, self.rev_vocab, self.embeddings = self.init_vocab(
      emb_configs, vocab_size)

  def init_vocab(self, emb_configs, vocab_size):
    # Load several pretrained embeddings and concatenate them.
    pretrained = [self.load(c['path'], vocab_size, c['size'], c['skip_first'])
                  for c in emb_configs]
    rev_vocab = common.flatten([list(e.keys()) for e in pretrained])
    rev_vocab = self.start_vocab + rev_vocab[:vocab_size]
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i

    # Merge pretrained embeddings.
    if self.normalize_embedding:
      # Normalize the pretrained embeddings for each of the embedding types.
      embeddings = [common.flatten([common.normalize_vector(emb[w]) for emb in pretrained]) for w in vocab]
    else:
      embeddings = [common.flatten([emb[w] for emb in pretrained]) for w in vocab]

    # tokens in START_VOCAB are randomly initialized.
    #rand_gen = random_embedding_generator(len(embeddings[0]))
    #for i in range(len(self.start_vocab)):
    #  embeddings[i] = rand_gen()

    embeddings = np.array(embeddings)
    sys.stderr.write("Done loading word embeddings.\n")

    return vocab, rev_vocab, embeddings

  def load(self, embedding_path, vocab_size, embedding_size, skip_first):
    """
    Load pretrained vocabularies.
    Args:
    - embedding_path: a string.
    - vocab_size: an integer.
    - embedding_size: an integer.
    """
    sys.stderr.write("Loading word embeddings from {}...\n".format(embedding_path))
    # TODO: embeddingも訓練するならランダム初期化で良いけど、くっつけるならゼロ初期化のほうがいいんじゃないか？
    #embedding_dict = collections.defaultdict(random_embedding_generator(embedding_size))
    embedding_dict = collections.defaultdict(zero_embedding_generator(embedding_size))

    with open(embedding_path) as f:
      for i, line in enumerate(f.readlines()):
        if skip_first and i == 0:
          continue
        if len(embedding_dict) >= vocab_size:
          break

        word_and_emb = line.split()
        word = self.tokenizer(word_and_emb[0])
        if len(word) > 1:
          continue
        word = word[0]
        vector = [float(s) for s in word_and_emb[1:]]

        # If a capitalized (or digit-normalized) word is changed to its lowercased, it is used as an alternative only when the exact one is not still registered. 
        # e.g. Texas -> texas, 1999->0000, etc.
        if word == word_and_emb:
          embedding_dict[word] = vector
        elif word not in embedding_dict: 
          embedding_dict[word] = vector
    return embedding_dict


class PredefinedCharVocab(CharVocabularyBase):
  def __init__(self, vocab_path, vocab_size, 
               lowercase=False, normalize_digits=False):
    super(PredefinedCharVocab, self).__init__()
    self.start_vocab = START_VOCAB
    self.tokenizer = char_tokenizer(lowercase=lowercase,
                                    normalize_digits=normalize_digits)
    self.vocab, self.rev_vocab = self.init_vocab(vocab_path, vocab_size)

  def init_vocab(self, vocab_path, vocab_size):
    with open(vocab_path) as f:
      rev_vocab = [l.split()[0] for i, l in enumerate(f) if i < vocab_size]
      sys.stderr.write("Done loading the vocabulary.\n")
    rev_vocab = self.start_vocab + rev_vocab
    vocab = collections.OrderedDict({t:i for i,t in enumerate(rev_vocab)})
    return vocab, rev_vocab
