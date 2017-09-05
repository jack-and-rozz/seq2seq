#coding: utf-8
import collections, os, time, re
from tensorflow.python.platform import gfile

from core.vocabulary.base import ERROR_ID, PAD_ID, BOS_ID, EOS_ID, UNK_ID, _PAD, _BOS, _EOS, _UNK 
from core.vocabulary.base import VocabularyBase, space_tokenizer,

class Vocabulary(VocabularyBase):
  def __init__(self, source_dir, processed_dir, vocab_file, suffix, vocab_size):
    self.START_VOCAB = [_PAD, _BOS, _EOS, _UNK]
    source_path = os.path.join(source_dir, vocab_file) + '.' + suffix
    target_path = os.path.join(processed_dir, vocab_file) + '.%s.Wvocab%d' %(suffix, vocab_size)
    self.tokenizer = space_tokenizer
    self.normalize_digits = False
    self.create_vocabulary(source_path, target_path, vocab_size)
    self.vocab, self.rev_vocab = self.load_vocabulary(target_path)
    self.suffix = suffix
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
      vocab_list = self.START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      for w in self.START_VOCAB:
        vocab[w] = 0
      n_unknown = sum([vocab[w] for w in vocab_list[max_vocabulary_size:]])
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      vocab[_UNK] = n_unknown
      vocab[_EOS] = counter
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write("%s\t%d\n" % (w, vocab[w]))

class VecVocabulary(Vocabulary):
  def __init__(self, source_dir, source_file, suffix, vocab_size, read_vec=True):
    source_path = os.path.join(source_dir, source_file)
    self.normalize_digits = False
    self.vocab, self.rev_vocab, self.embedding = self.load_vocabulary(source_path, vocab_size, read_vec)
    self.suffix = suffix
    self.size = vocab_size

  def create_vocabulary(self, data_path, vocabulary_path, max_vocabulary_size):
    raise NotImplementedError("The vocabulary type is intended to be pre-trained vectors and their keys.")

  def load_vocabulary(self, source_path, max_vocabulary_size, read_vec):
    vocab = collections.defaultdict(int)
    if gfile.Exists(source_path):
        rev_vocab = [] + self.START_VOCAB
        embedding = []
        counter = 0
        with gfile.GFile(source_path, mode="r") as f:
            for l in f:
              counter += 1
              if counter % 100000 == 0:
                print("  processing line %d" % counter)
              if read_vec:
                tokens = l.rstrip().split(' ')
                rev_vocab.append(tokens[0])
                embedding.append([float(v) for v in tokens[1:]])
              else:
                rev_vocab.append(l.split(' ', 1)[0])
              if counter + len(self.START_VOCAB) >= max_vocabulary_size:
                break
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        if read_vec:
          embedding = [[0] * len(embedding[-1])] * len(self.START_VOCAB) + embedding # prepend 4 zero vectors
        return vocab, rev_vocab, np.array(embedding)
    else:
        raise ValueError("Vector file %s not found.", source_path)


