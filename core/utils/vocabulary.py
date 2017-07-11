#coding: utf-8
import collections, os, re, subprocess, time
from nltk.corpus import wordnet as wn
from subprocess import Popen, PIPE, STDOUT
import core.utils.common as common

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def space_tokenizer(sent, do_format_zen_han=True, do_separate_numbers=True):
  if do_format_zen_han:
    sent = format_zen_han(sent)
  if do_separate_numbers:
    sent = separate_numbers(sent)
  return sent.replace('\n', '').split()


class VocabularyBase(object):
  def __init__(self):
    raise NotImplementedError

  def to_tokens(self, ids):
    raise NotImplementedError

  def to_ids(self, tokens):
    raise NotImplementedError

  def load_vocabulary(self, vocabulary_path):
    raise NotImplementedError


class Vocabulary(VocabularyBase):
  def __init__(self, source_dir, processed_dir, vocab_file, suffix, vocab_size):
    self.START_VOCAB = [_PAD, _GO, _EOS, _UNK]
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


class WordNetVocabularyBase(object):
  def __init__(self):
    self.embeddings = None

class WordNetSynsetVocabulary(WordNetVocabularyBase):
  def __init__(self, source_dir, processed_dir, vocab_file, size):
    super(WordNetSynsetVocabulary, self).__init__()
    self.source_path = os.path.join(source_dir, vocab_file)
    self.processed_path = os.path.join(processed_dir, vocab_file) + '.syn_vocab.bin'
    self.unk = _UNK
    self.unk_id = 0
    mids, synsets, definitions = self.create_vocabulary(self.source_path, self.processed_path, size)
    self.size = len(mids)
    self.mids = mids
    self.mid_to_id = {k:i for i, k in enumerate(self.mids)}
    self.synsets = synsets
    self.synset_to_id = {k:i for i, k in enumerate(self.synsets)}

  def to_id(self, key):
    if key in self.mid_to_id:
      return self.mid_to_id[key]
    elif key in self.synset_to_id:
      return self.synset_to_id[key]
    else:
      return self.unk_id

  def create_vocabulary(self, source_path, processed_path, size=None):
    def remove_example_sentence(sent):
      pattern = "(.+?)([;:]\s*``.+)"
      m = re.search(pattern, sent)
      if m :
        sent = m.group(1)
      return sent

    def format_synset_name(synset):
      pos_formats = {
        "NN": wn.NOUN,
        'VB': wn.VERB,
        "JJ": wn.ADJ,
        "RB": wn.ADV,
      }
      pattern = '__(.+)_(.+?)_([0-9]+)'
      m = re.match(pattern, synset)
      word = m.group(1)
      pos = pos_formats[m.group(2)]
      s_type = "%02d" % int(m.group(3))
      return "%s.%s.%s" % (word, pos, s_type)

    def _create_vocabulary():
      # triple(def) = (mid, synsets, definition)
      data = [l.replace('\n', '').split('\t') 
              for l in open(source_path)]
      mids = [x[0] for x in data]
      if size:
        mids = mids[:size]

      synsets = [format_synset_name(x[1]) for x in data][:len(mids)]
      definitions = "\n".join([x[2] for x in data][:len(mids)]) + "\n"

      tokenizer_cmd = [
        "java", 
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
      ]
      p = Popen(tokenizer_cmd, stdout=PIPE, stdin=PIPE, stderr=None)
      definitions = p.communicate(input=definitions)[0].split('\n')
      definitions = [remove_example_sentence(d) for d in definitions]

      mids.insert(self.unk_id, self.unk)
      synsets.insert(self.unk_id, self.unk)
      definitions.insert(self.unk_id, self.unk)
      data = [mids, synsets, definitions]
      return data
    return common.load_or_create(processed_path, _create_vocabulary)

class WordNetRelationVocabulary(WordNetVocabularyBase):
  def __init__(self, source_dir, processed_dir, vocab_file, size):
    super(WordNetRelationVocabulary, self).__init__()
    self.source_path = os.path.join(source_dir, vocab_file)
    self.processed_path = os.path.join(processed_dir, vocab_file) + '.rel_vocab.bin'
    self.unk = _UNK
    self.unk_id = 0

    self.relations = self.create_vocabulary(self.source_path, self.processed_path, size)
    self.relation_to_id = {k:i for i,k in enumerate(self.relations)}
    self.size = len(self.relations)

  def to_id(self, key):
    if key in self.relation_to_id:
      return self.relation_to_id[key]
    else:
      return self.unk_id

  def create_vocabulary(self, source_path, processed_path, size=None):
    def _create_vocabulary():
      data = [l.split()[1] for l in open(source_path)]
      data = list(set(data))
      data.insert(self.unk_id, self.unk)
      return data
    relations = common.load_or_create(processed_path, _create_vocabulary)
    if size:
      relations = data[:size]
    return relations
