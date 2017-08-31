#coding: utf-8
import collections, os, time
from nltk.corpus import wordnet as wn
from tensorflow.python.platform import gfile
import core.utils.common as common
from core.vocabulary.base import PAD_ID, GO_ID, EOS_ID, UNK_ID

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

    self.relations = self.create_vocabulary(self.source_path, 
                                            self.processed_path, size)
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
