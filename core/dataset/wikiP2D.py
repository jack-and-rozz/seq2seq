#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time, json
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from core.models.wikiP2D.category.evaluation import decorate_text

from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict
from core.utils.common import RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, MAGENTA, CYAN, colored
#from core.utils import visualize
from core.dataset.base import DatasetBase
from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, fill_empty_brackets, fill_zero, VocabularyWithEmbedding, FeatureVocab
from core.vocabulary.wikiP2D import WikiP2DRelVocabulary #WikiP2DVocabulary, WikiP2DSubjVocabulary, WikiP2DRelVocabulary, WikiP2DObjVocabulary

random.seed(0)

def define_length(batch, minlen=None, maxlen=None):
  if minlen is None:
    minlen = 0

  if maxlen:
    return max(maxlen, minlen) 
  else:
    return max([len(b) for b in batch] + [minlen])

def padding_2d(batch, minlen=None, maxlen=None, pad=PAD_ID, pad_type='post'):
  '''
  Args:
  batch: a 2D list. 
  maxlen: an integer.
  Return:
  A 2D tensor of which shape is [batch_size, max_num_word].
  '''
  if type(maxlen) == list:
    maxlen = maxlen[0]
  if type(minlen) == list:
    minlen = minlen[0]

  length_of_this_dim = define_length(batch, minlen, maxlen)
  return np.array([fill_zero(l[:length_of_this_dim], length_of_this_dim) for l in batch])

def padding(batch, minlen, maxlen):
  '''
  Args:
  - batch: A list of tensors with different shapes.
  - minlen, maxlen: A list of integers or None. Each i-th element specifies the minimum (or maximum) size of the tensor in the rank i+1.
    minlen[i] is considered as 0 if it is None, and maxlen[i] is automatically set to be equal to the maximum size of 'batch', the input tensor.
  
  e.g. 
  [[1], [2, 3], [4, 5, 6]] with minlen=[None], maxlen=[None] should be
  [[1, 0, 0], [2, 3, 0], [4, 5, 6]]
  '''
  assert len(minlen) == len(maxlen)
  rank = len(minlen) + 1
  length_of_this_dim = define_length(batch, minlen[0], maxlen[0])
  padded_batch = []
  if rank == 2:
    return padding_2d(batch, minlen=minlen[0], maxlen=maxlen[0])

  for l in batch:
    l = fill_empty_brackets(l[:length_of_this_dim], length_of_this_dim)
    if rank == 3:
      l = padding_2d(l, minlen=minlen[1:], maxlen=maxlen[1:])
    else:
      l = padding(l, minlen=minlen[1:], maxlen=maxlen[1:])

    padded_batch.append(l)
  largest_shapes = [max(n_dims) for n_dims in zip(*[tensor.shape for tensor in padded_batch])]
  target_tensor = np.zeros([len(batch)] + largest_shapes)

  for i, tensor in enumerate(padded_batch):
    pad_lengths = [x - y for x, y in zip(largest_shapes, tensor.shape)]
    pad_shape = [(0, l) for l in pad_lengths] 
    padded_batch[i] = np.pad(tensor, pad_shape, 'constant')
  return np.array(padded_batch)

def read_jsonlines(source_path, max_rows=0):
  data = []
  for i, l in enumerate(open(source_path)):
    if max_rows and i >= max_rows:
      break
    d = recDotDict(json.loads(l))
    data.append(d)
  return data

def mask_span(raw_text, position, token=_UNK):
  assert type(raw_text) == list
  raw_text = copy.deepcopy(raw_text)
  begin, end = position
  for i in range(begin, end+1):
    raw_text[i] = token
  return raw_text


class _WikiP2DDataset(object):
  def __init__(self, config, filename, vocab, max_rows):
    '''
    Args:
    - config:
    - filename:
    - vocab:
    '''
    self.source_path = os.path.join(config.source_dir, filename)
    self.config = config
    self.vocab = vocab
    self.data = [] # Lazy loading.
    self.max_rows = max_rows

  def preprocess(self, article):
    raise NotImplementedError

  def article2entries(self, article):
    '''
    Args:
    - article: An instance of recDotDict.
    Return:
    A list of entry which is an instance of recDotDict.
    '''
    raise NotImplementedError

  def padding(self, batch):
    raise NotImplementedError

  @property
  def size(self):
    if len(self.data) == 0:
      self.load_data()
    return len(self.data)

  def load_data(self):
    sys.stderr.write("Loading wikiP2D dataset from \'%s\'... \n" % self.source_path)
    data = read_jsonlines(self.source_path, max_rows=self.max_rows)
    data = [self.preprocess(d) for d in data]
    self.data = flatten([self.article2entries(d) for d in data])

  def tensorize(self, data):
    batch = recDotDefaultDict()
    for d in data:
      batch = batching_dicts(batch, d) # list of dictionaries to dictionary of lists.
    batch = self.padding(batch)
    return batch

  def get_batch(self, batch_size, do_shuffle=False):
    if not self.data:
      self.load_data()

    if do_shuffle:
      random.shuffle(self.data)
      if hasattr(self, 'iterations_per_epoch') and self.iterations_per_epoch:
        data = self.data[:self.iterations_per_epoch * batch_size]
      else:
        data = self.data

    else:
      data = self.data

    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      yield batch


class _WikiP2DGraphDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, properties, mask_link):
    super().__init__(config, filename, vocab, max_rows)
    self.properties = properties
    self.mask_link = mask_link

  def preprocess(self, article):
    raw_text = [s.split() for s in article.text]
    num_words = [len(s) for s in raw_text]
    links = {}

    # Convert a list of sentneces to a flattened sequence of words.
    for qid, link in article.link.items():
      (sent_id, (begin, end)) = link
      flatten_begin = begin + sum(num_words[:sent_id])
      flatten_end = end + sum(num_words[:sent_id])
      assert flatten_begin >= 0 and flatten_end >= 0
      links[qid] = (flatten_begin, flatten_end)
    article.link = links
    article.text = flatten(raw_text)
    article.desc = article.desc.split()
    return article

  def article2entries(self, article):
    if not (article.text and article.positive_triple and article.negative_triple):
      return []

    def qid2position(qid, article):
      assert qid in article.link
      begin, end = article.link[qid]
      entity =  recDotDefaultDict()
      entity.raw  = article.text[begin:end+1] 
      entity.position = (begin, end)
    return entity

    def triple2entry(triple, article, label):
      entry = recDotDefaultDict()
      entry.qid = article.qid

      subj_qid, rel_pid, obj_qid = triple
      rel = self.properties[rel_pid].name.split()
      entry.rel.raw = rel  # 1D tensor of str. 
      entry.rel.word = self.vocab.word.sent2ids(rel) # 1D tensor of int.
      entry.rel.char = self.vocab.char.sent2ids(rel) # 2D tensor of int.
      entry.subj = qid2position(subj_qid, article) # (begin, end)
      entry.obj = qid2position(obj_qid, article)# (begin, end)
      entry.label = label # 1 or 0.

      entry.text.raw = article.text
      raw_text = article.text
      if self.mask_link:
        raw_text = mask_span(raw_text, entry.subj.position)
        raw_text = mask_span(raw_text, entry.obj.position)
      entry.text.word = self.vocab.word.sent2ids(raw_text)
      entry.text.char = self.vocab.char.sent2ids(raw_text)
      
      return entry

    positive = triple2entry(article.positive_triple, article, 1)
    negative = triple2entry(article.negative_triple, article, 0)
    return [positive, negative]

  def padding(self, batch):
    batch.text.word = padding_2d(
       batch.text.word, 
       minlen=self.config.minlen.word,
       maxlen=self.config.maxlen.word)

    batch.text.char = padding(
      batch.text.char, 
      minlen=[self.config.minlen.word, self.config.minlen.char],
      maxlen=[self.config.maxlen.word, self.config.maxlen.char])

    cnn_max_filter_width = 3
    batch.rel.word = padding_2d(batch.rel.word, 
                                minlen=cnn_max_filter_width, 
                                maxlen=None)
    batch.rel.char = padding(batch.rel.char, 
                             minlen=[3, self.config.minlen.char], 
                             maxlen=[None, self.config.maxlen.char])
    return batch

class _WikiP2DRelExDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, properties, mask_link):
    super().__init__(config, filename, vocab, max_rows)
    self.properties = properties
    self.mask_link = mask_link
    self.max_mention_width = config.max_mention_width
    self.min_triples = config.min_triples

  def preprocess(self, article):
    raw_text = [s.split() for s in article.text]
    num_words = [len(s) for s in raw_text]
    article.text = raw_text
    article.flat_text = flatten(raw_text)
    article.desc = article.desc.split()
    article.num_words = sum([len(s) for s in article.text])
    return article

  def article2entries(self, article):
    def qid2entity(qid, article):
      assert qid in article.link
      s_id, (begin, end) = article.link[qid]

      # The offset is the number of words in previous sentences. 
      offset = sum([len(sent) for sent in article.text[:s_id]])
      entity = recDotDefaultDict()
      # Replace entity's name with the actual representation in the article.
      entity.raw  = ' '.join(article.text[s_id][begin:end+1]) 
      entity.position = article.link[qid]
      entity.flat_position = (begin + offset, end + offset)
      return entity

    entry = recDotDefaultDict()
    entry.qid = article.qid

    entry.text.raw = article.text
    entry.text.flat = article.flat_text
    entry.text.word = [self.vocab.word.sent2ids(s) for s in article.text]
    entry.text.char = [self.vocab.char.sent2ids(s) for s in article.text]

    entry.query = qid2entity(article.qid, article) # (begin, end)
    entry.mentions.raw = []
    entry.mentions.flat_position = []

    # Articles which contain triples less than self.min_triples are discarded since they can be incorrect.
    if len(article.triples.subjective.ids) + len(article.triples.objective.ids) < self.min_triples:
      return []

    for t_type in ['subjective', 'objective']:
      entry.triples[t_type]= []
      entry.target[t_type] =[[self.vocab.rel.UNK_ID for j in range(self.max_mention_width)] for i in range(article.num_words)]

      for triple_idx, triple in enumerate(article.triples[t_type].ids): # triple = [subj, rel, obj]
        is_subjective = triple[0] == article.qid
        query_qid, rel_pid, mention_qid = triple if is_subjective else reversed(triple)
        # TODO: 同じメンションがクエリと異なる関係を持つ場合は？
        mention = qid2entity(mention_qid, article)
        entry.mentions.raw.append(mention.raw)
        entry.mentions.flat_position.append(mention.flat_position)

        rel = dotDict({'raw': rel_pid, 'name': self.vocab.rel.token2name(rel_pid)})

        begin, end = mention.flat_position
        if end - begin < self.max_mention_width:
          entry.target[t_type][begin][end-begin] = self.vocab.rel.token2id(rel_pid)

        triple = [entry.query, rel, mention] if is_subjective else [mention, rel, entry.query]
        entry.triples[t_type].append(triple)
          #entry.triples[t_type].raw[triple_idx]
    return [entry]

  def padding(self, batch):
    # [batch_size, max_num_sent, max_num_word_in_sent]
    batch.text.word = padding(
       batch.text.word, 
       minlen=[0, self.config.minlen.word],
       maxlen=[0, self.config.maxlen.word])

    # [batch_size, max_num_sent, max_num_word_in_sent, max_num_char_in_word]
    batch.text.char = padding( 
      batch.text.char, 
      minlen=[0, self.config.minlen.word, self.config.minlen.char],
      maxlen=[0, self.config.maxlen.word, self.config.maxlen.char])

    # [batch_size, 2]
    batch.query.flat_position = padding(
      batch.query.flat_position,
      minlen=[0],
      maxlen=[0]
    )
    # [bach_size, max_num_word_in_doc, max_mention_width]
    batch.target.subjective = padding(
      batch.target.subjective,
      minlen=[0, self.max_mention_width],
      maxlen=[0, self.max_mention_width]
    )
    batch.target.objective = padding(
      batch.target.objective,
      minlen=[0, self.max_mention_width],
      maxlen=[0, self.max_mention_width]
    )

    return batch

class _WikiP2DDescDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows):
    super().__init__(config, filename, vocab, max_rows)
    self.max_contexts = config.max_contexts

  def preprocess(self, article):
    return article

  def article2entries(self, article):
    if not (article.desc and article.contexts):
      return []

    entry = recDotDefaultDict()
    desc = article.desc.split()
    entry.title.raw = article.title
    entry.desc.raw = desc
    entry.desc.word = self.vocab.word.sent2ids(desc)
    entry.contexts.link = []
    entry.contexts.raw = []
    entry.contexts.word = []
    entry.contexts.char = []
    for context, link in article.contexts[:self.max_contexts]:
      entry.link.append(link)
      context = context.split()
      entry.contexts.raw.append(context)
      if self.mask_link:
        context = mask_span(context, link)
      entry.contexts.word.append(self.vocab.word.sent2ids(context))
      entry.contexts.char.append(self.vocab.char.sent2ids(context))
    return [entry]

  def padding(self, batch):
    '''
    batch.desc.word: [batch_size, max_words]
    batch.contexts.word: [batch_size, max_contexts, max_words]
    batch.contexts.char: [batch_size, max_contexts, max_words, max_chars]
    '''
    batch.contexts.char = padding(
      batch.contexts.char,
      minlen=[None, self.config.minlen.word, self.config.minlen.char],
      maxlen=[None, self.config.maxlen.word, self.config.maxlen.char])

    batch.contexts.word = padding(
      batch.contexts.word, 
      minlen=[None, self.config.minlen.word],
      maxlen=[None, self.config.maxlen.word])

    batch.desc.word = padding(
      batch.desc.word, 
      minlen=[self.config.minlen.word],
      maxlen=[self.config.maxlen.word])

    return batch

class _WikiP2DCategoryDataset(_WikiP2DDataset):
  def __init__(self, config, filename, vocab, max_rows, mask_link):
    super().__init__(config, filename, vocab, max_rows)
    self.max_contexts = config.max_contexts
    self.mask_link = mask_link
    self.iterations_per_epoch = int(config.iterations_per_epoch)
    self.data_by_category = None

  def preprocess(self, article):
    return article

  def article2entries(self, article):
    if not (article.category and article.contexts):
      return []

    entry = recDotDefaultDict()
    entry.title.raw = article.title
    desc = article.desc.split()
    entry.desc.raw = desc
    entry.desc.word = self.vocab.word.sent2ids(desc)

    entry.category.raw = article.category
    entry.category.label = self.vocab.category.token2id(article.category)
    if entry.category.label == self.vocab.category.token2id(_UNK):
      return []
    entry.contexts.raw = []
    entry.contexts.word = []
    entry.contexts.char = []
    entry.contexts.link = []
    for context, link in article.contexts[:self.max_contexts]:
      context = context.split()
      entry.contexts.raw.append(context)

      if self.mask_link:
        context = mask_span(context, link)
      entry.contexts.word.append(self.vocab.word.sent2ids(context))
      entry.contexts.char.append(self.vocab.char.sent2ids(context))
      entry.contexts.link.append(link)
    return [entry]

  def padding(self, batch):
    '''
    batch.contexts.word: [batch_size, max_contexts, max_words]
    batch.contexts.char: [batch_size, max_contexts, max_words, max_chars]
    batch.link: [batch_size, max_contexts, 2]
    '''
    batch.contexts.char = padding(
      batch.contexts.char,
      minlen=[None, self.config.minlen.word, self.config.minlen.char],
      maxlen=[None, self.config.maxlen.word, self.config.maxlen.char])
    batch.contexts.word = padding(
      batch.contexts.word, 
      minlen=[None, self.config.minlen.word],
      maxlen=[None, self.config.maxlen.word])
    batch.contexts.link = padding(
      batch.contexts.link,
      minlen=[None, 2],
      maxlen=[None, 2])
    batch.desc.word = padding_2d(
      batch.desc.word,
      minlen=[self.config.minlen.word],
      maxlen=[self.config.maxlen.word])
    return batch

  def get_batch(self, batch_size, do_shuffle=False):
    if not self.data:
      self.load_data()

    if do_shuffle:
      random.shuffle(self.data)
      if not self.data_by_category:
        self.data_by_category = defaultdict(list)
        for d in self.data:
          self.data_by_category[d.category.raw].append(d)
      data = [random.choice(random.choice(list(self.data_by_category.values()))) 
              for _ in range(self.iterations_per_epoch * batch_size)]
    else:
      data = self.data

    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      sliced_data = [x[1] for x in b] # (id, data) -> data
      batch = self.tensorize(sliced_data)
      yield batch


class WikiP2DGraphDataset(DatasetBase):
  '''
  A class which contains train, valid, testing datasets.
  '''
  dataset_class =  _WikiP2DGraphDataset
  def __init__(self, config, vocab, mask_link_in_test=True):
    self.vocab = vocab
    properties_path = os.path.join(config.source_dir, config.prop_data)
    self.properties = OrderedDict([(d['qid'], d) for d in read_jsonlines(properties_path)])
    self.vocab.rel = WikiP2DRelVocabulary(self.properties.values(), 
                                          start_vocab=[_UNK])
    self.train = self.dataset_class(config, config.filename.train, vocab,
                                    config.max_rows.train,
                                    self.properties, config.mask_link)

    self.valid = self.dataset_class(config, config.filename.valid, vocab, 
                                    config.max_rows.valid,
                                    self.properties, mask_link_in_test)
    self.test = self.dataset_class(config, config.filename.test, vocab, 
                                    config.max_rows.test,
                                   self.properties, mask_link_in_test)

class WikiP2DRelExDataset(WikiP2DGraphDataset):
  dataset_class =  _WikiP2DRelExDataset
  def __init__(self, config, vocab, mask_link_in_test=False):
    super().__init__(config, vocab, mask_link_in_test)
    self.iterations_per_epoch = int(config.iterations_per_epoch)

  @classmethod
  def get_str_triple(self_class, triple):
   s, r, o = triple 
   return (s.raw, r.name, o.raw)

  @classmethod
  def evaluate(self_class, gold_triples, predicted_triples):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    for g, p in zip(gold_triples, predicted_triples):
      gold = set([self_class.get_str_triple(t) for t in g.subjective + g.objective])
      pred = set([self_class.get_str_triple(t) for t in p.subjective + p.objective])
      both = gold.intersection(pred)
      TP += len(both)
      FP += len(pred - both)
      FN += len(gold - both)
      
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = (precision+recall) / 2
    print ('P, R, F = %.3f, %.3f, %.3f' % (precision, recall, f1))
    return precision, recall, f1

  @classmethod
  def formatize_and_print(self_class, flat_batches, predictions, vocab=None):
    '''
    Args:
    - predictions: A list of a tuple (relations, mention_starts, mention_ends), which contains the predicted relations (both of subj, obj) and mention spans. Each element of the list corresponds to each example.

    '''
    n_data = 0
    n_success = 0
    all_gold_triples = []
    all_predicted_triples = []
    for i, (b, p) in enumerate(zip(flat_batches, predictions)):
      query = b.query
      gold_triples = b.triples
      predicted_triples = recDotDefaultDict()
      predicted_triples.subjective = []
      predicted_triples.objective = []
      for (subj_rel_id, obj_rel_id), mention_start, mention_end in zip(*p):
        if mention_end <= len(b.text.flat):
          mention = recDotDict()
          mention.raw = ' '.join(b.text.flat[mention_start:mention_end+1])
          mention.flat_position = (mention_start, mention_end)
        else:
          continue
        if subj_rel_id != vocab.rel.UNK_ID:
          rel = dotDict({
            'raw' : vocab.rel.id2token(subj_rel_id),
            'name': vocab.rel.id2name(subj_rel_id),
          })
          predicted_triples.subjective.append(
            [query, rel, mention])
        if obj_rel_id != vocab.rel.UNK_ID:
          rel = dotDict({
            'raw' : vocab.rel.id2token(obj_rel_id),
            'name': vocab.rel.id2name(obj_rel_id),
          })
          predicted_triples.objective.append(
            [mention, rel, query])
      all_gold_triples.append(gold_triples)
      all_predicted_triples.append(predicted_triples)
      _id = '<%04d>' % (i)
      print (_id)
      self_class.print_example(b, vocab, prediction=predicted_triples)
      print ('')
    return all_gold_triples, all_predicted_triples

  @classmethod
  def decorate_text(self_class, example, vocab, prediction=None):
    '''
    Args:
    - example: A recDotDefaultDict, one example of a flattened batch.
    Refer to WikiP2DRelExDataset.article2entries. 
    '''
    text = copy.deepcopy(example.text.flat)
    query = example.query
    for i, w in enumerate(text):
      if vocab.word.is_unk(w):
        text[i] = UNDERLINE + text[i]
      query_positions = set([j for j in range(query.flat_position[0], 
                                              query.flat_position[1]+1)])
      gold_mention_positions = set(flatten([
        [j for j in range(begin, end+1)]
        for begin, end in example.mentions.flat_position]))

      if i in query_positions:
        text[i] = MAGENTA + text[i]
      if i in gold_mention_positions:
        text[i] = BLUE + text[i]
      text[i] = text[i] + RESET
    return text #'\n'.join([' '.join(sent) for sent in text])

  @classmethod
  def print_example(self_class, example, vocab, prediction=None):
    def extract(position, text):
      return ' '.join(text[position[0]:position[1]+1])

    def print_triples(triples, text):
      for s, r, o in triples:
        triple_str = ', '.join([extract(s.flat_position, text), r.name, 
                                extract(o.flat_position, text)])
        print(triple_str)
      if not triples:
        print()

    if example.title:
      print('<Title>', example.title.raw)
    decorated_text = self_class.decorate_text(example, vocab, prediction)
    print('<Text>')
    print(' '.join(decorated_text))
    print('<Triples (Query-subj)>')
    print_triples(example.triples.subjective, decorated_text)
    print('<Triples (Query-obj)>')
    print_triples(example.triples.objective, decorated_text)

    if prediction is not None:
      print('<Predictions (Query-subj)>')
      print_triples(prediction.subjective, decorated_text)
      print('<Predictions (Query-obj)>')
      print_triples(prediction.objective, decorated_text)
      pass

class WikiP2DDescDataset(DatasetBase):
  
  def __init__(self, config, vocab):
    self.vocab = vocab
    dataset_class =  _WikiP2DDescDataset
    self.train = self.dataset_class(config, config.filename.train, vocab, 
                                    config.max_rows.train)
    self.valid = self.dataset_class(config, config.filename.valid, vocab,
                                    config.max_rows.valid)
    self.test = self.dataset_class(config, config.filename.test, vocab,
                                   config.max_rows.test)

class WikiP2DCategoryDataset(DatasetBase):
  dataset_class =  _WikiP2DCategoryDataset
  def __init__(self, config, vocab):
    self.vocab = vocab

    #categories = [l.split()[0] for l in open(os.path.join(config.source_dir, config.category_vocab))][:config.category_size]
    self.vocab.category = VocabularyWithEmbedding(
      config.embeddings_conf, config.category_size, 
      start_vocab=[_UNK],
    )
    self.train = self.dataset_class(config, config.filename.train, vocab, 
                                    config.max_rows.train,
                                    config.mask_link)
    self.valid = self.dataset_class(config, config.filename.valid, vocab, 
                                    config.max_rows.valid, True)
    self.test = self.dataset_class(config, config.filename.test, vocab, 
                                   config.max_rows.test, True)
  
