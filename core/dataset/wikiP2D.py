#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from core.models.wikiP2D.category.evaluation import decorate_text

from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict, read_jsonlines
from core.utils.common import RED, BLUE, RESET, UNDERLINE, BOLD, GREEN, MAGENTA, CYAN, colored
#from core.utils import visualize
from core.dataset.base import DatasetBase, padding
from core.vocabulary.base import _UNK, UNK_ID, PAD_ID, VocabularyWithEmbedding, FeatureVocab
from core.vocabulary.wikiP2D import WikiP2DRelVocabulary #WikiP2DVocabulary, WikiP2DSubjVocabulary, WikiP2DRelVocabulary, WikiP2DObjVocabulary

random.seed(0)

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
    batch.text.word = padding(
       batch.text.word, 
       minlen=[self.config.minlen.word],
       maxlen=[self.config.maxlen.word])

    batch.text.char = padding(
      batch.text.char, 
      minlen=[self.config.minlen.word, self.config.minlen.char],
      maxlen=[self.config.maxlen.word, self.config.maxlen.char])

    cnn_max_filter_width = 3
    batch.rel.word = padding(batch.rel.word, 
                                minlen=[cnn_max_filter_width], 
                                maxlen=[None])
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
    self.iterations_per_epoch = int(config.iterations_per_epoch)

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

    # Articles which contain triples less than self.min_triples are discarded since they can be incorrect.
    if len(article.triples.subjective.ids) + len(article.triples.objective.ids) < self.min_triples:
      return []
    entry.mentions.raw = []
    entry.mentions.flat_position = []

    for t_type in ['subjective', 'objective']:
      #entry.mentions[t_type].raw = []
      #entry.mentions[t_type].flat_position = []

      entry.triples[t_type]= []
      entry.target[t_type] =[[self.vocab.rel.UNK_ID for j in range(self.max_mention_width)] for i in range(article.num_words)]

      for triple_idx, triple in enumerate(article.triples[t_type].ids): # triple = [subj, rel, obj]
        is_subjective = triple[0] == article.qid
        query_qid, rel_pid, mention_qid = triple if is_subjective else reversed(triple)
        # TODO: 同じメンションがクエリと異なる関係を持つ場合は？
        mention = qid2entity(mention_qid, article)
        #entry.mentions[t_type].raw.append(mention.raw)
        #entry.mentions[t_type].flat_position.append(mention.flat_position)
        entry.mentions.raw.append(mention.raw)
        entry.mentions.flat_position.append(mention.flat_position)

        rel = dotDict({'raw': rel_pid, 'name': self.vocab.rel.token2name(rel_pid)})

        begin, end = mention.flat_position
        if end - begin < self.max_mention_width:
          entry.target[t_type][begin][end-begin] = self.vocab.rel.token2id(rel_pid)

        triple = [entry.query, rel, mention] if is_subjective else [mention, rel, entry.query]
        entry.triples[t_type].append(triple)
      relation_freqs = Counter(flatten(entry.target[t_type]))
      entry.loss_weights_by_label[t_type] = [1.0/relation_freqs[i] if i in relation_freqs else 0.0 for i in range(self.vocab.rel.size)]
      #entry.num_mentions[t_type] = len(entry.mentions[t_type].positions)
    entry.num_mentions = len(entry.mentions.flat_position)
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
    # [batch_size, max_num_word_in_doc, max_mention_width]
    batch.target.subjective = padding(
      batch.target.subjective,
      minlen=[0, self.max_mention_width],
      maxlen=[0, self.max_mention_width]
    )
    # [batch_size, max_num_word_in_doc, max_mention_width]
    batch.target.objective = padding(
      batch.target.objective,
      minlen=[0, self.max_mention_width],
      maxlen=[0, self.max_mention_width]
    )

    # [batch_size, max_num_mentions_in_batch, max_mention_width]
    batch.mentions.flat_position = padding(
      batch.mentions.flat_position,
      minlen = [0, 2],
      maxlen = [0, 2],
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
    batch.desc.word = padding(
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

  @classmethod
  def get_str_triple(self_class, triple):
   s, r, o = triple 
   return (s.raw, r.name, o.raw)

  @classmethod
  def evaluate_mentions(self_class, mentions):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    gold_mentions = mentions.gold
    predicted_mentions = mentions.prediction
    for g, p in zip(gold_mentions, predicted_mentions):
      gold = set([tuple(m.flat_position) for m in g])
      pred = set([tuple(m.flat_position) for m in p])
      both = gold.intersection(pred)
      TP += len(both)
      FP += len(pred - both)
      FN += len(gold - both)
      
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = (precision+recall) / 2
    print ('<Mention Evaluation>')
    print ('P, R, F = %.3f, %.3f, %.3f' % (precision, recall, f1))
    return precision, recall, f1

  @classmethod
  def evaluate_triples(self_class, triples):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    gold_triples = triples.gold
    predicted_triples = triples.prediction
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
    print ('<Triple Evaluation>')
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
    triples = recDotDict({'gold': [], 'prediction': []})
    mentions = recDotDict({'gold': [], 'prediction': []})

    for i, (b, p) in enumerate(zip(flat_batches, predictions)):
      query = b.query
      gold_triples = b.triples
      predicted_triples = recDotDefaultDict()
      predicted_triples.subjective = []
      predicted_triples.objective = []

      gold_mentions = [recDotDict({'raw': r, 'flat_position':p }) for r, p in zip(b.mentions.raw, b.mentions.flat_position)]

      predicted_mentions = []
      for (subj_rel_id, obj_rel_id), (mention_start, mention_end) in zip(*p):
        if mention_end <= len(b.text.flat) and (mention_start, mention_end) != (PAD_ID, PAD_ID):
          mention = recDotDict()
          mention.raw = ' '.join(b.text.flat[mention_start:mention_end+1])
          mention.flat_position = (mention_start, mention_end)
          predicted_mentions.append(mention)
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
      triples.gold.append(gold_triples)
      triples.prediction.append(predicted_triples)
      mentions.gold.append(gold_mentions)
      mentions.prediction.append(predicted_mentions)
      _id = BOLD + '<%04d>' % (i) + RESET
      print (_id)
      self_class.print_example(
        b, vocab, prediction=[predicted_triples, predicted_mentions])
      print ('')
    return triples, mentions
    #return all_gold_triples, all_predicted_triples
  
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
      if PAD_ID in gold_mention_positions:
        gold_mention_positions.remove(PAD_ID)

      if i in query_positions:
        text[i] = MAGENTA + text[i]
      if i in gold_mention_positions:
        text[i] = BLUE + text[i]
      text[i] = text[i] + RESET
    return text #'\n'.join([' '.join(sent) for sent in text])

  @classmethod
  def print_example(self_class, example, vocab, prediction=None):
    SPACE = '  '
    def extract(position, text):
      return ' '.join(text[position[0]:position[1]+1])

    def print_mentions(mentions, text):
      mentions_str = ', '.join([extract(m.flat_position, text) for m in mentions if m.flat_position != (PAD_ID, PAD_ID)])
      print(SPACE + mentions_str)

    def print_triples(triples, text):
      for s, r, o in triples:
        triple_str = ', '.join([extract(s.flat_position, text), r.name, 
                                extract(o.flat_position, text)])
        print(SPACE + triple_str)
      if not triples:
        print()

    if example.title:
      print('<Title>', example.title.raw)
    decorated_text = self_class.decorate_text(example, vocab, prediction)
    print('<Text>')
    print(SPACE + ' '.join(decorated_text))
    print('<Triples (Query-subj)>')
    print_triples(example.triples.subjective, decorated_text)
    print('<Triples (Query-obj)>')
    print_triples(example.triples.objective, decorated_text)

    if prediction is not None:
      triples, mentions = prediction
      print('<Mention Candidates>')
      print_mentions(mentions, decorated_text)
      print('<Predictions (Query-subj)>')
      print_triples(triples.subjective, decorated_text)
      print('<Predictions (Query-obj)>')
      print_triples(triples.objective, decorated_text)
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
  
