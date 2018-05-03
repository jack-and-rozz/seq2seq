#coding: utf-8

from pprint import pprint
import os, re, sys, random, copy, time
import subprocess, itertools
import numpy as np
from collections import OrderedDict, defaultdict, Counter

from tensorflow.python.platform import gfile
import core.utils.common as common
from core.utils import visualize
from core.dataset.base import DatasetBase
from core.vocabulary.wikiP2D import WikiP2DVocabulary, WikiP2DSubjVocabulary, WikiP2DRelVocabulary, WikiP2DObjVocabulary
try:
   import pickle as pickle
except:
   import pickle
random.seed(0)

def process_articles(articles, w_vocab, c_vocab):
  res = OrderedDict()
  for k, v in list(articles.items()):
    res[k] = [(w_vocab.sent2ids(sent), c_vocab.sent2ids(sent), s, e) for (sent, s, e) in v]
  return res

def process_triples(triples, r_vocab, o_vocab):
  res = OrderedDict()
  for k, v in list(triples.items()):
    res[k] = [(r_vocab.token2id(r), o_vocab.token2id(o)) for (r, o) in v]
  return res

def process_entities(entities, w_vocab, c_vocab):
  res = copy.deepcopy(entities)
  for k, v in list(entities.items()):
    res[k]['qid'] = k
    res[k]['w_desc'] = w_vocab.sent2ids(v['desc'])
    res[k]['c_desc'] = c_vocab.sent2ids(v['desc'])
  return res


####################################
#           WikiP2D
####################################

class DemoBatch(object):
  def __init__(self, batch):
    self.batch = batch

  def get_batch(self, batch_size, do_shuffle=False, 
                n_batches=1, n_pos_triples=None, n_neg_triples=1,
                carefully_negative=False, 
                min_sentence_length=None, max_sentence_length=None):
    yield self.batch

class _WikiP2DDataset(object):
  def __init__(self, articles, triples, subjects, relations, objects,
               all_triples=None):
    self.articles = articles # article = [w_article, c_article, l_start, l_end]
    self.triples = triples
    self.subjects = subjects
    self.relations = relations
    self.objects = objects
    self.all_triples = all_triples # Not to generate the positive triple as a negative one.
    n_articles = sum([len(v) for k, v in list(self.articles.items())])
    n_triples = sum([len(v) for k, v in list(self.triples.items())])
    n_subjects = len(self.subjects) #sum([len(v) for k, v in self.subjects.items()])
    self.size = (n_articles, n_triples, n_subjects)

  def get_batch(self, batch_size, do_shuffle=False, 
                n_batches=1, n_pos_triples=None, n_neg_triples=1,
                carefully_negative=False, n_articles=0, n_sentences=None,
                min_sentence_length=None, max_sentence_length=None):
    # TODO: remove carefully_negative
    '''
    n_pos_triples: The number of triples randomly sampled from true positive ones for an entity. If None, all triples are selected. 
    n_neg_triples: The number of triples randomly sampled from negative ones for an entity. The negative triples are created by replacing their object of true ones to the randomly sampled . If None, returns all objects. 
    '''
    def accepted(l):
      if min_sentence_length and len(l) < min_sentence_length:
        return False
      if max_sentence_length and len(l) > max_sentence_length:
        return False
      return True

    def select_sentences(sents, N=None):
      # sents = (w_sentence, c_sentence, l_start, l_end)
      accepted_sents = [l for l in sents if accepted(l[0])]
      if not N:
        return accepted_sents
      else:
        return random.sample(accepted_sents, min(N, len(accepted_sents)))

    # TODO: show stats of only accepted data
    articles = [(k, select_sentences(v, N=n_sentences))
                for k, v in list(self.articles.items()) if len(select_sentences(v, N=n_sentences)) != 0]
    if n_articles:
      articles = articles[:n_articles]
    if do_shuffle:
      random.shuffle(articles)

    # Extract n_batches * batch_size lines from data
    for i, b in itertools.groupby(enumerate(articles), 
                                  lambda x: x[0] // (batch_size*n_batches)):

      raw_batch = [x[1] for x in b] # (id, data) -> data

      # Yield 'n_batches' batches and each one has 'batch_size' lines.
      batch_articles = [[x[1] for x in b2] for j, b2 in itertools.groupby(enumerate(raw_batch), lambda x: x[0] // (len(raw_batch) // n_batches))]
      qids = [[k for k, v in b] for b in batch_articles]

      #_subj_descs = [[self.subjects[k]['w_desc'] for (k, _) in b] for b in batch_articles]
      _entities = [[self.subjects[k] for (k, _) in b] for b in batch_articles]

      _w_sentences = [[[w_sent for (w_sent, c_sent, s, e) in v] for k, v in b] 
                      for b in batch_articles]
      _c_sentences = [[[c_sent for (w_sent, c_sent, s, e) in v] for k, v in b] 
                      for b in batch_articles]
      _link_spans = [[[(s, e) for (w_sent, c_sent, s, e) in v] for k, v in b]
                     for b in batch_articles]

      _pos_triples = [self.get_positive_triples(q, N=n_pos_triples) for q in qids]
      _neg_triples = [self.get_negative_triples(q, p, N=n_neg_triples, carefully_negative=carefully_negative) for q, p in zip(qids, _pos_triples)]

      batches = []
      for ws, cs, ls, pt, nt, ent in zip(_w_sentences, _c_sentences, _link_spans, _pos_triples, _neg_triples, _entities):
        # TODO: Apply UNK masking if needed for the link spans of w_articles when testing.
        batch = common.dotDict({
          'w_articles': ws,
          'c_articles': cs,
          'link_spans': ls,
          'p_triples': pt,
          'n_triples': nt,
          'entities': ent,
          #'descriptions': desc
        })
        batches.append(batch)
      if len(batches) == 1:
        batches = batches[0]

      yield batches

  def get_positive_triples(self, qids, N=None):
    if N:
      triples = [[random.choice(self.triples[q]) for _ in range(N)] for q in qids]
    else:
      triples = [self.triples[q] for q in qids]
    return triples

  def get_negative_triples(self, qids, positives, N=1, carefully_negative=False):
    if N == 0:
      return None
    # for training.
    def random_sample(qid, p, N):
      negatives_by_sbj = []
      for rel, obj in p:
        ## replace relation instead of object.
        all_rels = [i for i in range(len(self.relations)) if i != rel]
        if carefully_negative:
          neg_rels = []
          while len(neg_rel) < N:
            i = random.choice(all_rels)
            if not self.all_triples or (qid, i, obj) not in self.all_triples:
              neg_rels.append(i)
        else:
          neg_rels = random.sample(all_rels, N)
        negatives_by_sbj.append([(neg_rel, obj) for neg_rel in neg_rels])
        # if carefully_negative:
        #   neg_objs = []
        #   while len(neg_objs) < N:
        #     i = random.choice(xrange(len(self.objects)))
        #     if not self.all_triples or (qid, rel, i) not in self.all_triples:
        #       neg_objs.append(i)
        # else:
        #   neg_objs = random.sample(xrange(len(self.objects)), N)
        # negatives_by_sbj.append([[rel, neg_obj] for neg_obj in neg_objs])
      return negatives_by_sbj

    # for test.
    def extract_all(qid, p):
      def _all_neg_obj(pos):
        return [x for x in range(len(self.objects)) if x != pos]

      all_obj = range(len(self.objects))
      if carefully_negative:
        negatives_by_sbj = [[(rel, i) for i in _all_neg_obj(obj)] for rel, obj in p]
      else:
        #negatives_by_sbj = [[[rel, i] for i in _all_neg_obj(obj)] for rel, obj in p] # こう書くと何故か妙に時間がかかるときがある・・・どこかでGCが走ったりしている？
        #negatives_by_sbj = [itertools.product([rel], _all_neg_obj(obj)) for rel, obj in p] # こう書いてproductを実体化させないと一度しか実行できなくなる
        negatives_by_sbj = [[x for x in itertools.product([rel], _all_neg_obj(obj))] for rel, obj in p]
      return negatives_by_sbj

    t = time.time()
    if N != None:
      negatives = [random_sample(qid, p, N) for qid, p in zip(qids, positives)]
    else:
      negatives = [extract_all(qid, p) for qid, p in zip(qids, positives)]
    return negatives
  
class WikiP2DDataset(DatasetBase):
  def __init__(self, w_vocab_size, c_vocab_size,
               w_vocab=None, c_vocab=None, # when using pretrained embs.
               filename='Q5O15000R300.micro.bin',
               source_dir='dataset/wikiP2D/source', 
               processed_dir='dataset/wikiP2D/processed', 
               lowercase=False, normalize_digits=False,
               cleanup=False):
    self.filename = filename

    w_vocab_size = w_vocab.size if w_vocab else w_vocab_size
    c_vocab_size = c_vocab.size if c_vocab else c_vocab_size
    w_suffix = w_vocab.name if w_vocab and w_vocab.name else w_vocab_size
    c_suffix = c_vocab.name if c_vocab and c_vocab.name else c_vocab_size

    # Add suffix to the binarized dataset to show what vocabulary it is tokenized by.
    suffix = '.W%sC%s' % (str(w_suffix), str(c_suffix))
    if lowercase:
      suffix += '.lower'
    if normalize_digits:
      suffix += '.normD'
    dataset_path = os.path.join(processed_dir, filename) + suffix

    # Load Data.
    if not os.path.exists(dataset_path) or cleanup:
      source_path = os.path.join(source_dir, filename)
      sys.stderr.write('Loading raw data from \'%s\' ...\n' % source_path)
      raw_data = pickle.load(open(source_path, 'rb'))

      # Create word and char vocabs even if pretrained ones are given.
      sys.stderr.write('Initializing vocab ...\n')
      vocab_corpus = common.flatten([[sent for sent, _, _ in v ] 
                                     for v in list(raw_data['articles']['train'].values())])
    else:
      sys.stderr.write('Loading tokenized data from \'%s\' ...\n' % dataset_path)
      self._data = pickle.load(open(dataset_path, 'rb'))
      raw_data = None
      vocab_corpus = None

    # Create vocab data if not given.
    if not w_vocab:
      self.w_vocab = WikiP2DVocabulary(
        vocab_corpus, dataset_path + '.w_vocab', w_vocab_size,
        cbase=False, lowercase=lowercase, normalize_digits=normalize_digits) 
    else:
      self.w_vocab = w_vocab

    if not c_vocab:
      self.c_vocab = WikiP2DVocabulary(
        vocab_corpus, dataset_path + '.c_vocab', c_vocab_size,
        cbase=True, lowercase=lowercase, 
        normalize_digits=normalize_digits)
    else:
      self.c_vocab = c_vocab

    s_vocab_path = os.path.join(processed_dir, filename) + '.s_vocab'
    r_vocab_path = os.path.join(processed_dir, filename) + '.r_vocab'
    o_vocab_path = os.path.join(processed_dir, filename) + '.o_vocab'


    subjects = raw_data['subjects']['train'] if raw_data else None
    self.s_vocab = WikiP2DSubjVocabulary(subjects, 
                                         s_vocab_path, vocab_size=None)
    relations = raw_data['relations'] if raw_data else None
    self.r_vocab = WikiP2DRelVocabulary(relations, r_vocab_path,
                                        vocab_size=None)
    objects = raw_data['objects'] if raw_data else None
    self.o_vocab = WikiP2DObjVocabulary(objects, o_vocab_path,
                                        vocab_size=None)

    # Restore tokenized dataset.
    if not os.path.exists(dataset_path):
      sys.stderr.write('Tokenizing dataset ...\n')
      assert raw_data != None
      self._data = self.tokenize_data(raw_data, self.w_vocab, self.c_vocab, self.s_vocab, self.r_vocab, self.o_vocab)
      pickle.dump(self._data, open(dataset_path, 'wb'))

    all_triples = None
    # Devided into train, valid, and test.
    self.train = _WikiP2DDataset(self._data['articles']['train'], 
                                 self._data['triples']['train'],
                                 self._data['subjects']['train'],
                                 self._data['relations'], self._data['objects'],
                                 all_triples=all_triples)
    self.valid = _WikiP2DDataset(self._data['articles']['valid'], 
                                 self._data['triples']['valid'],
                                 self._data['subjects']['valid'],
                                 self._data['relations'], self._data['objects'],
                                 all_triples=all_triples)
    self.test = _WikiP2DDataset(self._data['articles']['test'], 
                                 self._data['triples']['test'],
                                 self._data['subjects']['test'],
                                 self._data['relations'], self._data['objects'],
                                 all_triples=all_triples)
    # Shared among them.
    self.relations = self._data['relations']
    self.objects = self._data['objects']

    # Log the data statistics.
    data_stat_dir = os.path.join(processed_dir, filename) + '.stat'
    self.stat(data_stat_dir, 'train',
              self._data['articles']['train'],
              self._data['triples']['train'],
              self._data['subjects']['train'])
    self.stat(data_stat_dir, 'valid',
              self._data['articles']['valid'],
              self._data['triples']['valid'],
              self._data['subjects']['valid'])
    self.stat(data_stat_dir, 'test',
              self._data['articles']['test'],
              self._data['triples']['test'],
              self._data['subjects']['test'])


  def tokenize_data(self, data, w_vocab, c_vocab, s_vocab, r_vocab, o_vocab):
    articles = {k:process_articles(v, w_vocab, c_vocab) for k, v in list(data['articles'].items())}
    triples = {k:process_triples(v, r_vocab, o_vocab) for k, v in list(data['triples'].items())}
    subjects = {k:process_entities(v, w_vocab, c_vocab) for k, v in list(data['subjects'].items())}
    relations = process_entities(data['relations'], w_vocab, c_vocab)
    objects = process_entities(data['objects'], w_vocab, c_vocab)
    res = {
      'articles': articles,
      'triples': triples,
      'subjects': subjects,
      'relations': relations,
      'objects': objects,
    }
    return res

  def stat(self, log_dir, filename, articles, triples, subjects):
    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(log_dir): 
      os.makedirs(log_dir)

    if not os.path.exists(file_path + '.len.hist.eps'):
      lengthes = common.flatten([[len(w_sent) for (w_sent, _, _, _) in v] 
                                 for k, v in list(articles.items())])
      visualize.histgram([lengthes], ['article length'], 
                         file_path=file_path + '.len.hist.eps')

    wlink_file = file_path + '.link.hist.txt'
    if not os.path.exists(wlink_file):
      linked_phrases = common.flatten([[(qid, self.c_vocab.ids2tokens(c_sent[s:e+1])) for w_sent, c_sent, s ,e in v] for qid, v in list(articles.items())])

      linked_entity_by_phrase = defaultdict(dict)
      for qid, phrase in linked_phrases:
        if qid in linked_entity_by_phrase[phrase]:
          linked_entity_by_phrase[phrase][qid] += 1
        else:
          linked_entity_by_phrase[phrase][qid] = 1
      linked_phrases = [phrase for qid, phrase in linked_phrases]

      linked_phrases = sorted([(k, freq) for k, freq in list(Counter(linked_phrases).items())], key=lambda x:-x[1])
      with open(wlink_file, 'w') as f:
        for phrase, freq in linked_phrases:
          linked_entities = ["%s:%d" % (qid, freq_by_qid) for qid, freq_by_qid in list(linked_entity_by_phrase[phrase].items())]
          entities_list = ",".join(linked_entities)
          f.write('%s\t%d\t%s\n' % (phrase, freq, entities_list))

  def get_all_triples(self):
    # for demo.
    return [list(x) for x in itertools.product(range(len(self.relations)), 
                                               range(len(self.objects)))]
  def batch2text(self, batch):
    w_articles = batch['w_articles']
    c_articles = batch['c_articles'] 
    link_spans = batch['link_spans']
    p_triples = batch['p_triples'] 
    entities = batch['entities']

    def _batch2text(w_sentences, c_sentences, l):
      res = [(self.w_vocab.ids2tokens(ws, link_span=l), 
              self.c_vocab.ids2tokens(cs, link_span=l))
             for ws, cs, l in zip(w_sentences, c_sentences, ls)]
      return list(map(list, list(zip(*res))))

    texts = []
    for entity, w_sentences, c_sentences, ls, pt in zip(entities, w_articles, c_articles, link_spans, p_triples):
      _ws, _cs = _batch2text(w_sentences, c_sentences, ls)
      _triples = [(self.r_vocab.id2name(r), self.o_vocab.id2name(o)) for (r, o) in pt]
      texts.append([
        entity['name'], _ws, _cs, _triples
      ])
    return texts


