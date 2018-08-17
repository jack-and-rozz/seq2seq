#coding: utf-8
from pprint import pprint
import os, time, re, sys
from collections import defaultdict, OrderedDict, Counter
import core.utils.common as common
from core.vocabulary.base import  PAD_ID, BOS_ID, UNK_ID, _PAD, _BOS, _UNK, word_tokenizer, char_tokenizer
from core.vocabulary.base import WordVocabularyBase, CharVocabularyBase, VocabularyBase

class WikiP2DVocabularyBase(VocabularyBase):
  def __init__(self, sentences, vocab_path, vocab_size,
               cbase=False, lowercase=False, special_words=None,
               normalize_digits=False):
    super(WikiP2DVocabularyBase, self).__init__()


# class WikiP2DVocabulary(VocabularyBase):
#   def __init__(self, sentences, vocab_path, vocab_size,
#                cbase=False, lowercase=False, special_words=None,
#                normalize_digits=False):
#     self.cbase = cbase
#     if self.cbase:
#       self.tokenizer = char_tokenizer(special_words=special_words, 
#                                       lowercase=lowercase,
#                                       normalize_digits=normalize_digits) 
#     else: 
#       self.tokenizer = word_tokenizer(lowercase=lowercase,
#                                       normalize_digits=normalize_digits) 

#     self.vocab, self.rev_vocab = self.init_vocab(sentences, vocab_path, vocab_size)


#   def init_vocab(self, sentences, vocab_path, vocab_size):
#     if os.path.exists(vocab_path):
#       return self.load_vocab(vocab_path)
#     elif not sentences:
#       raise ValueError('This vocabulary does not exist and no sentences were passed when initializing.')

#     START_VOCAB = [(_PAD, 0), (_BOS, 0),  (_UNK, 0) ]
#     tokenized = common.flatten([self.tokenizer(s) for s in sentences])
#     if isinstance(tokenized[0], list):
#       tokenized = common.flatten(tokenized)
#     tokens = Counter(tokenized)
#     tokens = sorted([(k, f) for (k, f) in list(tokens.items())], key=lambda x: -x[1])

#     rev_vocab = [k for k, _ in START_VOCAB + tokens[:(vocab_size - len(START_VOCAB))]]
#     vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})

#     START_VOCAB[UNK_ID] = (_UNK, sum([f for _, f in tokens[vocab_size:]]))
#     START_VOCAB[BOS_ID] = (_BOS, len(sentences))
#     restored_data = START_VOCAB + tokens[:(vocab_size - len(START_VOCAB))]
#     self.save_vocab(restored_data, vocab_path)
#     return vocab, rev_vocab

#   def load_vocab(self, vocab_path):
#     sys.stderr.write('Loading vocabulary from \'%s\' ...\n' % vocab_path)
#     rev_vocab = [l.split('\t')[0] for l in open(vocab_path, 'r')]
#     vocab = OrderedDict({t:i for i,t in enumerate(rev_vocab)})
#     return vocab, rev_vocab

#   def save_vocab(self, restored_data, vocab_path):
#     with open(vocab_path, 'w') as f:
#       f.write('\n'.join(["%s\t%d"% tuple(x) for x in restored_data]) + '\n')

#   def id2token(self, _id):
#     if _id < 0 or _id > len(self.rev_vocab):
#       raise ValueError('Token ID must be between 0 and %d' % len(self.rev_vocab))
#     elif _id in set([PAD_ID, BOS_ID]):
#       return ''
#     else:
#       return self.rev_vocab[_id]

#   def token2id(self, token):
#     return self.vocab.get(token, UNK_ID)

#   def sent2ids(self, sentence):
#     if type(sentence) == list:
#       sentence = " ".join(sentence)
#     tokens = self.tokenizer(sentence) 
#     if self.cbase:
#       res = [[self.token2id(char) for char in word] for word in tokens]
#     else:
#       res = [self.token2id(word) for word in tokens]
#     return res

#   def ids2tokens(self, ids, link_span=None):
#     if self.cbase:
#       sent_tokens = ["".join([self.id2token(char_id) for char_id in word]) 
#                      for word in ids]
#     else:
#       sent_tokens = [self.id2token(word_id) for word_id in ids]
#     if link_span:
#       for i in range(link_span[0], link_span[1]+1):
#         sent_tokens[i] = common.colored(sent_tokens[i], 'link')
#       sent_tokens = [w for w in sent_tokens if w]
#     return " ".join(sent_tokens)

#   def padding(self, sentences, max_sentence_length=None, max_word_length=None):
#     '''
#     '''
#     if not max_sentence_length:
#       max_sentence_length = max([len(s) for s in sentences])
#     if not max_word_length and self.cbase:
#       max_word_length = max([max([len(w) for w in s]) for s in sentences])

#     def wsent_padding(sentences, max_s_length):
#       def w_pad(sent):
#         padded_s = self.start_offset + sent[:max_s_length] + self.end_offset
#         size = len(padded_s)
#         padded_s += [PAD_ID] * (max_s_length + self.n_start_offset + self.n_end_offset - size)
#         return padded_s, size
#       res = [w_pad(s) for s in sentences]
#       return list(map(list, list(zip(*res))))

#     def csent_padding(sentences, max_s_length, max_w_length):
#       def c_pad(w):
#         padded_w = w[:max_w_length] 
#         size = len(padded_w)
#         padded_w += [PAD_ID] * (max_w_length - size)
#         return padded_w, size
#       def s_pad(s):
#         s = s[:max_s_length]
#         padded_s, word_lengthes = list(map(list, list(zip(*[c_pad(w) for w in s]))))
#         if self.start_offset:
#           padded_s.insert(0, self.start_offset + [PAD_ID] * (max_w_length - len(self.start_offset)))
#           word_lengthes.insert(0, 1)
#         if self.end_offset:
#           padded_s.append(self.end_offset + [PAD_ID] * (max_w_length-len(self.end_offset)))
#           word_lengthes.extend([1]+[0] * (max_s_length + self.n_start_offset + self.n_end_offset - sentence_length))
#         sentence_length = len(padded_s)
#         padded_s += [[PAD_ID] * max_w_length] * (max_s_length + self.n_start_offset + self.n_end_offset - sentence_length)
#         return padded_s, sentence_length, word_lengthes
#       res = [s_pad(s) for s in sentences]
#       return list(map(list, list(zip(*res))))
#     if self.cbase:
#       return csent_padding(sentences, max_sentence_length, max_word_length)
#     else:
#       return wsent_padding(sentences, max_sentence_length)


class WikiP2DRelVocabulary(WordVocabularyBase):
  def __init__(self, data, start_vocab=[_UNK], vocab_size=None):
    '''
    data : Ordereddict[Pid] = {'name': str, 'freq': int, 'aka': set, 'desc': str}
    '''
    
    rev_vocab, rev_names = (list(x) for x in self.load_data(data))
    if vocab_size:
      rev_vocab = rev_vocab[:vocab_size]
      rev_names = rev_names[:vocab_size]
    self.rev_vocab = start_vocab + rev_vocab
    self.rev_names = start_vocab + rev_names
    self.vocab = OrderedDict({t:i for i,t in enumerate(self.rev_vocab)})
    self.names = OrderedDict([(self.id2name(_id), _id) for _id in range(len(self.rev_names))])
  def name2token(self, name):
    return self.id2token(self.name2id(name))

  def token2name(self, token):
    return self.id2name(self.token2id(token))

  def name2id(self, name):
    return self.names.get(name, self.names.get(_UNK))

  def id2name(self, _id):
    return self.rev_names[_id] #self.data[self.id2token(_id)]['name']

  def load_data(self, data):
    return list(zip(*[(d.qid, d.name) for d in data]))

  def load_vocab(self, vocab_path):
    raise NotImplementedError

class WikiP2DObjVocabulary(WikiP2DRelVocabulary):
  pass

class WikiP2DSubjVocabulary(WikiP2DRelVocabulary):
  pass


