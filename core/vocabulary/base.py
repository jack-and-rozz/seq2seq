#coding: utf-8
import collections, os, time, re
from tensorflow.python.platform import gfile
import core.utils.common as common

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
  l = l.decode('utf-8') if type(l) == str else l
  l = mojimoji.zen_to_han(l, kana=False) #全角数字・アルファベットを半角に
  l = mojimoji.han_to_zen(l, digit=False, ascii=False) #半角カナを全角に
  l = l.encode('utf-8')
  return l

def separate_numbers(sent):
  def addspace(m):
    return ' ' + m.group(0) + ' '
  return re.sub(_DIGIT_RE, addspace, sent).replace('  ', ' ')

def space_tokenizer(sent, do_format_zen_han=True, 
                    do_separate_numbers=True):
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
