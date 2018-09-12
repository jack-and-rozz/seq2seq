#coding: utf-8
import sys, os, argparse
from collections import OrderedDict, Counter
from pprint import pprint
from core.utils.common import dotDict, recDotDefaultDict, recDotDict, flatten, batching_dicts, dbgprint, flatten_recdict, str2bool, read_jsonlines
from core.vocabulary.base import word_tokenizer

def read_text(text, tokenizer):
  if isinstance(text, list):
    assert type(text[0]) == str
    words = flatten(tokenizer(sent) for sent in text)
  else:
    words = tokenizer(text)
  return words

def read_embedding(emb_path):
  d = OrderedDict()
  for l in open(emb_path):
    l = l.replace('\n', '').split(' ')
    word = l[0]
    vector = [float(x) for x in l[1:]]
    d[word] = vector
  return d

def main(args):
  word_embs = read_embedding(args.source_emb)
  data = read_jsonlines(args.dataset_path, max_rows=0)
  tokenizer = word_tokenizer(args.lowercase, args.normalize_digits)
  words = flatten([read_text(d.text, tokenizer) for d in data])
  word_freq = sorted(Counter(words).items(), key=lambda x:-x[1])
  for word, freq in word_freq:
    if word in word_embs:
      line = [word] + word_embs[word]
      line = ' '.join([str(x) for x in line])
      print(line)

if __name__ == "__main__":
  desc = "This is a script to reorder words and their embeddings in order of their frequencies. The words which don't appear in training dataset are removed."
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--source_emb', default='embeddings/glove.840B.300d.txt', 
                      type=str, help ='')
  parser.add_argument('--dataset_path', 
                      default='dataset/wikiP2D/source/relex/train.jsonlines', 
                      type=str, help ='')
  parser.add_argument('-l', '--lowercase', default=False, 
                      type=str2bool, help ='')
  parser.add_argument('-n', '--normalize_digits', default=False, 
                      type=str2bool, help ='')
  args = parser.parse_args()
  main(args)
