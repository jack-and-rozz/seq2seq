# coding:utf-8
import sys, argparse, os, re
sys.path.append(os.getcwd())
from core.vocabulary.base import word_tokenizer
from collections import OrderedDict, Counter
from orderedset import OrderedSet
from pprint import pprint
import numpy as np 
from core.utils.common import read_jsonlines, read_json, str2bool, flatten, timewatch

@timewatch()
def read_pretrained_emb(word_freq, tokenizer):
  embedding_dict = OrderedDict([(k, None) for k in word_freq.keys()])
  for i, l in enumerate(open(args.embedding_path)):
    word = l.split(' ')[0]
    vector = [float(x) for x in l.split(' ')[1:]]
    embedding_size = len(vector)

    tokenized_word = tokenizer(word)[0] if len(tokenizer(word)) == 1 else word
    if word in embedding_dict:
      embedding_dict[word] = vector
    elif tokenized_word in embedding_dict and not embedding_dict[tokenized_word]:
      embedding_dict[tokenized_word] = vector

  # for k in embedding_dict:
  #   if not embedding_dict[k]:
  #     embedding_dict[k] = np.zeros(embedding_size)
  return embedding_dict

def main(args):
  tokenizer = word_tokenizer(args.lowercase, args.normalize_digits,
                             separative_tokens=['-', '/'])
  data = read_jsonlines(args.descdata_path)

  word_freq = OrderedDict(sorted([(k, freq) for k, freq in Counter(flatten([tokenizer(d.desc) for d in data])).items()], key=lambda x: -x[1]))
  embedding_dict = read_pretrained_emb(word_freq, tokenizer)

  with open(args.emb_target_path, 'w') as f:
    for w, v in embedding_dict.items():
      if not v:
        continue
      line = "%s %s\n" % (w, ' '.join([str(x) for x in v]))
      f.write(line)

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-e', '--embedding_path', 
                      default='embeddings/glove.840B.300d.txt', help ='')
  parser.add_argument('--emb_target_path', 
                      default='embeddings/glove.840B.300d.txt.for_desc', help ='')
  # parser.add_argument('--desc_word_freq_target_path', 
  #                     default='embeddings/desc_words.freq.txt', help ='')
  parser.add_argument('-d', '--descdata_path', 
                      default='dataset/wikiP2D/source/desc_and_category/train.jsonlines', help ='')
  parser.add_argument('-l', '--lowercase', default=True, type=str2bool)
  parser.add_argument('-n', '--normalize_digits', default=True, type=str2bool)
  args = parser.parse_args()
  main(args)
