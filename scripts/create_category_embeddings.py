# coding:utf-8
import sys, argparse
from collections import OrderedDict
from orderedset import OrderedSet
from pprint import pprint
import numpy as np 

def main(args):

  categories = OrderedSet([l.split(' ')[0] for l in open(args.category_path) if l.split(' ')[0] not in ['', '-'] ])
  embedding_dict = OrderedDict()
  embedding_size = None
  for i, l in enumerate(open(args.embedding_path)):
    #if i > 10000:
    #   break
    word = l.split(' ')[0]
    if word in categories:
      vector = [float(x) for x in l.split(' ')[1:]]
      embedding_dict[word] = vector
      embedding_size = len(vector)
  with open(args.target_path, 'w') as f:
    for c in categories:
      v = [str(x) for x in np.zeros(embedding_size)]
      if c in embedding_dict:
        v = [str(x) for x in embedding_dict[c]]
      line = "%s %s\n" % (c, ' '.join(v))
      f.write(line)

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--embedding_path', 
                      default='embeddings/glove.840B.300d.txt',help ='')
  parser.add_argument('--target_path', 
                      default='embeddings/categories.glove.300d.txt',help ='')
  parser.add_argument('--category_path', default='dataset/wikiP2D/source/desc_and_category/category_freq.txt',help ='')
  args = parser.parse_args()
  main(args)
