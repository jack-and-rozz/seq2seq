#!/usr/bin/env python
from pprint import pprint
from collections import Counter, OrderedDict
import sys, json, argparse, os
sys.path.append(os.getcwd())
from core.utils.common import read_jsonlines, read_embeddings, recDotDefaultDict, flatten, separate_path_and_filename

def filter_embeddings(embeddings, data_by_genre, unused_genre=None):
  data = flatten([v for k, v in data_by_genre.items() if not unused_genre or k != unused_genre])
  words = Counter(flatten(flatten([d.sentences for d in data])))
  words = [k for k,v in sorted(list(words.items()), key=lambda x: -x[1])]
  filtered_emb = OrderedDict()
  for w in words:
    if w in embeddings:
      filtered_emb[w] = embeddings[w]
  return filtered_emb

def main(args):
  embeddings = read_embeddings(args.source_embedding)
  #embeddings = {}
  data = read_jsonlines(args.train_data)
  data_by_genre = {}
  for d in data:
    genre = d.doc_key[:2]
    if not genre in data_by_genre:
      data_by_genre[genre] = []
    data_by_genre[genre].append(d)
  
  genres = list(data_by_genre.keys())
  sys.stderr.write('Genres: %s\n' % str(genres))
  filtered_emb = {}
  for genre in genres:
    filtered_emb['wo_%s' % genre] = filter_embeddings(embeddings, data_by_genre, unused_genre=genre)
  filtered_emb['all_train'] = filter_embeddings(embeddings, data_by_genre, unused_genre=None)

  emb_dir, emb_filename = separate_path_and_filename(args.source_embedding)
  target_dir = emb_dir + '/filtered.conll'
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  for k, v in filtered_emb.items():
    target_path = '%s/%s.%s' % (target_dir, emb_filename, k)
    with open(target_path, 'w') as f:
      for word, vec in v.items():
        line = '%s %s\n' % (word, ' '.join([str(x) for x in vec]))
        f.write(line)

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-td', '--train_data', type=str, help ='',
                      default='dataset/coref/source/train.english.jsonlines')
  parser.add_argument('-se', '--source_embedding', type=str, help ='',
                      default='embeddings/glove.840B.300d.txt')
  args = parser.parse_args()
  main(args)


    # if len(sys.argv) < 3:
    #     sys.exit("Usage: {} <embeddings> <json1> <json2> ...".format(sys.argv[0]))

    # words_to_keep = set()
    # for json_filename in sys.argv[2:]:
    #     with open(json_filename) as json_file:
    #         for line in json_file.readlines():
    #             for sentence in json.loads(line)["sentences"]:
    #                 words_to_keep.update(sentence)

    # print "Found {} words in {} dataset(s).".format(len(words_to_keep), len(sys.argv) - 2)

    # total_lines = 0
    # kept_lines = 0
    # out_filename = "{}.filtered".format(sys.argv[1])
    # with open(sys.argv[1]) as in_file:
    #     with open(out_filename, "w") as out_file:
    #         for line in in_file.readlines():
    #             total_lines += 1
    #             word = line.split()[0]
    #             if word in words_to_keep:
    #                 kept_lines += 1
    #                 out_file.write(line)

    # print "Kept {} out of {} lines.".format(kept_lines, total_lines)
    # print "Wrote result to {}.".format(out_filename)
