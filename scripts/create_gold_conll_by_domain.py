# coding:utf-8
import sys, argparse, os, re, argparse
sys.path.append(os.getcwd())


def create_gold_file(fpath, genre):
  is_indomain = False
  for line in open(fpath):
    if row[0].startswith("#"):

def main(args):
  genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
  dev_file = os.path.join(args.source_dir, 'dev.english.v4_auto_conll')
  test_file = os.path.join(args.source_dir, 'test.english.v4_gold_conll')
  for genre in genres:
    for fpath in [dev_file, test_file]:
      create_gold_file(fpath, genre)

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-s', '--source_dir', 
                      default='dataset/coref/source', help ='')
  args = parser.parse_args()
  main(args)
