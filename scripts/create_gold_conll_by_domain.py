# coding:utf-8
import sys, argparse, os, re, argparse
sys.path.append(os.getcwd())


def create_gold_file(fpath, genre):
  is_printed = False
  for line in open(fpath):
    row = line.split()
    if row and row[0] == '#begin':
      key = re.search('\((.+)\);', row[2]).group(1)
      if key[:2] == genre:
        is_printed = True
    elif row and row[0] == '#end':
      if is_printed:
        print(line)
      is_printed = False
    if is_printed:
      sys.stdout.write(line)

def main(args):
  #create_gold_file('/tmp/tmpzylcs1zp', 'bc')
  #exit(1)
  genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
  for genre in genres:
    for filename in [args.train_file, args.dev_file, args.test_file]:
      target_dir = args.source_dir + '/' +  genre
      if not os.path.exists(target_dir):
        os.makedirs(target_dir)
      source_path = os.path.join(args.source_dir, filename)
      target_path = os.path.join(target_dir, filename) 
      with open(target_path, 'w') as f:
        sys.stdout = f
        create_gold_file(source_path, genre)
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-s', '--source_dir', 
                      default='dataset/coref/source', help ='')
  parser.add_argument('--train_file', 
                      default='train.english.v4_auto_conll', help ='')
  parser.add_argument('--dev_file', 
                      default='dev.english.v4_auto_conll', help ='')
  parser.add_argument('--test_file', 
                      default='test.english.v4_gold_conll', help ='')
  
  args = parser.parse_args()
  main(args)
