#coding: utf-8
import argparse, re, os, subprocess
import nltk
from core.utils import common

def main(args):
  try:
    reference_file = re.search('(.+)\.decode\..+', args.hypothesis_file).group(1)
  except:
    raise ValueError('hypothesis_file must be this format: [test_file].decode.(.+)')
  hypothesis_file = args.hypothesis_file

  multi_bleu_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                 "multi-bleu.perl")
  bleu_cmd = [multi_bleu_path]
  if args.lowercase:
    bleu_cmd += ["-lc"]
  bleu_cmd += [reference_file]
  bleu_out = subprocess.check_output(
    bleu_cmd, stdin=open(hypothesis_file), stderr=subprocess.STDOUT)
  bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
  bleu_score = float(bleu_score)
  print(bleu_out)

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('hypothesis_file', help ='')
  parser.add_argument('--lowercase', type=bool, default=False)
  args = parser.parse_args()
  main(args)
