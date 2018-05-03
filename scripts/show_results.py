# coding:utf-8
import sys, re, argparse, subprocess, os
from utils import common 
import glob

def equalize_str_length(models):
  max_len = max([len(m) for m in models])
  return [m + ' ' * (max_len-len(m)) for m in models]
  

def read_corpus(test_file, source_lang, target_lang, 
                dataset_path='dataset/source'):
  sources = common.read_file(dataset_path + '/%s.%s' % (test_file, source_lang),
                             do_tokenize=False)
  targets = common.read_file(dataset_path + '/%s.%s' % (test_file, target_lang),
                             do_tokenize=False)
  return sources, targets

def main(args):
  dirs = [m for m in subprocess.getoutput('ls -d %s' % args.proj_dir + '/*').split()
          if os.path.exists(m + '/config')]
  models = []
  results = []
  configs = []
  for m in dirs:
    mconfig = common.read_mconfig(m, 'config')
    decode_file = '%s/tests/%s.%s.decode.*' % (m, mconfig.test_data, 
                                               mconfig.target_lang)
    if glob.glob(decode_file):
      query = ' ls -d %s' % (decode_file)
      decode_results = subprocess.getoutput(query).split()
      newest = sorted([(d, int(re.search('ep([0-9]+)', d).group(1))) 
                for d in decode_results], key=lambda x: -x[1])[0][0]
      result = common.read_file(newest, do_tokenize=False)
      models.append(m)
      results.append(result)
      configs.append(mconfig)
  if len(set([mc.test_data for mc in configs])) != 1:
    raise Exception("All models must have a same test file.")
  
  sources_with_unk, targets_with_unk = read_corpus(
    configs[0].test_data, 
    configs[0].source_lang,
    configs[0].target_lang,
    dataset_path=models[0] + '/tests'
  )
  sources, targets = read_corpus(
    configs[0].test_data, 
    configs[0].source_lang,
    configs[0].target_lang,
  )

  models = ['source', 'target', 'source(unk)', 'target(unk)'] + [re.search('.+/(.+?)$', m).group(1) for m in models]
  models = equalize_str_length(models)
  results = [sources, targets, sources_with_unk, targets_with_unk] + results
  show(models, results)

def show(models, results):
  for i in range(len(results[0])):
    print(('<%d>' % i))
    for m, r in zip(models, results):
      print(('%s : %s' % (m, r[i])))
    print ('')

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('proj_dir', help ='')
  args = parser.parse_args()
  main(args)
