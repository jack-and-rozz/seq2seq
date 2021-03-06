#coding: utf-8
import tensorflow as tf
import sys, os, random, copy, socket, time, re, argparse
sys.path.append(os.getcwd())
from collections import OrderedDict
from pprint import pprint
import numpy as np
from core.utils.common import flatten, flatten_recdict, recDotDefaultDict
from bins.wikiP2D import ExperimentManager


def to_wikiP2D_batch(coref_batch):
  batch = recDotDefaultDict()
  batch.contexts.word = np.expand_dims(coref_batch.text.word, axis=0)
  batch.contexts.char = np.expand_dims(coref_batch.text.char, axis=0)
  print(flatten(coref_batch.clusters))
  return batch

class CorefDescTestManager(ExperimentManager):
  def combine_test(self):
    m = self.create_model(self.config, load_best=True)
    #tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if k != 'adversarial'])
    exit(1)
    #print(m.tasks)
    mode = 'test'
    batches = self.get_batch(mode)
    print('<coref>')
    for bb in batches['coref']:
      b = to_wikiP2D_batch(bb)
      continue
      b = flatten_recdict(bb)
      for k,v in b.items():
        if type(v) == np.ndarray:
          print(k, v.shape)
        else:
          print(k)

      b = to_wikiP2D_batch(bb)
      b = flatten_recdict(b)
      for k,v in b.items():
        if type(v) == np.ndarray:
          print(k, v.shape)
      break

    print('<desc>')
    for b in batches['desc']:
      b = flatten_recdict(b)
      for k,v in b.items():
        if type(v) == np.ndarray:
          print(k, v.shape)
      break
      


def main(args):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    random.seed(0)
    tf.set_random_seed(0)
    manager = CorefDescTestManager(args, sess)
    manager.combine_test()


if __name__ == '__main__':
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root_path', type=str, help ='')
  parser.add_argument('--mode', default='test', type=str, help ='')
  args = parser.parse_args()
  main(args)
