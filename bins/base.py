#coding: utf-8
from logging import FileHandler
from pprint import pprint
import sys, os, random
import numpy as np
import tensorflow as tf

from core.utils import common
random.seed(0)
np.random.seed(0)

def get_logger(logfile_path=None):
  logger = common.logManager(handler=FileHandler(logfile_path)) if logfile_path else common.logManager()
  return logger

class ManagerBase(object):
  def __init__(self, args, sess):
    self.root_path = args.checkpoint_path
    self.checkpoints_path = args.checkpoint_path +'/checkpoints'
    self.tests_path = args.checkpoint_path + '/tests'
    self.summaries_path = args.checkpoint_path + '/summaries'
    self.sess = sess
    self.create_dir(args)
    self.logger = get_logger(logfile_path=os.path.join(args.checkpoint_path, args.mode + '.log'))

  def load_config(self, args):
    config_stored_path = os.path.join(args.checkpoint_path, 'experiments.conf')
    if os.path.exists(config_stored_path): 
      config_read_path = config_stored_path
      config = common.get_config(config_read_path)
      sys.stderr.write("Found an existing config file %s.\n" % (config_stored_path))
    else: 
      config_read_path = args.config_path
      config = common.get_config(config_read_path)[args.config_type]
      sys.stderr.write("Store the specified config file %s(%s) to %s.\n" % (config_read_path, args.config_type, config_stored_path))
      with open(config_stored_path, 'w') as f:
        sys.stdout = f
        common.print_config(config)
        sys.stdout = sys.__stdout__
    config = common.recDotDict(config)
    sys.stderr.write(str(config) + '\n')
    return config

  def create_dir(self, args):
    if not os.path.exists(args.checkpoint_path):
      os.makedirs(args.checkpoint_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)


