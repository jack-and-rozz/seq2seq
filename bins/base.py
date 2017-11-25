#coding: utf-8
from __future__ import absolute_import
from pprint import pprint
import sys, os, random
from logging import FileHandler
import numpy as np
import tensorflow as tf

#from tensorflow.python.platform import gfile
#from six.moves import xrange  # pylint: disable=redefined-builtin

from core.utils import common
random.seed(0)
np.random.seed(0)


## temporal flags (not saved in config)
tf.app.flags.DEFINE_string("mode", "train", "")
tf.app.flags.DEFINE_string("log_file", "train.log", "")
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/model.ckpt', 
                           'Directory to save the trained model.')
tf.app.flags.DEFINE_string('config_path', 'configs/experiments.conf', '')


log_file = tf.app.flags.FLAGS.log_file if tf.app.flags.FLAGS.log_file else None
logger = common.logManager(handler=FileHandler(log_file)) if log_file else common.logManager()


class BaseManager(object):
  def __init__(self, FLAGS, sess):
    self.FLAGS = FLAGS
    FLAGS = self.FLAGS
    self.TMP_FLAGS = ['mode', 'log_file', 'checkpoint_path', 'write_summary']
    self.CHECKPOINTS_PATH = FLAGS.checkpoint_path +'/checkpoints'
    self.TESTS_PATH = FLAGS.checkpoint_path + '/tests'
    self.VARIABLES_PATH = FLAGS.checkpoint_path +'/variables'
    self.SUMMARIES_PATH = FLAGS.checkpoint_path + '/summaries'
    self.sess = sess
    self.config = common.get_config(FLAGS.config_path)
    config_restored_path = os.path.join(FLAGS.checkpoint_path, 'experiments.conf')
    with open(config_restored_path, 'w') as f:
      sys.stdout = f
      common.print_config(self.config)
      sys.stdout = sys.__stdout__
    self.config = common.recDotDict(self.config)['main']
    
  def create_dir(self):
    FLAGS = self.FLAGS
    if not os.path.exists(FLAGS.checkpoint_path):
      os.makedirs(FLAGS.checkpoint_path)
    if not os.path.exists(self.CHECKPOINTS_PATH):
      os.makedirs(self.CHECKPOINTS_PATH)
    if not os.path.exists(self.TESTS_PATH):
      os.makedirs(self.TESTS_PATH)
    if not os.path.exists(self.VARIABLES_PATH):
      os.makedirs(self.VARIABLES_PATH)
    if not os.path.exists(self.SUMMARIES_PATH):
      os.makedirs(self.SUMMARIES_PATH)

    # flags_dir = self.FLAGS.__dict__['__flags']
    # with open(self.FLAGS.checkpoint_path + '/config', 'w') as f:
    #   for k,v in flags_dir.items():
    #     if not k in self.TMP_FLAGS:
    #       f.write('%s=%s\n' % (k, str(v)))


