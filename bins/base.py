#coding: utf-8
from __future__ import absolute_import
from __future__ import division

import sys, os, random
from logging import FileHandler
import numpy as np
import tensorflow as tf
#from tensorflow.python.platform import gfile
#from six.moves import xrange  # pylint: disable=redefined-builtin

import core
from core.utils import common
from core.models.base import MultiGPUTrainWrapper

random.seed(0)
np.random.seed(0)

tf.app.flags.DEFINE_string("model_type", "Baseline", "")
tf.app.flags.DEFINE_integer("max_to_keep", 5, "Number of checkpoints to be kept")

## temporal flags (not saved in config)
tf.app.flags.DEFINE_string("mode", "train", "")
tf.app.flags.DEFINE_string("log_file", "train.log", "")
tf.app.flags.DEFINE_string('checkpoint_path', 'models/local/tmp', 'Directory to put the training data.')

log_file = tf.app.flags.FLAGS.log_file if tf.app.flags.FLAGS.log_file else None
logger = common.logManager(handler=FileHandler(log_file)) if log_file else common.logManager()


class BaseManager(object):
  def __init__(self, FLAGS, sess):
    self.FLAGS = FLAGS
    FLAGS = self.FLAGS
    self.TMP_FLAGS = ['mode', 'log_file', 'checkpoint_path']
    self.CHECKPOINTS_PATH = FLAGS.checkpoint_path +'/checkpoints'
    self.TESTS_PATH = FLAGS.checkpoint_path + '/tests'
    self.VARIABLES_PATH = FLAGS.checkpoint_path +'/variables'
    self.SUMMARIES_PATH = FLAGS.checkpoint_path + '/summaries'
    self.sess = sess

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

  def save_config(self):
    flags_dir = self.FLAGS.__dict__['__flags']
    with open(self.FLAGS.checkpoint_path + '/config', 'w') as f:
      for k,v in flags_dir.items():
        if not k in self.TMP_FLAGS:
          f.write('%s=%s\n' % (k, str(v)))

  @common.timewatch(logger)
  def create_model(self, FLAGS, forward_only, do_update, reuse=None):
    sess = self.sess
    with tf.variable_scope("Model", reuse=reuse):
      if do_update and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        m = MultiGPUTrainWrapper(sess, FLAGS)
      else:
        m = self.model_type(sess, FLAGS, forward_only, do_update)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path + '/checkpoints')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
  
    if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.index'):
      if reuse==None:
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      if reuse==None:
        logger.info("Created model with fresh parameters.")
      tf.global_variables_initializer().run()
      with open(FLAGS.checkpoint_path + '/variables/variables.list', 'w') as f:
        f.write('\n'.join([v.name for v in tf.global_variables()]) + '\n')
    return m

  # def create_train(FLAGS):
  #   return self.create_model(FLAGS, False, True)

  # def create_valid(FLAGS):
  #   return self.create_model(FLAGS, False, False, reuse=True)

