#coding: utf-8

from pprint import pprint
import sys, os, random
import numpy as np
import tensorflow as tf

#from tensorflow.python.platform import gfile
#from six.moves import xrange  # pylint: disable=redefined-builtin

from core.utils import common
random.seed(0)
np.random.seed(0)


CONF_NAME = 'experiments.conf'

tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/local/wikidata/tmp', 
                           'Directory to save the trained model.')
tf.app.flags.DEFINE_string("mode", "train", "")
tf.app.flags.DEFINE_string("config_type", "main", "")
tf.app.flags.DEFINE_string("config_path", "configs/" + CONF_NAME, "")

if not os.path.exists(tf.app.flags.FLAGS.checkpoint_path):
  os.makedirs(tf.app.flags.FLAGS.checkpoint_path)

class ManagerBase(object):
  def __init__(self, FLAGS, sess):
    self.FLAGS = FLAGS
    FLAGS = self.FLAGS
    self.checkpoints_path = FLAGS.checkpoint_path +'/checkpoints'
    self.tests_path = FLAGS.checkpoint_path + '/tests'
    self.summaries_path = FLAGS.checkpoint_path + '/summaries'
    self.sess = sess
    self.create_dir()
    
    config_stored_path = os.path.join(FLAGS.checkpoint_path, CONF_NAME)
    if os.path.exists(config_stored_path): 
      config_read_path = config_stored_path
      config = common.get_config(config_read_path)
    else: 
      config_read_path = FLAGS.config_path
      config = common.get_config(config_read_path)[FLAGS.config_type]
      with open(config_stored_path, 'w') as f:
        sys.stdout = f
        common.print_config(config)
        sys.stdout = sys.__stdout__
    self.config = common.recDotDict(config)
    sys.stderr.write(str(config))

  def create_dir(self):
    FLAGS = self.FLAGS
    if not os.path.exists(FLAGS.checkpoint_path):
      os.makedirs(FLAGS.checkpoint_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)


