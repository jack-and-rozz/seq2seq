#coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import MeCab
import sys, io, os, codecs, time, itertools, math, random
from logging import FileHandler
import numpy as np
import tensorflow as tf
#from six.moves import xrange  # pylint: disable=redefined-builtin
#from tensorflow.python.platform import gfile

from utils import common
from utils.dataset import Vocabulary, ASPECDataset
import models

# about dataset
tf.app.flags.DEFINE_string("source_data_dir", "dataset/source", "Data directory")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/processed", "Data directory")
tf.app.flags.DEFINE_string("vocab_data", "train", "")
tf.app.flags.DEFINE_string("train_data", "train", "")
tf.app.flags.DEFINE_string("valid_data", "dev", "")
tf.app.flags.DEFINE_string("test_data", "test", "")
tf.app.flags.DEFINE_string("source_lang", "ja", "")
tf.app.flags.DEFINE_string("target_lang", "en", "")
tf.app.flags.DEFINE_integer("max_train_rows", 10000000, "Maximum number of rows to be used as train data.")

#
tf.app.flags.DEFINE_string("model_type", "Baseline", "")

# about hyperparameters
tf.app.flags.DEFINE_float("keep_prob", 0.75,
                          "the keeping probability of active neurons in dropout")
tf.app.flags.DEFINE_integer("num_samples", 512, "")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Size of each token embedding.")
tf.app.flags.DEFINE_integer("source_vocab_size", 30000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 30000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("max_to_keep", 1, "Number of checkpoints to be kept")
tf.app.flags.DEFINE_float("init_scale", 0.1, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")

# for RNN LM
tf.app.flags.DEFINE_string("seq2seq_type", "embedding_attention_seq2seq", "Cell type")
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_boolean("state_is_tuple", True, "The type of RNN states")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")

## temporal flags (not saved in config)
tf.app.flags.DEFINE_string("mode", "train", "")
tf.app.flags.DEFINE_string("log_file", "train.log", "")
tf.app.flags.DEFINE_string('checkpoint_path', 'models/local/tmp', 'Directory to put the training data.')

BUCKETS = [(i, i) for i in xrange(10, 120, 10)]
print BUCKETS
FLAGS = tf.app.flags.FLAGS
TMP_FLAGS = ['mode', 'log_file', 'checkpoint_path']
log_file = FLAGS.log_file if FLAGS.log_file else None
logger = common.logManager(handler=FileHandler(log_file)) if log_file else common.logManager()

def create_dir():
  if not os.path.exists(FLAGS.checkpoint_path):
    os.makedirs(FLAGS.checkpoint_path)
  if not os.path.exists(FLAGS.checkpoint_path + '/checkpoints'):
    os.makedirs(FLAGS.checkpoint_path + '/checkpoints')
  if not os.path.exists(FLAGS.checkpoint_path + '/tests'):
    os.makedirs(FLAGS.checkpoint_path + '/tests')
  if not os.path.exists(FLAGS.checkpoint_path + '/variables'):
    os.makedirs(FLAGS.checkpoint_path + '/variables')

def save_config():
  flags_dir = FLAGS.__dict__['__flags']
  with open(FLAGS.checkpoint_path + '/config', 'w') as f:
    for k,v in flags_dir.items():
      if not k in TMP_FLAGS:
        f.write('%s=%s\n' % (k, str(v)))



@common.timewatch()
def create_model(sess, reuse=None):
  #initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
  #                                            FLAGS.init_scale)
  #with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
  #  m = TaskManager(mode, FLAGS, sess, vocab, logger)
  m = getattr(models, FLAGS.model_type)(FLAGS, BUCKETS)
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path + '/checkpoints')
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path + '.index'):
    logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    logger.info("Created model with fresh parameters.")
    tf.global_variables_initializer().run()
    print '\n'.join([v.name for v in tf.global_variables()])
    with open(FLAGS.checkpoint_path + '/variables/variables.list', 'w') as f:
      f.write('\n'.join([v.name for v in tf.global_variables()]) + '\n')

  return m


def train(sess):
  data_path = os.path.join(FLAGS.source_data_dir, FLAGS.vocab_data)
  vocab_path = os.path.join(FLAGS.processed_data_dir, FLAGS.vocab_data)
  s_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.source_lang, FLAGS.source_vocab_size)
  t_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.target_lang, FLAGS.target_vocab_size)
  #train = ASPECDataset(FLAGS.source_data_dir, FLAGS.target_data_dir, 
  #                     FLAGS.train_data, s_vocab, t_vocab)
  logger.info("Reading dataset")
  train = dev = test = ASPECDataset(FLAGS.source_data_dir, FLAGS.processed_data_dir, 
                                    FLAGS.test_data, s_vocab, t_vocab)
  dataset = common.dotDict({'train': train, 'dev': dev, 'test' : test})
  model = create_model(sess)
  pass

def main(_):
  log_device_placement=False
  with tf.Graph().as_default(), tf.Session(
      config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # 乱数シードの設定はgraphの生成後、opsの生成前 (http://qiita.com/yuyakato/items/9a5d80e6c7c41e9a9d22)
    np.random.seed(0)
    tf.set_random_seed(0)
    create_dir()
    save_config()

    if FLAGS.mode == "train":
      train(sess)
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
    tf.app.run()
