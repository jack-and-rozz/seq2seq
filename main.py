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
from tensorflow.python.platform import gfile

from utils import common
from utils.dataset import Vocabulary, ASPECDataset, EOS_ID
import models

# about dataset
tf.app.flags.DEFINE_string("source_data_dir", "dataset/source", "Data directory")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/processed", "Data directory")
tf.app.flags.DEFINE_string("vocab_data", "train", "")
tf.app.flags.DEFINE_string("train_data", "train", "")
tf.app.flags.DEFINE_string("dev_data", "dev", "")
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
tf.app.flags.DEFINE_integer("max_epoch", 20, "")
tf.app.flags.DEFINE_integer("max_sequence_length", 120, "")
tf.app.flags.DEFINE_float("init_scale", 0.1, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")

# for RNN LM
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_string("seq2seq_type", "BasicSeq2Seq", "Cell type")
tf.app.flags.DEFINE_string("encoder_type", "BidirectionalRNNEncoder", "")
tf.app.flags.DEFINE_string("decoder_type", "RNNDecoder", "")
tf.app.flags.DEFINE_boolean("use_sequence_length", True, "If True, PAD_ID tokens are not input to RNN. (This option shouldn't be used when reversing encoder's inputs.)")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")

## temporal flags (not saved in config)
tf.app.flags.DEFINE_string("mode", "train", "")
tf.app.flags.DEFINE_string("log_file", "train.log", "")
tf.app.flags.DEFINE_string('checkpoint_path', 'models/local/tmp', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer("num_gpus", 1, "")


FLAGS = tf.app.flags.FLAGS
BUCKETS = [(i, i) for i in xrange(10, FLAGS.max_sequence_length+1, 10)]
print BUCKETS
TMP_FLAGS = ['mode', 'log_file', 'checkpoint_path', 'num_gpus']
log_file = FLAGS.log_file if FLAGS.log_file else None
logger = common.logManager(handler=FileHandler(log_file)) if log_file else common.logManager()

CHECKPOINTS_PATH = '/checkpoints'
TESTS_PATH = '/tests'
VARIABLES_PATH = '/variables'
SUMMARIES_PATH = '/summaries'
def create_dir():
  if not os.path.exists(FLAGS.checkpoint_path):
    os.makedirs(FLAGS.checkpoint_path)
  if not os.path.exists(FLAGS.checkpoint_path + CHECKPOINTS_PATH):
    os.makedirs(FLAGS.checkpoint_path + CHECKPOINTS_PATH)
  if not os.path.exists(FLAGS.checkpoint_path + TESTS_PATH):
    os.makedirs(FLAGS.checkpoint_path + TESTS_PATH)
  if not os.path.exists(FLAGS.checkpoint_path + VARIABLES_PATH):
    os.makedirs(FLAGS.checkpoint_path + VARIABLES_PATH)
  if not os.path.exists(FLAGS.checkpoint_path + SUMMARIES_PATH):
    os.makedirs(FLAGS.checkpoint_path + SUMMARIES_PATH)

def save_config():
  flags_dir = FLAGS.__dict__['__flags']
  with open(FLAGS.checkpoint_path + '/config', 'w') as f:
    for k,v in flags_dir.items():
      if not k in TMP_FLAGS:
        f.write('%s=%s\n' % (k, str(v)))


@common.timewatch(logger)
def create_model(sess, forward_only, do_update, reuse=None):
  with tf.variable_scope("Model", reuse=reuse):
    model_type = getattr(models, FLAGS.model_type)
    m = model_type(FLAGS, BUCKETS, forward_only, do_update)
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path + '/checkpoints')
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path + '.index'):
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

def decode(sess):
  data_path = os.path.join(FLAGS.source_data_dir, FLAGS.vocab_data)
  vocab_path = os.path.join(FLAGS.processed_data_dir, FLAGS.vocab_data)
  s_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.source_lang, FLAGS.source_vocab_size)
  t_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.target_lang, FLAGS.target_vocab_size)
  test = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.test_data, s_vocab, t_vocab, buckets=BUCKETS)
  logger.info("Number of tests: %d " % test.size)

  mtest = create_model(sess, True, False)
  for i, raw_batch in enumerate(test.get_batch(FLAGS.batch_size)):
    _, outputs = mtest.decode(sess, raw_batch)
    for b, o in zip(raw_batch, outputs):
      idx, s, t = b
      if EOS_ID in o:
        o = o[:o.index(EOS_ID)]
   
      print "%d" % idx
      print " ".join(s_vocab.to_tokens(s))
      print " ".join(t_vocab.to_tokens(t))
      print " ".join(t_vocab.to_tokens(o))
      

  
def train(sess):
  data_path = os.path.join(FLAGS.source_data_dir, FLAGS.vocab_data)
  vocab_path = os.path.join(FLAGS.processed_data_dir, FLAGS.vocab_data)
  s_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.source_lang, FLAGS.source_vocab_size)
  t_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.target_lang, FLAGS.target_vocab_size)

  logger.info("Reading dataset.")
  train = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.train_data, s_vocab, t_vocab, 
    buckets=BUCKETS, max_rows=FLAGS.max_train_rows)
  dev = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.dev_data, s_vocab, t_vocab, buckets=BUCKETS)
  test = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.test_data, s_vocab, t_vocab, buckets=BUCKETS)
  logger.info("(train dev test) = (%d %d %d)" % (train.size, dev.size, test.size))

  with tf.name_scope('train'):
    mtrain = create_model(sess, False, True)
  summary_writer = tf.summary.FileWriter(
    FLAGS.checkpoint_path + SUMMARIES_PATH, sess.graph) 
  with tf.name_scope('dev'):
    mvalid = create_model(sess, False, False, reuse=True)
  
  def run_batch(m, data):
    start_time = time.time()
    loss = 0.0
    for i, raw_batch in enumerate(data.get_batch(FLAGS.batch_size)):
      step_loss = m.step(sess, raw_batch)
      loss += step_loss 
      print i, step_loss
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    ppx = math.exp(loss / (i+1))
    return epoch_time, step_time, ppx

  for epoch in xrange(mtrain.epoch.eval(), FLAGS.max_epoch):
    logger.info("Epoch %d: Start training." % epoch)
    epoch_time, step_time, train_ppx = run_batch(mtrain, train)
    logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, ppx %.4f" % (epoch, epoch_time, step_time, train_ppx))
    epoch_time, step_time, valid_ppx = run_batch(mvalid, dev)
    logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, ppx %.4f" % (epoch, epoch_time, step_time, valid_ppx))

    mtrain.add_epoch(sess)
    checkpoint_path = FLAGS.checkpoint_path + CHECKPOINTS_PATH + "/model.ckpt"
    mtrain.saver.save(sess, checkpoint_path, global_step=mtrain.epoch)
  pass

def main(_):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    gpu_options=tf.GPUOptions(
      allow_growth=False # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    # 乱数シードの設定はgraphの生成後、opsの生成前 (http://qiita.com/yuyakato/items/9a5d80e6c7c41e9a9d22)
    np.random.seed(0)
    tf.set_random_seed(0)
    create_dir()

    if FLAGS.mode == "train":
      save_config()
      train(sess)
    elif FLAGS.mode == "decode":
      decode(sess)
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
    tf.app.run()
