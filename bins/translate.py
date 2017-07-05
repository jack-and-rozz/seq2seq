#coding: utf-8
from __future__ import absolute_import
from __future__ import division

import sys, os, time, random
from logging import FileHandler
import numpy as np
import tensorflow as tf
import nltk
#from tensorflow.python.platform import gfile
#from six.moves import xrange  # pylint: disable=redefined-builtin

import core
from core.utils import common
from core.utils.dataset import Vocabulary, ASPECDataset, EOS_ID
from core.models.base import MultiGPUTrainWrapper

# about dataset
tf.app.flags.DEFINE_string("source_data_dir", "dataset/source", "Data directory")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/processed", "Data directory")
tf.app.flags.DEFINE_string("vocab_data", "train", "")
tf.app.flags.DEFINE_string("train_data", "train", "")
tf.app.flags.DEFINE_string("dev_data", "dev", "")
tf.app.flags.DEFINE_string("test_data", "test", "")
tf.app.flags.DEFINE_string("source_lang", "en", "")
tf.app.flags.DEFINE_string("target_lang", "ja", "")
tf.app.flags.DEFINE_integer("max_train_rows", 2000000, "Maximum number of rows to be used as train data.")

#
tf.app.flags.DEFINE_string("model_type", "Baseline", "")

# about hyperparameters
tf.app.flags.DEFINE_float("keep_prob", 0.5,
                          "the keeping probability of active neurons in dropout")
tf.app.flags.DEFINE_integer("num_samples", 512, "")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Size of each token embedding.")
tf.app.flags.DEFINE_integer("source_vocab_size", 30000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 30000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("max_to_keep", 5, "Number of checkpoints to be kept")
tf.app.flags.DEFINE_integer("max_epoch", 50, "")
tf.app.flags.DEFINE_integer("max_sequence_length", 64, "")
tf.app.flags.DEFINE_float("init_scale", 0.1, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")

# for RNN LM
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_string("seq2seq_type", "BasicSeq2Seq", "Cell type")
tf.app.flags.DEFINE_string("encoder_type", "RNNEncoder", "")
tf.app.flags.DEFINE_string("decoder_type", "RNNDecoder", "")
tf.app.flags.DEFINE_boolean("use_sequence_length", True, "If True, PAD_ID tokens are not input to RNN. (This option shouldn't be used when reversing encoder's inputs.)")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("trainable_source_embedding", True, "")
tf.app.flags.DEFINE_boolean("trainable_target_embedding", True, "")
tf.app.flags.DEFINE_boolean("share_embedding", False, "If true, a decoder uses encoder's embedding (for dialogue)")


## temporal flags (not saved in config)
tf.app.flags.DEFINE_string("mode", "train", "")
tf.app.flags.DEFINE_string("log_file", "train.log", "")
tf.app.flags.DEFINE_string('checkpoint_path', 'models/local/tmp', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer("beam_size", 1, "")


FLAGS = tf.app.flags.FLAGS
TMP_FLAGS = ['mode', 'log_file', 'checkpoint_path', 'beam_size']
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
def create_model(sess, max_sequence_length, forward_only, do_update, reuse=None):
  with tf.variable_scope("Model", reuse=reuse):
    if do_update and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
      m = MultiGPUTrainWrapper(sess, FLAGS, max_sequence_length)
    else:
      model_type = getattr(core.models.base, FLAGS.model_type)
      m = model_type(sess, FLAGS, max_sequence_length, forward_only, do_update)
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

def decode_interact(sess):
  s_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.source_lang, FLAGS.source_vocab_size)
  t_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data,
                       FLAGS.target_lang, FLAGS.target_vocab_size)
  mtest = create_model(sess, FLAGS.max_sequence_length, True, False)
  while True:
    sys.stdout.write("> ",)
    source = sys.stdin.readline()
    source = source.split()
    raw_batch = [(None, s_vocab.to_ids(source), [])]
    _, outputs = mtest.decode(sess, raw_batch)
    output = outputs[0]
    if EOS_ID in output:
      output = output[:output.index(EOS_ID)]
    output = " " .join(t_vocab.to_tokens(output))
    print (output)

def decode_test(sess):
  def get_gold_results(data):
    # read original test corpus (unknown words still remain)
    sources = [data.tokenizer(l) for l in open(data.s_source_path)]
    targets = [data.tokenizer(l) for l in open(data.t_source_path)]

    s_ref_path = '%s/%s/%s.%s' % (FLAGS.checkpoint_path, TESTS_PATH, FLAGS.test_data, FLAGS.source_lang)
    t_ref_path = '%s/%s/%s.%s' % (FLAGS.checkpoint_path, TESTS_PATH, FLAGS.test_data, FLAGS.target_lang)

    if not os.path.exists(s_ref_path):
      with open(s_ref_path, 'w') as f:
        for s in sources:
          f.write(" ".join([str(x) for x in s]) + '\n')
    if not os.path.exists(t_ref_path):
      with open(t_ref_path, 'w') as f:
        for t in targets:
          f.write(" ".join([str(x) for x in t]) + '\n')

    return sources, targets

  def get_decode_results(data, s_vocab, t_vocab):
    max_sequence_length = max(data.largest_bucket)
    mtest = create_model(sess, max_sequence_length, True, False)
    decode_path = '%s/%s/%s.%s.decode.beam%d.ep%d' % (
      FLAGS.checkpoint_path,
      TESTS_PATH,
      FLAGS.test_data, 
      FLAGS.target_lang, 
      FLAGS.beam_size, 
      mtest.epoch.eval()
    )
    if not os.path.exists(decode_path):
      sources = []
      targets = []
      results = []
      for i, raw_batch in enumerate(data.get_batch(FLAGS.batch_size)):
        _, outputs = mtest.decode(raw_batch)
        for b, o in zip(raw_batch, outputs):
          idx, s, t = b
          if EOS_ID in o:
            o = o[:o.index(EOS_ID)]
          source = s_vocab.to_tokens(s)
          target = t_vocab.to_tokens(t)
          result = t_vocab.to_tokens(o)
          sources.append(source)
          targets.append(target)
          results.append(result)
          print "<%d>" % idx
          print (' '.join(source))
          print (' '.join(target))
          print (' '.join(result))
      with open(decode_path, 'w') as f:
        f.write("\n".join([' '.join(x) for x in results]) + "\n")
    else:
      results = [l.replace('\n', '').split() for l in open(decode_path)]
    return results

  data_path = os.path.join(FLAGS.source_data_dir, FLAGS.vocab_data)
  vocab_path = os.path.join(FLAGS.processed_data_dir, FLAGS.vocab_data)
  s_vocab = Vocabulary(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.vocab_data, FLAGS.source_lang, FLAGS.source_vocab_size
  )
  t_vocab = Vocabulary(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.vocab_data, FLAGS.target_lang, FLAGS.target_vocab_size
  )
  test = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.test_data, s_vocab, t_vocab,
    max_sequence_length=None, max_rows=None)
  sources, targets = get_gold_results(test)
  results = get_decode_results(test, s_vocab, t_vocab)
  bleu_score = nltk.translate.bleu_score.corpus_bleu(targets, results)
  logger.info("Number of tests: %d " % test.size)
  logger.info("BLEU Score: %f " % bleu_score)

def train(sess):
  data_path = os.path.join(FLAGS.source_data_dir, FLAGS.vocab_data)
  vocab_path = os.path.join(FLAGS.processed_data_dir, FLAGS.vocab_data)
  s_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data, FLAGS.source_lang, FLAGS.source_vocab_size)
  t_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, FLAGS.vocab_data, FLAGS.target_lang, FLAGS.target_vocab_size)

  logger.info("Reading dataset.")
  train = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.train_data, s_vocab, t_vocab, 
    max_sequence_length=FLAGS.max_sequence_length,
    max_rows=FLAGS.max_train_rows)
  dev = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.dev_data, s_vocab, t_vocab)
  test = ASPECDataset(
    FLAGS.source_data_dir, FLAGS.processed_data_dir, 
    FLAGS.test_data, s_vocab, t_vocab)
  logger.info("(train dev test) = (%d %d %d)" % (train.size, dev.size, test.size))

  with tf.name_scope('train'):
    mtrain = create_model(sess, FLAGS.max_sequence_length, False, True)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path + SUMMARIES_PATH,
                                           sess.graph) 

  with tf.name_scope('dev'):
    mvalid = create_model(sess, FLAGS.max_sequence_length, False, False, reuse=True)
  for epoch in xrange(mtrain.epoch.eval(), FLAGS.max_epoch):
    logger.info("Epoch %d: Start training." % epoch)
    epoch_time, step_time, train_ppx = mtrain.run_batch(train, FLAGS.batch_size, 
                                                        do_shuffle=True)
    logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, ppx %.4f" % (epoch, epoch_time, step_time, train_ppx))

    epoch_time, step_time, valid_ppx = mvalid.run_batch(dev, FLAGS.batch_size)

    logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, ppx %.4f" % (epoch, epoch_time, step_time, valid_ppx))

    mtrain.add_epoch()
    checkpoint_path = FLAGS.checkpoint_path + CHECKPOINTS_PATH + "/model.ckpt"
    mtrain.saver.save(sess, checkpoint_path, global_step=mtrain.epoch)
  pass

def main(_):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    # 乱数シードの設定はgraphの生成後、opsの生成前 (http://qiita.com/yuyakato/items/9a5d80e6c7c41e9a9d22)
    # tf v1.1でlossが同一になるのを確認(2017/5/25)
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    create_dir()

    if FLAGS.mode == "train":
      save_config()
      train(sess)
    elif FLAGS.mode == "decode":
      decode_test(sess)
    elif FLAGS.mode == "decode_interact":
      decode_interact(sess)
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
