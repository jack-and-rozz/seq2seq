#coding: utf-8
import sys, os, random
import tensorflow as tf
from base import BaseManager, logger
from core.utils.vocabulary import WordNetSynsetVocabulary, WordNetRelationVocabulary
from core.utils import common
from core.utils.dataset import WordNetDataset
import core.models.graph

tf.app.flags.DEFINE_string("source_data_dir", "dataset/graph/wordnet-mlj12/source", "")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/graph/wordnet-mlj12/processed", "")

tf.app.flags.DEFINE_string("syn_vocab_data", "wordnet-mlj12-definitions.txt", "")
tf.app.flags.DEFINE_string("rel_vocab_data", "wordnet-mlj12-train.txt", "")
tf.app.flags.DEFINE_string("train_data", "wordnet-mlj12-train.txt", "")
tf.app.flags.DEFINE_string("dev_data", "wordnet-mlj12-valid.txt", "")
tf.app.flags.DEFINE_string("test_data", "wordnet-mlj12-test.txt", "")

tf.app.flags.DEFINE_integer("syn_size", None, "")
tf.app.flags.DEFINE_integer("rel_size", None, "")
tf.app.flags.DEFINE_integer("batch_size", 50, "")
tf.app.flags.DEFINE_integer("hidden_size", 50, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("max_epoch", 50, "")


class GraphManager(BaseManager):
  def __init__(self, FLAGS, sess):
    super(GraphManager, self).__init__(FLAGS, sess)
    self.model_type = getattr(core.models.graph, FLAGS.model_type)
    self.FLAGS = FLAGS
    self.syn_vocab = WordNetSynsetVocabulary(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.syn_vocab_data, FLAGS.syn_size,
    )
    self.rel_vocab = WordNetRelationVocabulary(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.rel_vocab_data, FLAGS.rel_size,
    )
  @common.timewatch(logger)
  def create_model(self, FLAGS, reuse):
    do_update = not reuse
    with tf.variable_scope("Model", reuse=reuse):
      m = self.model_type(self.sess, FLAGS, do_update,
                          syn_vocab=self.syn_vocab, rel_vocab=self.rel_vocab)
    ckpt = tf.train.get_checkpoint_state(self.CHECKPOINTS_PATH)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
    if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.index'):
      if reuse==None:
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
      if reuse==None:
        logger.info("Created model with fresh parameters.")
      tf.global_variables_initializer().run()
      with open(FLAGS.checkpoint_path + '/variables/variables.list', 'w') as f:
        f.write('\n'.join([v.name for v in tf.global_variables()]) + '\n')
    return m

  def train(self):
    FLAGS = self.FLAGS
    train = WordNetDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.train_data, self.syn_vocab, self.rel_vocab
    )
    dev = WordNetDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.dev_data, self.syn_vocab, self.rel_vocab
    )
    test = WordNetDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.test_data, self.syn_vocab, self.rel_vocab
    )
    with tf.name_scope('train'):
      mtrain = self.create_model(FLAGS, False)
    with tf.name_scope('dev'):
      mvalid = self.create_model(FLAGS, True)

    for epoch in xrange(mtrain.epoch.eval(), FLAGS.max_epoch):
      logger.info("Epoch %d: Start training." % epoch)
      epoch_time, step_time, train_ppx = mtrain.run_batch(
        train, FLAGS.batch_size, do_shuffle=True)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %.4f" % (epoch, epoch_time, step_time, train_ppx))

      epoch_time, step_time, valid_ppx = mvalid.run_batch(dev, FLAGS.batch_size)

      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %.4f" % (epoch, epoch_time, step_time, valid_ppx))

      mtrain.add_epoch()
      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      self.saver.save(self.sess, checkpoint_path, global_step=mtrain.epoch)



def main(_):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )
  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.set_random_seed(0)
    FLAGS = tf.app.flags.FLAGS
    manager = GraphManager(FLAGS, sess)
    manager.create_dir()

    if FLAGS.mode == "train":
      manager.save_config()
      manager.train()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
