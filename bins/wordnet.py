#coding: utf-8
import sys, os, random, copy
import tensorflow as tf
from base import BaseManager, logger
from core.utils.vocabulary import WordNetSynsetVocabulary, WordNetRelationVocabulary
from core.utils import common
from core.utils.dataset import WordNetDataset
import core.models.wordnet

tf.app.flags.DEFINE_string("source_data_dir", "dataset/graph/wordnet-mlj12/source", "")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/graph/wordnet-mlj12/processed", "")

tf.app.flags.DEFINE_string("model_type", "Baseline", "")
tf.app.flags.DEFINE_string("syn_vocab_data", "wordnet-mlj12-definitions.txt", "")
tf.app.flags.DEFINE_string("rel_vocab_data", "wordnet-mlj12-train.txt", "")
tf.app.flags.DEFINE_string("train_data", "wordnet-mlj12-train.txt", "")
tf.app.flags.DEFINE_string("dev_data", "wordnet-mlj12-valid.txt", "")
tf.app.flags.DEFINE_string("test_data", "wordnet-mlj12-test.txt", "")

tf.app.flags.DEFINE_integer("syn_size", None, "")
tf.app.flags.DEFINE_integer("rel_size", None, "")
tf.app.flags.DEFINE_integer("max_rows", None, 
                            "Maximum number of rows to be used as train data.")

tf.app.flags.DEFINE_integer("batch_size", 1000, "")
tf.app.flags.DEFINE_integer("hidden_size", 100, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("keep_prob", 0.75, "Dropout rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("negative_sampling_rate", 1.0, "")
tf.app.flags.DEFINE_integer("max_epoch", 100, "")

tf.app.flags.DEFINE_boolean("share_embedding", False, "Whether to share syn/rel embedding between subjects and objects")

#tf.app.flags.DEFINE_string("loss_type", '', "Whether to share syn/rel embedding between subjects and objects")


class WordNetManager(BaseManager):
  def __init__(self, FLAGS, sess):
    super(WordNetManager, self).__init__(FLAGS, sess)
    self.model_type = getattr(core.models.wordnet, FLAGS.model_type)
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
  def create_model(self, FLAGS, mode, reuse=False):
    if mode == 'train':
      do_update = True
    elif mode == 'dev' or mode == 'test':
      do_update = False
    else:
      raise ValueError("The argument \'mode\' must be \'train\', \'dev\', or \'test\'.")
    summary_path = os.path.join(self.SUMMARIES_PATH, mode)

    with tf.variable_scope("Model", reuse=reuse):
      m = self.model_type(self.sess, FLAGS, do_update,
                          node_vocab=self.syn_vocab, 
                          edge_vocab=self.rel_vocab,
                          summary_path=summary_path)
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

  @common.timewatch(logger)
  def train(self):
    FLAGS = self.FLAGS
    train_data = WordNetDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.train_data, self.syn_vocab, self.rel_vocab, FLAGS.max_rows
    )
    dev_data = WordNetDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.dev_data, self.syn_vocab, self.rel_vocab
    )
    test_data = WordNetDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir,
      FLAGS.test_data, self.syn_vocab, self.rel_vocab
    )
    with tf.name_scope('train'):
      mtrain = self.create_model(FLAGS, 'train', reuse=False)

    with tf.name_scope('dev'):
      config.negative_sampling_rate = 0.0
      mvalid = self.create_model(FLAGS, 'dev', reuse=True)

    if mtrain.epoch.eval() == 0:
      logger.info("(train, dev, test) = (%d, %d, %d)" % (train_data.size, dev_data.size, test_data.size))
      logger.info("(Synset, Relation) = (%d, %d)" % (self.syn_vocab.size, self.rel_vocab.size))

    for epoch in xrange(mtrain.epoch.eval(), FLAGS.max_epoch):
      logger.info("Epoch %d: Start training." % epoch)
      epoch_time, step_time, train_loss = mtrain.train_or_valid(train_data, FLAGS.batch_size, do_shuffle=True)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %f" % (epoch, epoch_time, step_time, train_loss))

      epoch_time, step_time, valid_loss = mvalid.train_or_valid(dev_data, FLAGS.batch_size)
      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %f" % (epoch, epoch_time, step_time, valid_loss))

      mtrain.add_epoch()
      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      self.saver.save(self.sess, checkpoint_path, global_step=mtrain.epoch)
      if (epoch + 1) % 10 == 0:
        results = mvalid.test(test_data, FLAGS.batch_size)
        results, ranks, mrr, hits_10 = results
        logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (mtrain.epoch.eval(), mrr, hits_10))

  @common.timewatch(logger)
  def test(self, test_data=None, mtest=None):
    FLAGS = self.FLAGS
    if not test_data:
      test_data = WordNetDataset(
        FLAGS.source_data_dir, FLAGS.processed_data_dir,
        FLAGS.test_data, self.syn_vocab, self.rel_vocab, FLAGS.max_rows
      )
    
    with tf.name_scope('test'):
      if not mtest:
        mtest= self.create_model(FLAGS, 'test', reuse=False)
      results = mtest.test(test_data, FLAGS.batch_size)
      results, mrr, hits_10 = results
    logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (mtest.epoch.eval(), mrr, hits_10))
    return results, mrr, hits_10

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
    manager = WordNetManager(FLAGS, sess)
    manager.create_dir()

    if FLAGS.mode == "train":
      manager.save_config()
      manager.train()
    elif FLAGS.mode == "test":
      manager.test()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
