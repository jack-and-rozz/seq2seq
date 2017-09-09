#coding: utf-8
import sys, os, random, copy
import tensorflow as tf
from base import BaseManager, logger

from core.utils import common
import core.models.wikiP2D as model
from core.dataset.wikiP2D import WikiP2DDataset

tf.app.flags.DEFINE_string("source_data_dir", "dataset/wikiP2D/source", "")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/wikiP2D/processed", "")
tf.app.flags.DEFINE_string("model_type", "WikiP2D", "")
tf.app.flags.DEFINE_string("dataset", "Q5O15000R300.micro.bin", "")

## Hyperparameters
tf.app.flags.DEFINE_integer("max_epoch", 100, "")
tf.app.flags.DEFINE_integer("batch_size", 100, "")
tf.app.flags.DEFINE_float("negative_sampling_rate", 1.0, "")
tf.app.flags.DEFINE_integer("w_vocab_size", 30000, "")
tf.app.flags.DEFINE_integer("c_vocab_size", 1000, "")
tf.app.flags.DEFINE_integer("hidden_size", 100, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("in_keep_prob", 1.0, "Dropout rate.")
tf.app.flags.DEFINE_float("out_keep_prob", 0.75, "Dropout rate.")
tf.app.flags.DEFINE_integer("num_layers", 1, "")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")

## Text processing methods
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_string("encoder_type", "BidirectionalRNNEncoder", "")
tf.app.flags.DEFINE_boolean("cbase", True,  "Whether to make the model character-based or not.")
tf.app.flags.DEFINE_boolean("wbase", True,  "Whether to make the model word-based or not.")

tf.app.flags.DEFINE_boolean("state_is_tuple", True,  "")
tf.app.flags.DEFINE_integer("max_sentence_length", 40, "")
tf.app.flags.DEFINE_integer("max_word_length", 0, "")

#tf.app.flags.DEFINE_boolean("share_embedding", False, "Whether to share syn/rel embedding between subjects and objects")


class GraphManager(BaseManager):
  @common.timewatch(logger)
  def __init__(self, FLAGS, sess):
    super(GraphManager, self).__init__(FLAGS, sess)
    self.model_type = getattr(model, FLAGS.model_type)
    self.FLAGS = FLAGS
    self.dataset = WikiP2DDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir, 
      FLAGS.dataset, FLAGS.w_vocab_size, FLAGS.c_vocab_size)

  @common.timewatch(logger)
  def create_model(self, FLAGS, mode, reuse=False):
    if mode == 'train':
      do_update = True
    elif mode == 'valid' or mode == 'test':
      do_update = False
    else:
      raise ValueError("The argument \'mode\' must be \'train\', \'valid\', or \'test\'.")
    summary_path = os.path.join(self.SUMMARIES_PATH, mode)

    with tf.variable_scope("Model", reuse=reuse):
      m = self.model_type(
        self.sess, FLAGS, do_update,
        self.dataset.w_vocab, self.dataset.c_vocab,
        self.dataset.o_vocab, self.dataset.r_vocab,
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
    config = self.FLAGS
    train_data = self.dataset.train
    #train_data = self.dataset.test
    valid_data = self.dataset.valid
    test_data = self.dataset.test
    
    with tf.name_scope('train'):
      mtrain = self.create_model(config, 'train', reuse=False)

    with tf.name_scope('valid'):
      config.negative_sampling_rate = 0.0
      mvalid = self.create_model(config, 'valid', reuse=True)
      config.negative_sampling_rate = 1.0
    print mtrain.epoch.eval()
    if mtrain.epoch.eval() == 0:
      logger.info("(train) articles, triples, subjects = (%d, %d, %d)" % (train_data.size))
      logger.info("(valid) articles, triples, subjects = (%d, %d, %d)" % (valid_data.size))
      logger.info("(test)  articles, triples, subjects = (%d, %d, %d)" % (test_data.size))

    for epoch in xrange(mtrain.epoch.eval(), config.max_epoch):
      logger.info("Epoch %d: Start training." % epoch)
      epoch_time, step_time, train_loss = mtrain.train_or_valid(train_data, config.batch_size, do_shuffle=True)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %f" % (epoch, epoch_time, step_time, train_loss))

      epoch_time, step_time, valid_loss = mvalid.train_or_valid(valid_data, config.batch_size, do_shuffle=False)
      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %f" % (epoch, epoch_time, step_time, valid_loss))

      mtrain.add_epoch()
      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      self.saver.save(self.sess, checkpoint_path, global_step=mtrain.epoch)
      #if (epoch + 1) % 10 == 0:
      #  results = mvalid.test(valid_data, config.batch_size)
      #  results, ranks, mrr, hits_10 = results
      #  logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (mtrain.epoch.eval(), mrr, hits_10))

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
      results, ranks, mrr, hits_10 = results
    logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (mtest.epoch.eval(), mrr, hits_10))

@common.timewatch(logger)
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
    elif FLAGS.mode == "test":
      manager.test()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
