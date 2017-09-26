#coding: utf-8
import sys, os, random, copy
import socket
import tensorflow as tf
from pprint import pprint
import numpy as np

from base import BaseManager, logger
from core.utils import common
import core.models.wikiP2D as model
from core.dataset.wikiP2D import WikiP2DDataset, DemoBatch

tf.app.flags.DEFINE_string("source_data_dir", "dataset/wikiP2D/source", "")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/wikiP2D/processed", "")
tf.app.flags.DEFINE_string("model_type", "WikiP2D", "")
tf.app.flags.DEFINE_string("dataset", "Q5O15000R300.micro.bin", "")

## Hyperparameters
tf.app.flags.DEFINE_integer("max_epoch", 50, "")
tf.app.flags.DEFINE_integer("batch_size", 128, "")
tf.app.flags.DEFINE_integer("w_vocab_size", 30000, "")
tf.app.flags.DEFINE_integer("c_vocab_size", 1000, "")
tf.app.flags.DEFINE_integer("hidden_size", 100, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("in_keep_prob", 1.0, "Dropout rate.")
tf.app.flags.DEFINE_float("out_keep_prob", 0.75, "Dropout rate.")
tf.app.flags.DEFINE_integer("num_layers", 1, "")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("n_triples", 0, "If 0, all positive triples are used per an article.")

## Text processing methods
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_string("encoder_type", "BidirectionalRNNEncoder", "")
tf.app.flags.DEFINE_string("c_encoder_type", "BidirectionalRNNEncoder", "")
tf.app.flags.DEFINE_boolean("cbase", True,  "Whether to make the model character-based or not.")
tf.app.flags.DEFINE_boolean("wbase", True,  "Whether to make the model word-based or not.")

tf.app.flags.DEFINE_boolean("state_is_tuple", True,  "")
tf.app.flags.DEFINE_integer("max_a_sent_length", 40, "")
tf.app.flags.DEFINE_integer("max_d_sent_length", 40, "")
tf.app.flags.DEFINE_integer("max_a_word_length", 0, "")

#tf.app.flags.DEFINE_boolean("share_embedding", False, "Whether to share syn/rel embedding between subjects and objects")

class GraphManager(BaseManager):
  @common.timewatch()
  def __init__(self, FLAGS, sess):
    super(GraphManager, self).__init__(FLAGS, sess)
    self.model_type = getattr(model, FLAGS.model_type)
    self.FLAGS = FLAGS
    self.dataset = WikiP2DDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir, 
      FLAGS.dataset, FLAGS.w_vocab_size, FLAGS.c_vocab_size)

  @common.timewatch()
  def create_model(self, FLAGS, mode, reuse=False, write_summary=True):
    if mode == 'train':
      do_update = True
    elif mode == 'valid' or mode == 'test':
      do_update = False
    else:
      raise ValueError("The argument \'mode\' must be \'train\', \'valid\', or \'test\'.")
    summary_path = os.path.join(self.SUMMARIES_PATH, mode) if write_summary else None

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
      mvalid = self.create_model(config, 'valid', reuse=True)

    if mtrain.epoch.eval() == 0:
      logger.info("(train) articles, triples, subjects = (%d, %d, %d)" % (train_data.size))
      logger.info("(valid) articles, triples, subjects = (%d, %d, %d)" % (valid_data.size))
      logger.info("(test)  articles, triples, subjects = (%d, %d, %d)" % (test_data.size))

    for epoch in xrange(mtrain.epoch.eval(), config.max_epoch):
      #logger.info("Epoch %d: Start training." % epoch)
      epoch_time, step_time, train_loss = mtrain.train_or_valid(train_data, config.batch_size, do_shuffle=True)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %f" % (epoch, epoch_time, step_time, train_loss))

      epoch_time, step_time, valid_loss = mvalid.train_or_valid(valid_data, config.batch_size, do_shuffle=False)
      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %f" % (epoch, epoch_time, step_time, valid_loss))

      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      if epoch == 0 or (epoch+1) % 5 == 0:
        self.saver.save(self.sess, checkpoint_path, global_step=mtrain.epoch)
        #results, ranks, mrr, hits_10 = mvalid.test(test_data, 20)
        #logger.info("Epoch %d (valid): MRR %f, Hits@10 %f" % (epoch, mrr, hits_10))
      mtrain.add_epoch()

  @common.timewatch()
  def print_results(self, data, scores, ranks, output_file=None):
    FLAGS = self.FLAGS
    batches = data.get_batch(FLAGS.batch_size, 
                             max_sentence_length=FLAGS.max_a_sent_length, 
                             n_neg_triples=None, n_pos_triples=None)
    cnt = 0
    if output_file:
      sys.stdout = output_file
    for batch, score_by_batch, ranks_by_batch in zip(batches, scores, ranks): # per a batch
      for batch_by_art, score_by_art, rank_by_art in zip(self.dataset.batch2text(batch), score_by_batch, ranks_by_batch): # per an article
        wa, ca, pts = batch_by_art
        print '<%d>' % cnt
        print  "Article(word):\t%s" % wa
        print  "Article(char):\t%s" % ca
        print  "Triple, Score, Rank:"
        for (r, o), scores, rank in zip(pts, score_by_art, rank_by_art): # per a positive triple
          s = scores[0] # scores = [pos, neg_0, neg_1, ...]
          N = 5
          pos_rank, sorted_idx = rank
          pos_id = self.dataset.o_vocab.name2id(o)
          idx2id = [pos_id] + [x for x in xrange(self.dataset.o_vocab.size) if x != pos_id] # pos_objectを先頭に持ってきているのでidxを並び替え

          top_n_scores = [x for x in sorted_idx[:N]]
          top_n = [self.dataset.o_vocab.id2name(idx2id[x]) for x in sorted_idx[:N]]
          top_n = ", ".join(["%s:%.4f" % (x, score)for x, score in zip(sorted_idx[:N], top_n_scores[:N])])
          print "(%s, %s) - %f, %d, [Top-%d Objects]: %s" % (r, o, s, pos_rank, N, top_n) 
        print
        cnt += 1 
    sys.stdout = sys.__stdout__

  @common.timewatch(logger)
  def test(self, test_data=None, mtest=None):
    FLAGS = self.FLAGS
    if not test_data:
      test_data = self.dataset.test

    with tf.name_scope('test'):
      if not mtest:
        mtest= self.create_model(FLAGS, 'test', reuse=False)
      res = mtest.test(test_data, FLAGS.batch_size)
      scores, ranks, mrr, hits_10 = res

    output_path = self.TESTS_PATH + '/g_test.ep%02d' % mtest.epoch.eval()
    with open(output_path, 'w') as f:
      self.print_results(test_data, scores, ranks, output_file=f)
      #self.print_results(test_data, results, ranks, output_file=f)
   
    logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (mtest.epoch.eval(), mrr, hits_10))

  def demo(self):
    with tf.name_scope('demo'):
      mtest= self.create_model(self.FLAGS, 'test', 
                               reuse=False, write_summary=False)

    # for debug
    parser = common.get_parser()
    def get_inputs():
      article = 'How about making the graph look nicer?'
      link_span = (4, 4)
      return article, link_span

    def get_result(article, link_span):
      article = " ".join(parser(article))
      w_article = self.dataset.w_vocab.sent2ids(article)
      c_article =  self.dataset.c_vocab.sent2ids(article)
      p_triples = [(0, i) for i in xrange(10)]
      batch = {
        'w_articles': [w_article],
        'c_articles': [c_article],
        'link_spans': [link_span],
        'p_triples': [p_triples], #self.dataset.get_all_triples(),
        'n_triples': None
      }
      demo_data = DemoBatch(batch)
      results = mtest.test(demo_data, 1)[0][0]
      results = common.flatten(results)
      def id2text(r, o):
        rid = self.dataset.r_vocab.id2token(r)
        rname = self.dataset.r_vocab.id2name(r)
        rr = "%s(%s)" % (rid, rname)
        oid = self.dataset.o_vocab.id2token(o)
        oname = self.dataset.o_vocab.id2name(o)
        oo = "%s (%s)" % (oid, oname)
        return (rr, oo)
      return [(id2text(r, o), score) for (r, o), score in results]
    print get_result(*get_inputs())
    exit(1)
    #inputs = get_inputs()
    #print inputs
    HOST = '127.0.0.1'
    PORT = 50007
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    while True:
      print '-----------------'
      conn, addr = s.accept()
      print 'Connected by', addr
      data = conn.recv(1024)
      article, start, end = data.split('\t')
      results = get_result(article, (int(start), int(end)))
      print results[:10]
      conn.send(str(results))
      conn.close()
    return

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
    elif FLAGS.mode == "demo":
      manager.demo()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
