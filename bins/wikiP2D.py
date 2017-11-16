#coding: utf-8
import sys, os, random, copy
import socket
import tensorflow as tf
from pprint import pprint
import numpy as np
import collections
from base import BaseManager, logger
from core.utils import common, tf_utils
#import core.models.wikiP2D as model
import core.models.wikiP2D.mtl as model
from core.dataset.wikiP2D import WikiP2DDataset, DemoBatch
from core.dataset.coref import CoNLL2012CorefDataset
from core.vocabulary.base import VocabularyWithEmbedding


tf.app.flags.DEFINE_string("model_type", "MeanLoss", "")
tf.app.flags.DEFINE_string("w2p_dataset", "Q5O15000R300.micro.bin", "")

## Hyperparameters
tf.app.flags.DEFINE_integer("max_epoch", 50, "")
tf.app.flags.DEFINE_integer("batch_size", 128, "")
tf.app.flags.DEFINE_integer("hidden_size", 100, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("in_keep_prob", 1.0, "Dropout rate.")
tf.app.flags.DEFINE_float("out_keep_prob", 0.75, "Dropout rate.")
tf.app.flags.DEFINE_integer("num_layers", 1, "")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_boolean("state_is_tuple", True,  "")

tf.app.flags.DEFINE_boolean("graph_task", False,  "Whether to run graph link predction task.")
tf.app.flags.DEFINE_boolean("desc_task", False,  "Whether to run description generation task.")
tf.app.flags.DEFINE_boolean("coref_task", True,  "Whether to run description coreference resolution task.")


## Text processing methods
tf.app.flags.DEFINE_boolean("cbase", True,  "Whether to make the model character-based or not.")
tf.app.flags.DEFINE_boolean("wbase", True,  "Whether to make the model word-based or not.")
tf.app.flags.DEFINE_integer("w_vocab_size", 50000, "")
tf.app.flags.DEFINE_integer("c_vocab_size", 1000, "")
tf.app.flags.DEFINE_integer("w_embedding_size", 300, "This parameter is ignored when using pretrained embeddings.")
tf.app.flags.DEFINE_integer("c_embedding_size", 8, "")
tf.app.flags.DEFINE_boolean("lowercase", True,  "")
tf.app.flags.DEFINE_boolean("use_pretrained_emb", True,  "")
tf.app.flags.DEFINE_boolean("trainable_emb", False,  "Whether to train pretrained embeddings (if not pretrained, this parameter always becomes True)")
tf.app.flags.DEFINE_string("embeddings", "glove.840B.300d.txt.filtered,turian.50d.txt",  "")

## Coref
tf.app.flags.DEFINE_integer("f_embedding_size", 20, "")
tf.app.flags.DEFINE_integer("max_antecedents", 250, "")
tf.app.flags.DEFINE_integer("max_training_sentences", 50, "")
tf.app.flags.DEFINE_float("mention_ratio", 0.4, "")
tf.app.flags.DEFINE_integer("max_mention_width", 10, "")
tf.app.flags.DEFINE_boolean("use_features", True,  "")
tf.app.flags.DEFINE_boolean("use_metadata", True,  "")
tf.app.flags.DEFINE_boolean("model_heads", True,  "")
tf.app.flags.DEFINE_integer("ffnn_depth", 2, "")

## Graph
tf.app.flags.DEFINE_integer("max_a_sent_length", 40, "")
tf.app.flags.DEFINE_integer("n_triples", 0, "If 0, all positive triples are used per an article.")
##Desc
tf.app.flags.DEFINE_integer("max_d_sent_length", 40, "")

#tf.app.flags.DEFINE_boolean("share_embedding", False, "Whether to share syn/rel embedding between subjects and objects")


## The names of dataset and tasks.

class MTLManager(BaseManager):
  @common.timewatch()
  def __init__(self, FLAGS, sess):
    # If embeddings are not pretrained, make trainable True.
    FLAGS.trainable_emb = FLAGS.trainable_emb or not FLAGS.use_pretrained_emb
    super(MTLManager, self).__init__(FLAGS, sess)
    self.model_type = getattr(model, FLAGS.model_type)
    self.FLAGS = FLAGS
    self.reuse = None

    if FLAGS.use_pretrained_emb:
      emb_files = FLAGS.embeddings.split(',')
      self.w_vocab = VocabularyWithEmbedding(emb_files, lowercase=FLAGS.lowercase)
      self.c_vocab = None
    else:
      self.w_vocab = None
      self.c_vocab = None

    self.w2p_dataset = WikiP2DDataset(
      FLAGS.w_vocab_size, FLAGS.c_vocab_size,
      filename=FLAGS.w2p_dataset,
      lowercase=FLAGS.lowercase,
      w_vocab=self.w_vocab, c_vocab=self.c_vocab
    )
    self.w_vocab = self.w2p_dataset.w_vocab
    self.c_vocab = self.w2p_dataset.c_vocab
    self.r_vocab = self.w2p_dataset.r_vocab
    self.o_vocab = self.w2p_dataset.o_vocab

    self.coref_dataset = CoNLL2012CorefDataset(
      self.w_vocab, self.c_vocab
    )
    self.speaker_vocab = self.coref_dataset.speaker_vocab
    self.genre_vocab = self.coref_dataset.genre_vocab

    # Defined after the computational graph is completely constracted.
    # self.summary_writer = None
    self.summary_writer = {'train': None, 'valid':None, 'test':None}
    self.saver = None

  def get_batch(self, batch_type):
    batches = {}
    FLAGS = self.FLAGS
    if batch_type == 'train':
      do_shuffle = True
      batches['wikiP2D'] = self.w2p_dataset.train.get_batch(
        FLAGS.batch_size, do_shuffle=do_shuffle,
        min_sentence_length=None, max_sentence_length=FLAGS.max_a_sent_length,
      n_pos_triples=FLAGS.n_triples)
      batches['coref'] = self.coref_dataset.train.get_batch(1, do_shuffle=do_shuffle)
    elif batch_type == 'valid':
      do_shuffle = False
      batches['wikiP2D'] = self.w2p_dataset.valid.get_batch(
        FLAGS.batch_size, do_shuffle=do_shuffle,
        min_sentence_length=None, max_sentence_length=FLAGS.max_a_sent_length,
        n_pos_triples=FLAGS.n_triples)
      batches['coref'] = self.coref_dataset.valid.get_batch(1, do_shuffle=do_shuffle)
    elif batch_type == 'test':
      do_shuffle = False
      batches['wikiP2D'] = self.w2p_dataset.test.get_batch(
        FLAGS.batch_size, do_shuffle=do_shuffle,
        min_sentence_length=None, max_sentence_length=None,
        n_pos_triples=FLAGS.n_triples)
      batches['coref'] = self.coref_dataset.test.get_batch(1, do_shuffle=do_shuffle)
    return batches

  @common.timewatch()
  def create_model(self, FLAGS, mode, write_summary=True):
    if mode == 'train':
      config = FLAGS
      do_update = True
    elif mode == 'valid' or mode == 'test':
      config = copy.deepcopy(FLAGS)
      config.in_keep_prob = 1.0
      config.out_keep_prob = 1.0
      do_update = False
    else:
      raise ValueError("The argument \'mode\' must be \'train\', \'valid\', or \'test\'.")
    #summary_path = os.path.join(self.SUMMARIES_PATH, mode) if write_summary else None
    summary_path = self.SUMMARIES_PATH if write_summary else None
    with tf.name_scope(mode):
      with tf.variable_scope("Model", reuse=self.reuse):
        m = self.model_type(
          self.sess, config, do_update, mode,
          self.w_vocab, self.c_vocab, # for encoder
          self.o_vocab, self.r_vocab, # for graph
          self.speaker_vocab, self.genre_vocab, # for coref
        )

    ckpt = tf.train.get_checkpoint_state(self.CHECKPOINTS_PATH)
    if not self.saver:
      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
    if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.index'):
      if not self.reuse:
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
      if not self.reuse:
        logger.info("Created model with fresh parameters.")
      tf.global_variables_initializer().run()
      with open(FLAGS.checkpoint_path + '/variables/variables.list', 'w') as f:
        f.write('\n'.join([v.name for v in tf.global_variables()]) + '\n')
    #if not self.summary_writer:
    if not self.summary_writer[mode]:
      #self.summary_writer = tf.summary.FileWriter(self.SUMMARIES_PATH, self.sess.graph)
      self.summary_writer[mode] = tf.summary.FileWriter(os.path.join(self.SUMMARIES_PATH, mode), self.sess.graph)
    self.reuse = True
    return m

  @common.timewatch(logger)
  def train(self):
    FLAGS = self.FLAGS

    #with tf.name_scope('train'):
    mtrain = self.create_model(FLAGS, 'train')
    #with tf.name_scope('valid'):
    mvalid = self.create_model(FLAGS, 'valid')

    if mtrain.epoch.eval() == 0:
      logger.info("Dataset stats (WikiP2D)")
      logger.info("(train) articles, triples, subjects = (%d, %d, %d)" % (self.w2p_dataset.train.size))
      logger.info("(valid) articles, triples, subjects = (%d, %d, %d)" % (self.w2p_dataset.valid.size))
      logger.info("(test)  articles, triples, subjects = (%d, %d, %d)" % (self.w2p_dataset.test.size))
    for epoch in xrange(mtrain.epoch.eval(), FLAGS.max_epoch):
      batches = self.get_batch('train')
      epoch_time, step_time, train_loss, summary = mtrain.train_or_valid(batches)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, train_loss))

      #self.summary_writer.add_summary(summary, mtrain.global_step.eval())
      self.summary_writer['train'].add_summary(summary, mtrain.global_step.eval())
      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = mvalid.train_or_valid(batches)
      #self.summary_writer.add_summary(summary, mvalid.global_step.eval())
      self.summary_writer['valid'].add_summary(summary, mvalid.global_step.eval())
      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, valid_loss))

      ##debug
      #mtrain.add_epoch()
      #continue
      ##debug

      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      if epoch == 0 or (epoch+1) % 1 == 0:
        self.saver.save(self.sess, checkpoint_path, global_step=mtrain.epoch)
        #results, ranks, mrr, hits_10 = mvalid.test(test_data, 20)
        #logger.info("Epoch %d (valid): MRR %f, Hits@10 %f" % (epoch, mrr, hits_10))
      mtrain.add_epoch()

  @common.timewatch(logger)
  def c_test(self):
    evaluated_checkpoints = set()
    max_f1 = 0.0
      
    while True:
      ckpt = tf.train.get_checkpoint_state(self.CHECKPOINTS_PATH)
      if ckpt and ckpt.model_checkpoint_path and ckpt.model_checkpoint_path not in evaluated_checkpoints:
        # Move it to a temporary location to avoid being deleted by the training supervisor.
        tmp_checkpoint_path = os.path.join(self.CHECKPOINTS_PATH, 
                                           "model.ctmp.ckpt")
        tf_utils.copy_checkpoint(ckpt.model_checkpoint_path, tmp_checkpoint_path)
        mtest = self.create_model(self.FLAGS, 'test')
        print "Found a new checkpoint: %s" % ckpt.model_checkpoint_path
        output_path = self.TESTS_PATH + '/c_test.ep%02d' % mtest.epoch.eval()
        batches = self.get_batch('test')[mtest.coref.dataset]
        #conll_eval_path = 'dataset/coref/source/test.english.dev.english.v4_auto_conll'
        conll_eval_path = 'dataset/coref/source/test.english.v4_gold_conll'
        with open(output_path, 'w') as f:
          sys.stdout = f
          eval_summary, f1 = mtest.coref.test(batches, conll_eval_path)
          sys.stdout = sys.__stdout__

          if f1 > max_f1:
            max_f1 = f1
            max_checkpoint_path = os.path.join(self.CHECKPOINTS_PATH, "model.cmax.ckpt")
            tf_utils.copy_checkpoint(tmp_checkpoint_path, max_checkpoint_path)
            print "Current max F1: {:.2f}".format(max_f1)
        self.summary_writer['test'].add_summary(eval_summary, mtest.global_step.eval())
        #print "Evaluation written to {} at step {}".format(self.CHECKPOINTS_PATH, global_step)
        print "Evaluation written to {} at epoch {}".format(self.CHECKPOINTS_PATH, mtest.global_step.eval())
        evaluated_checkpoints.add(ckpt.model_checkpoint_path)

  @common.timewatch(logger)
  def g_test(self):
    test_data = self.w2p_dataset.test

    #with tf.name_scope('test'):
    if not mtest:
      mtest= self.create_model(FLAGS, 'test')

    batches = self.get_batch('test')[mtest.dataset]
    summary, res = mtest.graph.test(batches)
    scores, ranks, mrr, hits_10 = res
    mtest.summary_writer.add_summary(summary, mtest.global_step.eval())

    output_path = self.TESTS_PATH + '/g_test.ep%02d' % mtest.epoch.eval()
    with open(output_path, 'w') as f:
      #mtest.graph.print_results(batches, scores, ranks, output_file=None)
      mtest.graph.print_results(batches, scores, ranks, output_file=f)

    logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (mtest.epoch.eval(), mrr, hits_10))

  def demo(self):
    with tf.name_scope('demo'):
      mtest= self.create_model(self.FLAGS, 'test')

    # for debug
    parser = common.get_parser()
    def get_inputs():
      article = 'How about making the graph look nicer?'
      link_span = (4, 4)
      return article, link_span

    def get_result(article, link_span):
      article = " ".join(parser(article))
      w_article = self.w_vocab.sent2ids(article)
      c_article =  self.c_vocab.sent2ids(article)
      p_triples = [(0, i) for i in xrange(10)]
      batch = {
        'w_articles': [w_article],
        'c_articles': [c_article],
        'link_spans': [link_span],
        'p_triples': [p_triples], #self.w2p_dataset.get_all_triples(),
        'n_triples': None
      }
      demo_data = DemoBatch(batch)
      results = mtest.test(demo_data, 1)[0][0]
      results = common.flatten(results)
      def id2text(r, o):
        rid = self.r_vocab.id2token(r)
        rname = self.r_vocab.id2name(r)
        rr = "%s(%s)" % (rid, rname)
        oid = self.o_vocab.id2token(o)
        oname = self.o_vocab.id2name(o)
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
    manager = MTLManager(FLAGS, sess)
    manager.create_dir()
    if FLAGS.mode == "train":
      manager.save_config()
      manager.train()
    elif FLAGS.mode == "g_test":
      manager.g_test()
    elif FLAGS.mode == "c_test":
      manager.c_test()
    elif FLAGS.mode == "demo":
      manager.demo()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
