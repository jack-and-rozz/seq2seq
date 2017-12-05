#coding: utf-8
import sys, os, random, copy, socket, collections, time
from pprint import pprint
import tensorflow as tf
import numpy as np
import multiprocessing as mp

from base import BaseManager, logger
from core.utils import common, tf_utils
import core.models.wikiP2D.mtl as model
from core.dataset.wikiP2D import WikiP2DDataset, DemoBatch
from core.dataset.coref import CoNLL2012CorefDataset
from core.vocabulary.base import VocabularyWithEmbedding, PredefinedCharVocab

class MTLManager(BaseManager):
  @common.timewatch()
  def __init__(self, FLAGS, sess):
    # If embeddings are not pretrained, make trainable True.
    super(MTLManager, self).__init__(FLAGS, sess)
    self.config.trainable_emb = self.config.trainable_emb or not self.config.use_pretrained_emb
    self.model_type = getattr(model, self.config.model_type)
    self.mode = FLAGS.mode
    self.checkpoint_path = FLAGS.checkpoint_path
    self.reuse = None
    self.summary_writer = None

    self.use_wikiP2D = True if self.config.graph_task or self.config.desc_task else False

    self.w_vocab = None
    self.c_vocab = None
    self.r_vocab = None
    self.o_vocab = None

    if self.config.use_pretrained_emb and len(self.config.embeddings) > 0:
      self.w_vocab = VocabularyWithEmbedding(
        self.config.embeddings,
        source_dir=self.config.embeddings_dir,
        lowercase=self.config.lowercase)
      self.c_vocab = PredefinedCharVocab(
        os.path.join(self.config.embeddings_dir, self.config.char_vocab_path),
        lowercase=self.config.lowercase,
      )

    if self.use_wikiP2D:
      self.w2p_dataset = WikiP2DDataset(
        self.config.w_vocab_size, self.config.c_vocab_size,
        filename=self.config.wikiP2D.dataset,
        lowercase=self.config.lowercase,
        w_vocab=self.w_vocab, c_vocab=self.c_vocab
      )
      self.w_vocab = self.w2p_dataset.w_vocab
      self.c_vocab = self.w2p_dataset.c_vocab
      self.r_vocab = self.w2p_dataset.r_vocab
      self.o_vocab = self.w2p_dataset.o_vocab

    self.coref_dataset = CoNLL2012CorefDataset(
      self.w_vocab, self.c_vocab
    )
    #self.speaker_vocab = self.coref_dataset.speaker_vocab
    self.genre_vocab = self.coref_dataset.genre_vocab

    # Defined after the computational graph is completely constracted.
    self.saver = None

  def get_batch(self, batch_type):
    batches = {'is_training': False}
    if batch_type == 'train':
      do_shuffle = True
      batches['is_training'] = True
      batches['wikiP2D'] = self.w2p_dataset.train.get_batch(
        self.config.wikiP2D.batch_size, do_shuffle=do_shuffle,
        min_sentence_length=None, 
        max_sentence_length=self.config.wikiP2D.max_sent_length.encode,
        n_pos_triples=self.config.wikiP2D.n_triples) if self.use_wikiP2D else None
      batches['coref'] = self.coref_dataset.train.get_batch(
        self.config.coref.batch_size, do_shuffle=do_shuffle)

    elif batch_type == 'valid':
      do_shuffle = False
      batches['wikiP2D'] = self.w2p_dataset.valid.get_batch(
        self.config.wikiP2D.batch_size, do_shuffle=do_shuffle,
        min_sentence_length=None, 
        max_sentence_length=self.config.wikiP2D.max_sent_length.encode,
        n_pos_triples=self.config.wikiP2D.n_triples) if self.use_wikiP2D else None
      batches['coref'] = self.coref_dataset.valid.get_batch(
        self.config.coref.batch_size, do_shuffle=do_shuffle)
    elif batch_type == 'test':
      do_shuffle = False
      batches['wikiP2D'] = self.w2p_dataset.test.get_batch(
        self.config.wikiP2D.batch_size, do_shuffle=do_shuffle,
        min_sentence_length=None, 
        max_sentence_length=self.config.wikiP2D.max_sent_length.encode,
        n_pos_triples=self.config.wikiP2D.n_triples) if self.use_wikiP2D else None
      batches['coref'] = self.coref_dataset.test.get_batch(
        self.config.coref.batch_size, do_shuffle=do_shuffle)
    return batches

  @common.timewatch()
  def create_model(self, config, mode, ckpt=None):
    with tf.variable_scope("Model", reuse=self.reuse):
      m = self.model_type(
        self.sess, config, mode,
        self.w_vocab, self.c_vocab, # for encoder
        self.o_vocab, self.r_vocab, # for graph
        self.genre_vocab, # for coref
        #self.speaker_vocab, self.genre_vocab, # for coref
      )
    if not ckpt:
      ckpt = tf.train.get_checkpoint_state(self.CHECKPOINTS_PATH) 
    if not self.saver:
      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.max_to_keep)
    if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.index'):
      if not self.reuse:
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
      if not self.reuse:
        logger.info("Created model with fresh parameters.")
      tf.global_variables_initializer().run()
      variables_path = self.checkpoint_path + '/variables/variables.list'
      with open(variables_path, 'w') as f:
        f.write('\n'.join([v.name for v in tf.global_variables()]) + '\n')

    if not self.summary_writer:
      self.summary_writer = {mode:tf.summary.FileWriter(os.path.join(self.SUMMARIES_PATH, mode), self.sess.graph) for mode in ['valid', 'test']}

    self.reuse = True
    return m

  @common.timewatch(logger)
  def train(self):
    m = self.create_model(self.config, 'train')

    if m.epoch.eval() == 0:
      if self.use_wikiP2D:
        logger.info("Dataset stats (WikiP2D)")
        logger.info("(train) articles, triples, subjects = (%d, %d, %d)" % (self.w2p_dataset.train.size))
        logger.info("(valid) articles, triples, subjects = (%d, %d, %d)" % (self.w2p_dataset.valid.size))
        logger.info("(test)  articles, triples, subjects = (%d, %d, %d)" % (self.w2p_dataset.test.size))
    for epoch in xrange(m.epoch.eval(), self.config.max_epoch):
      batches = self.get_batch('train')
      epoch_time, step_time, train_loss, summary = m.train_or_valid(batches)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, train_loss))

      #self.summary_writer['train'].add_summary(summary, m.global_step.eval())
      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = m.train_or_valid(batches)
      self.summary_writer['valid'].add_summary(summary, m.global_step.eval())
      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, valid_loss))

      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      if epoch == 0 or (epoch+1) % 1 == 0:
        self.saver.save(self.sess, checkpoint_path, global_step=m.epoch)
        #results, ranks, mrr, hits_10 = mvalid.test(test_data, 20)
        #logger.info("Epoch %d (valid): MRR %f, Hits@10 %f" % (epoch, mrr, hits_10))
      m.add_epoch()

  def self_test(self):
    ##############################################
    ## DEBUG
    mode = 'valid'
    conll_eval_path = {
      'train': 'dataset/coref/source/train.english.v4_auto_conll',
      'valid': 'dataset/coref/source/dev.english.v4_auto_conll',
      'test': 'dataset/coref/source/test.english.v4_gold_conll',
    }
    m = self.create_model(self.config, mode)
    batches = self.get_batch(mode)[m.coref.dataset]
    eval_summary, f1 = m.coref.test(batches, conll_eval_path[mode])
    exit(1)
    ############################################


  @common.timewatch(logger)
  def c_test(self, mode="valid"): # mode: 'valid' or 'test'
    evaluated_checkpoints = set()
    max_f1 = 0.0
    conll_eval_path = {
      'train': 'dataset/coref/source/train.english.v4_auto_conll',
      'valid': 'dataset/coref/source/dev.english.v4_auto_conll',
      'test': 'dataset/coref/source/test.english.v4_gold_conll',
    }
    while True:
      time.sleep(1)
      ckpt = tf.train.get_checkpoint_state(self.CHECKPOINTS_PATH)
      if ckpt and ckpt.model_checkpoint_path and ckpt.model_checkpoint_path not in evaluated_checkpoints:
        # Move it to a temporary location to avoid being deleted by the training supervisor.
        tmp_checkpoint_path = os.path.join(self.CHECKPOINTS_PATH, 
                                           "model.ctmp.ckpt")
        tf_utils.copy_checkpoint(ckpt.model_checkpoint_path, tmp_checkpoint_path)
        m = self.create_model(self.config, mode, ckpt=ckpt)
        print "Found a new checkpoint: %s" % ckpt.model_checkpoint_path
        output_path = self.TESTS_PATH + '/c_test.%s.ep%02d' % (mode, m.epoch.eval())
        batches = self.get_batch(mode)[m.coref.dataset]

        with open(output_path, 'w') as f:
          sys.stdout = f
          eval_summary, f1 = m.coref.test(batches, conll_eval_path[mode])
          if f1 > max_f1:
            max_f1 = f1
            max_checkpoint_path = os.path.join(self.CHECKPOINTS_PATH, 
                                               "model.cmax.ckpt")
            tf_utils.copy_checkpoint(tmp_checkpoint_path, max_checkpoint_path)
            print "Current max F1: {:.2f}".format(max_f1)
          sys.stdout = sys.__stdout__

        self.summary_writer[mode].add_summary(eval_summary, m.global_step.eval())
        print "Evaluation written to {} at epoch {}".format(self.CHECKPOINTS_PATH, m.global_step.eval())
        evaluated_checkpoints.add(ckpt.model_checkpoint_path)

  @common.timewatch(logger)
  def g_test(self):
    test_data = self.w2p_dataset.test

    if not m:
      m = self.create_model(self.config, 'test')

    batches = self.get_batch('test')[m.dataset]
    summary, res = m.graph.test(batches)
    scores, ranks, mrr, hits_10 = res
    m.summary_writer.add_summary(summary, m.global_step.eval())

    output_path = self.TESTS_PATH + '/g_test.ep%02d' % m.epoch.eval()
    with open(output_path, 'w') as f:
      m.graph.print_results(batches, scores, ranks, output_file=f)

    logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (m.epoch.eval(), mrr, hits_10))

  def demo(self):
    m = self.create_model(self.self.config, 'test')

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
      results = m.test(demo_data, 1)[0][0]
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
      # TODO: set a process that simultaneously evaluate the model at each epoch.
      #       (Some techniques are required to parallely run instance methods...)
      # with tf.device('/cpu:0'):
      #    worker = mp.Process(target=manager.c_test, kwargs={'mode':'valid'})
      #    worker.daemon = True 
      #    worker.start()
      manager.train()
    elif FLAGS.mode == "g_test":
      manager.g_test()
    elif FLAGS.mode == "c_test":
      manager.c_test()
    elif FLAGS.mode == "demo":
      manager.demo()
    elif FLAGS.mode == 'self_test':
      manager.self_test()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  tf.app.run()
