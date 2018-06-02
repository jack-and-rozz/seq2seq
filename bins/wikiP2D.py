#coding: utf-8
import sys, os, random, copy, socket, collections, time, re, argparse
from pprint import pprint
import tensorflow as tf
import numpy as np

from base import ManagerBase
from core.utils import common, tf_utils
import core.dataset 
import core.models.wikiP2D.mtl as mtl_model
from core.models.wikiP2D.coref.demo import run_model

from core.vocabulary.base import VocabularyWithEmbedding, PredefinedCharVocab
from tensorflow.contrib.tensorboard.plugins import projector

class MTLManager(ManagerBase):
  @common.timewatch()
  def __init__(self, args, sess):
    # If embeddings are not pretrained, make trainable True.
    super(MTLManager, self).__init__(args, sess)
    self.config = config = self.load_config(args)

    #self.port = int(args.port) if args.port.isdigit() else None
    self.vocab = common.dotDict()

    # Load pretrained embeddings.
    self.vocab.word = VocabularyWithEmbedding(
      config.embeddings_conf, config.w_vocab_size,
      lowercase=config.lowercase,
      normalize_digits=config.normalize_digits
    )
    self.vocab.char = PredefinedCharVocab(
      config.char_vocab_path, config.c_vocab_size,
      lowercase=False,
    )

    # Load Dataset.
    self.dataset = common.recDotDict()
    for k, v in config.tasks.items():
      dataset_type = getattr(core.dataset, v.dataset.dataset_type)
      if dataset_type == core.dataset.WikiP2DDataset:
        self.dataset[k] = dataset_type(v.dataset, self.vocab)
        #self.vocab.rel = self.dataset[k].r_vocab
        #self.vocab.obj = self.dataset[k].o_vocab
      elif dataset_type == core.dataset.CoNLL2012CorefDataset:
        self.dataset[k] = dataset_type(v.dataset, self.vocab)
        self.vocab.genre = self.dataset[k].genre_vocab
      else:
        raise ValueError('Dataset type %s is undefined.' % t.dataset.dataset_type)
    self.use_coref = 'coref' in self.config.tasks
    self.use_wikiP2D = 'wikiP2D' in self.config.tasks

  def get_batch(self, batch_type):
    batches = {'is_training': False}
    if batch_type == 'train':
      do_shuffle = True
      batches['is_training'] = True
      batches['wikiP2D'] = self.dataset.wikiP2D.train.get_batch(
        self.config.tasks.wikiP2D.batch_size, do_shuffle=True,
        min_sent_len=None, 
        max_sent_len=self.config.tasks.wikiP2D.max_sent_length.encode,
        n_pos_triples=self.config.tasks.wikiP2D.n_triples) if self.use_wikiP2D else None
      batches['coref'] = self.dataset.coref.train.get_batch(
        self.config.tasks.coref.batch_size, do_shuffle=True) if self.use_coref else None

    elif batch_type == 'valid':
      batches['wikiP2D'] = self.dataset.wikiP2D.valid.get_batch(
        self.config.tasks.wikiP2D.batch_size, do_shuffle=False,
        min_sent_len=None, 
        max_sent_len=self.config.tasks.wikiP2D.max_sent_length.encode,
        n_pos_triples=None) if self.use_wikiP2D else None
      batches['coref'] = self.dataset.coref.valid.get_batch(
        self.config.tasks.coref.batch_size, 
        do_shuffle=False) if self.use_coref else None
    elif batch_type == 'test':
      batches['wikiP2D'] = self.dataset.wikiP2D.test.get_batch(
        self.config.tasks.wikiP2D.batch_size, do_shuffle=False,
        min_sent_len=None, 
        max_sent_len=self.config.tasks.wikiP2D.max_sent_length.encode,
        n_pos_triples=None, n_neg_triples=None) if self.use_wikiP2D else None
      batches['coref'] = self.dataset.coref.test.get_batch(
        self.config.tasks.coref.batch_size, 
        do_shuffle=False) if self.use_coref else None
    return batches

  @common.timewatch()
  def create_model(self, config, mode, checkpoint_path=None):
    mtl_model_type = getattr(mtl_model, config.model_type)
    m = mtl_model_type(self.sess, config, self.vocab) # Define computation graph

    if not checkpoint_path:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.max_to_keep)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      self.logger.info("Reading model parameters from %s" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      self.logger.info("Created model with fresh parameters.")
      tf.global_variables_initializer().run()

    variables_path = self.root_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      f.write('\n'.join(variable_names) + '\n')

    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    return m

  def debug(self):
    # coref = self.dataset.coref
    # for batch in self.dataset.coref.valid.get_batch(self.config.tasks.coref.batch_size, do_shuffle=False):
    #   pprint(batch)

    batches = self.dataset.wikiP2D.valid.get_batch(self.config.tasks.wikiP2D.batch_size, do_shuffle=False)

    for i, batch in enumerate(batches):
      print ('----------')
      for j ,k in enumerate(batch):
        print ('<%s>' % k)
        pprint(batch[k])
      exit(1)


  def train(self):
    m = self.create_model(self.config, 'train')
    if m.epoch.eval() == 0:
      if self.use_coref:
        self.logger.info("Dataset stats (CoNLL 2012)")
        self.logger.info("train, valid, test = (%d, %d, %d)" % (self.dataset.coref.train.size, self.dataset.coref.valid.size, self.dataset.coref.test.size))

      if self.use_wikiP2D:
        self.logger.info("Dataset stats (WikiP2D)")
        self.logger.info("(train) articles, triples, subjects = (%d, %d, %d)" % (self.dataset.wikiP2D.train.size))
        self.logger.info("(valid) articles, triples, subjects = (%d, %d, %d)" % (self.dataset.wikiP2D.valid.size))
        self.logger.info("(test)  articles, triples, subjects = (%d, %d, %d)" % (self.dataset.wikiP2D.test.size))

    for epoch in range(m.epoch.eval(), self.config.max_epoch):
      batches = self.get_batch('train')
      epoch_time, step_time, train_loss, summary = m.train(batches, summary_writer=self.summary_writer)
      self.logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, train_loss))

      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = m.valid(batches)
      self.summary_writer.add_summary(summary, m.global_step.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, valid_loss))

      checkpoint_path = self.checkpoints_path + "/model.ckpt"
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
    #batches = self.get_batch(mode)[m.coref.dataset]
    batches = self.get_batch(mode)[m.graph.dataset]
    for i, b in enumerate(batches):
      print(('======== %02d ==========' % i))
      m.graph.print_batch(b)
    #eval_summary, f1 = m.coref.test(batches, conll_eval_path[mode])
    exit(1)
    ############################################


  def c_test(self, mode="valid"): # mode: 'valid' or 'test'
    conll_eval_path = {
      'train': 'dataset/coref/source/train.english.v4_auto_conll',
      'valid': 'dataset/coref/source/dev.english.v4_auto_conll',
      'test': 'dataset/coref/source/test.english.v4_gold_conll',
    }
    tmp_checkpoint_path = os.path.join(self.checkpoints_path, "model.ctmp.ckpt")
    max_checkpoint_path = os.path.join(self.checkpoints_path, "model.cmax.ckpt")

    self.config.desc_task = False
    self.config.graph_task = False
    self.config.adv_task = False

    # Retry evaluation if the best checkpoint is already found.
    if os.path.exists(max_checkpoint_path + '.index'):
      sys.stderr.write('Found a checkpoint {}.\n'.format(max_checkpoint_path))
      m = self.create_model(self.config, mode, 
                            checkpoint_path=max_checkpoint_path)
      batches = self.get_batch(mode)[m.coref.dataset]
      eval_summary, f1, results = m.coref.test(batches, conll_eval_path[mode])
      output_path = self.tests_path + '/c_test.%s.ep%02d.detailed' % (mode, m.epoch.eval())
      sys.stderr.write('Output the predicted and gold clusters to {}.\n'.format(output_path))
      with open(output_path, 'w') as f:
        sys.stdout = f
        m.coref.print_results(results)
        sys.stdout = sys.__stdout__
      return

    evaluated_checkpoints = set()
    max_f1 = 0.0
    # Evaluate each checkpoint while training and save the best one.
    while True:
      time.sleep(1)
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

      if checkpoint_path and checkpoint_path not in evaluated_checkpoints:
        # Move it to a temporary location to avoid being deleted by the training supervisor.
        tf_utils.copy_checkpoint(checkpoint_path, tmp_checkpoint_path)
        m = self.create_model(self.config, mode, checkpoint_path=checkpoint_path)
        print(("Found a new checkpoint: %s" % checkpoint_path))
        batches = self.get_batch(mode)[m.coref.dataset]
        output_path = self.tests_path + '/c_test.%s.ep%02d' % (mode, m.epoch.eval())
        with open(output_path, 'w') as f:
          sys.stdout = f
          eval_summary, f1, results = m.coref.test(batches, conll_eval_path[mode])
          if f1 > max_f1:
            max_f1 = f1
            tf_utils.copy_checkpoint(tmp_checkpoint_path, max_checkpoint_path)
            print(("Current max F1: {:.2f}".format(max_f1)))
          sys.stdout = sys.__stdout__

        self.summary_writer.add_summary(eval_summary, m.global_step.eval())
        print(("Evaluation written to {} at epoch {}".format(self.checkpoints_path, m.epoch.eval())))
        evaluated_checkpoints.add(checkpoint_path)

  def c_demo(self):
    max_checkpoint_path = os.path.join(self.checkpoints_path, "model.cmax.ckpt")
    ckpt_path = None
    if os.path.exists(max_checkpoint_path + '.index'):
      sys.stderr.write('Found a checkpoint {}.\n'.format(max_checkpoint_path))
      ckpt_path = max_checkpoint_path

    self.config.graph_task = False
    self.config.desc_task = False
    m = self.create_model(self.config, checkpoint_path=ckpt_path)
    eval_data = [d for d in self.get_batch('valid')['coref']]
    run_model(m, eval_data, self.port)

  def g_test(self):

    self.config.coref_task = False
    self.config.desc_task = False
    m = self.create_model(self.config, 'test')

    batches = self.get_batch('test')[m.graph.dataset]
    summary, res = m.graph.test(batches)
    scores, ranks, mrr, hits_10 = res
    self.summary_writer.add_summary(summary, m.global_step.eval())

    output_path = self.tests_path + '/g_test.ep%02d' % m.epoch.eval()
    with open(output_path, 'w') as f:
      m.graph.print_results(batches, scores, ranks, 
                            output_file=f, batch2text=self.dataset.wikiP2D.batch2text)

    self.logger.info("Epoch %d (test): MRR %f, Hits@10 %f" % (m.epoch.eval(), mrr, hits_10))

  def g_demo(self):
    m = self.create_model(self.self.config, 'test')

    # for debug
    parser = common.get_parser()
    def get_inputs():
      article = 'How about making the graph look nicer?'
      link_span = (4, 4)
      return article, link_span

    def get_result(article, link_span):
      article = " ".join(parser(article))
      w_article = self.vocab.word.sent2ids(article)
      c_article =  self.vocab.char.sent2ids(article)
      p_triples = [(0, i) for i in range(10)]
      batch = {
        'w_articles': [w_article],
        'c_articles': [c_article],
        'link_spans': [link_span],
        'p_triples': [p_triples], #self.dataset.wikiP2D.get_all_triples(),
        'n_triples': None
      }
      demo_data = DemoBatch(batch)
      results = m.test(demo_data, 1)[0][0]
      results = common.flatten(results)
      def id2text(r, o):
        rid = self.vocab.rel.id2token(r)
        rname = self.vocab.rel.id2name(r)
        rr = "%s(%s)" % (rid, rname)
        oid = self.vocab.obj.id2token(o)
        oname = self.vocab.obj.id2name(o)
        oo = "%s (%s)" % (oid, oname)
        return (rr, oo)
      return [(id2text(r, o), score) for (r, o), score in results]
    print((get_result(*get_inputs())))
    exit(1)
    #inputs = get_inputs()
    #print inputs
    HOST = '127.0.0.1'
    PORT = 50007
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    while True:
      print('-----------------')
      conn, addr = s.accept()
      print(('Connected by', addr))
      data = conn.recv(1024)
      article, start, end = data.split('\t')
      results = get_result(article, (int(start), int(end)))
      print((results[:10]))
      conn.send(str(results))
      conn.close()
    return

def main(args):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.set_random_seed(0)
    args = args
    manager = MTLManager(args, sess)
    getattr(manager, args.mode)()
    return 
    if args.mode == "train":
      # TODO: set a process that simultaneously evaluate the model at each epoch.
      #       (Some techniques are required to parallely run instance methods...)
      # with tf.device('/cpu:0'):
      #    worker = mp.Process(target=manager.c_test, kwargs={'mode':'valid'})
      #    worker.daemon = True 
      #    worker.start()
      manager.train()
    elif args.mode == "g_test":
      manager.g_test()
    elif args.mode == "c_test":
      manager.c_test()
    elif args.mode == "g_demo":
      manager.g_demo()
    elif args.mode == "c_demo":
      manager.c_demo()
    elif args.mode == 'self_test':
      manager.self_test()
    elif args.mode == 'debug':
      manager.self_test()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)


if __name__ == "__main__":
  # Common arguments are defined in base.py
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('checkpoint_path', type=str, help ='')
  parser.add_argument('mode', type=str, help ='')
  parser.add_argument('--config_type', default='small', 
                      type=str, help ='')
  parser.add_argument('--config_path', default='configs/experiments.conf',
                      type=str, help ='')
  parser.add_argument('--debug', default=False,
                      type=common.str2bool, help ='')
  parser.add_argument('--cleanup', default=False,
                      type=common.str2bool, help ='')
  args = parser.parse_args()
  main(args)
