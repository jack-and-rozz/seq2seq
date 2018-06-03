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

BEST_CHECKPOINT_NAME = '/model.ckpt.best'
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
      elif dataset_type == core.dataset.CoNLL2012CorefDataset:
        self.dataset[k] = dataset_type(v.dataset, self.vocab)
        self.vocab.genre = self.dataset[k].genre_vocab
      else:
        raise ValueError('Dataset type %s is undefined.' % t.dataset.dataset_type)

    self.use_coref = 'coref' in self.config.tasks
    self.use_graph = 'graph' in self.config.tasks

  def get_batch(self, batch_type):
    batches = common.recDotDict({'is_training': False})
    if batch_type == 'train':
      batches.is_training = True
      batches.graph = self.dataset.graph.train.get_batch(
        self.config.tasks.graph.batch_size, 
        do_shuffle=True) if self.use_graph else None
      batches.coref = self.dataset.coref.train.get_batch(
        self.config.tasks.coref.batch_size, 
        do_shuffle=True) if self.use_coref else None

    elif batch_type == 'valid':
      batches.graph = self.dataset.graph.valid.get_batch(
        self.config.tasks.graph.batch_size, 
        do_shuffle=False) if self.use_graph else None
      batches.coref = self.dataset.coref.valid.get_batch(
        self.config.tasks.coref.batch_size, 
        do_shuffle=False) if self.use_coref else None

    elif batch_type == 'test':
      batches.graph = self.dataset.graph.test.get_batch(
        self.config.tasks.graph.batch_size, 
        do_shuffle=False) if self.use_graph else None
      batches.coref = self.dataset.coref.test.get_batch(
        self.config.tasks.coref.batch_size, 
        do_shuffle=False) if self.use_coref else None
    return batches

  @common.timewatch()
  def create_model(self, config, mode, checkpoint_path=None):
    mtl_model_type = getattr(mtl_model, config.model_type)
    m = mtl_model_type(self.sess, config, self.vocab) # Define computation graph

    if not checkpoint_path or not os.path.exists(checkpoint_path + '.index'):
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

    vocab_path = self.root_path + '/vocab.word.list'
    with open(vocab_path, 'w') as f:
      f.write('\n'.join(self.vocab.word.rev_vocab) + '\n')
    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    return m

  def debug(self):
    # coref = self.dataset.coref
    # for batch in self.dataset.coref.valid.get_batch(self.config.tasks.coref.batch_size, do_shuffle=False):
    #   pprint(batch)

    batches = self.dataset.graph.valid.get_batch(
      self.config.tasks.graph.batch_size, do_shuffle=False)
    for i, batch in enumerate(batches):
      #pprint(batch)
      print ('#####################################')
      print (common.flatten_batch(batch)[0])
      exit(1)
    exit(1)
    for i, batch in enumerate(batches):
      print ('----------')
      for j ,k in enumerate(batch):
        print ('<%s>' % k)
        pprint(batch[k])
      print(len(batch.text))
      for text, subj, obj in zip(batch.text.raw, batch.text.subj, batch.text.obj):
        print ('##########')
        print (len(text), text)
        pprint (subj)
        pprint (obj)
      exit(1)

  def train(self):
    m = self.create_model(self.config, 'train')
    max_coref_f1 = 0
    max_graph_f1 = 0

    if m.epoch.eval() == 0:
      if self.use_coref:
        self.logger.info("Dataset stats (CoNLL 2012)")
        sizes = (self.dataset.coref.train.size, self.dataset.coref.valid.size, self.dataset.coref.test.size)
        self.logger.info("train, valid, test = (%d, %d, %d)" % sizes)

      if self.use_graph:
        sizes = (self.dataset.graph.train.size, self.dataset.graph.valid.size, self.dataset.graph.test.size)
        self.logger.info("Dataset stats (WikiP2D)")
        self.logger.info("train, valid, test = (%d, %d, %d)" % sizes)

    for epoch in range(m.epoch.eval(), self.config.max_epoch):
      batches = self.get_batch('train')
      epoch_time, step_time, train_loss, summary = m.train(batches, summary_writer=self.summary_writer)
      self.logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, " ".join(["%.3f" % l for l in train_loss])))

      batches = self.get_batch('valid')
      epoch_time, step_time, valid_loss, summary = m.valid(batches)
      self.summary_writer.add_summary(summary, m.global_step.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, loss %s" % (epoch, epoch_time, step_time, " ".join(["%.3f" % l for l in valid_loss])))
      
      save_as_best = False
      if self.use_coref:
        coref_f1 = self.c_test(model=m, use_test_data=False)
        if coref_f1 > max_coref_f1:
          self.logger.info("Epoch %d (valid): coref max f1 update (%.3f->%.3f): " % (m.epoch.eval(), max_coref_f1, coref_f1))
          save_as_best = True
          max_coref_f1 = coref_f1

      if self.use_graph:
        acc, prec, recall = self.g_test(model=m, use_test_data=False)
        graph_f1 = (prec + recall) /2
        if graph_f1 > max_graph_f1:
          self.logger.info("Epoch %d (valid): graph max f1 update (%.3f->%.3f): " % (m.epoch.eval(), max_graph_f1, graph_f1))
          if not self.use_coref:
            save_as_best = True
          max_graph_f1 = graph_f1

      checkpoint_path = self.checkpoints_path + "/model.ckpt"
      if epoch == 0 or (epoch+1) % 1 == 0:
        self.save_model(m, save_as_best=save_as_best)

      m.add_epoch()

  def save_model(self, model, save_as_best=False):
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    self.saver.save(self.sess, checkpoint_path, global_step=model.epoch)
    if save_as_best:
      suffixes = ['data-00000-of-00001', 'index', 'meta']
      for sfx in suffixes:
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (model.epoch.eval(), sfx)
        target_path = self.checkpoints_path + "/%s.%s" % (BEST_CHECKPOINT_NAME, sfx)
        if os.path.exists(source_path):
          cmd = "cp %s %s" % (source_path, target_path)
          os.system(cmd)

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
    #batches = self.get_batch(mode)[m.tasks.coref.dataset]
    batches = self.get_batch(mode)[m.tasks.graph.dataset]
    for i, b in enumerate(batches):
      print(('======== %02d ==========' % i))
      m.tasks.graph.print_batch(b)
    #eval_summary, f1 = m.tasks.coref.test(batches, conll_eval_path[mode])
    exit(1)
    ############################################

  def c_test(self, model=None, use_test_data=True): # mode: 'valid' or 'test'
    m = model
    mode = 'test' if use_test_data else 'valid'
    conll_eval_path = os.path.join(
      self.config.tasks.coref.dataset.source_dir, 
      self.config.tasks.coref.dataset['%s_gold' % mode]
    )
    if not m:
      best_ckpt = os.path.join(self.checkpoint_path, BEST_CHECKPOINT_NAME)
      m = self.create_model(self.config, mode, checkpoint_path=best_ckpt)
    batches = self.get_batch(mode)[m.tasks.coref.dataset]
    eval_summary, f1, results = m.tasks.coref.test(batches, conll_eval_path, mode)
    output_path = self.tests_path + '/c_%s.ep%02d.detailed' % (mode, m.epoch.eval())
    self.summary_writer.add_summary(eval_summary, m.global_step.eval())
    sys.stderr.write('Output the predicted and gold clusters to {}.\n'.format(output_path))
    with open(output_path, 'w') as f:
      sys.stdout = f
      m.tasks.coref.print_results(results)
      sys.stdout = sys.__stdout__

    return f1


  def c_demo(self):
    max_checkpoint_path = os.path.join(self.checkpoints_path, "model.cmax.ckpt")
    ckpt_path = None
    if os.path.exists(max_checkpoint_path + '.index'):
      sys.stderr.write('Found a checkpoint {}.\n'.format(max_checkpoint_path))
      ckpt_path = max_checkpoint_path
    m = self.create_model(self.config, checkpoint_path=ckpt_path)
    eval_data = [d for d in self.get_batch('valid')['coref']]
    run_model(m, eval_data, self.port)

  def g_test(self, model=None, use_test_data=True):
    m = model
    mode = 'test' if use_test_data else 'valid'
    if not m:
      best_ckpt = os.path.join(self.checkpoint_path, BEST_CHECKPOINT_NAME)
      m = self.create_model(self.config, mode, checkpoint_path=best_ckpt)

    batches = self.get_batch(mode)[m.tasks.graph.dataset]
    output_path = self.tests_path + '/g_%s.ep%02d' % (mode, m.epoch.eval())
    (acc, precision, recall), summary = m.tasks.graph.test(
      batches, mode, output_path=output_path)
    self.logger.info("Epoch %d (%s): accuracy, precision, recall, f1 = (%.3f, %.3f, %.3f, %.3f): " % (m.epoch.eval(), mode, acc, precision, recall, (precision+recall)/2)) 
    self.summary_writer.add_summary(summary, m.global_step.eval())
    return acc, precision, recall
  
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
        'p_triples': [p_triples], #self.dataset.graph.get_all_triples(),
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
  parser.add_argument('-ct','--config_type', default='small', 
                      type=str, help ='')
  parser.add_argument('-cp','--config_path', default='configs/experiments.conf',
                      type=str, help ='')
  parser.add_argument('-d', '--debug', default=False,
                      type=common.str2bool, help ='')
  parser.add_argument('--cleanup', default=False,
                      type=common.str2bool, help ='')
  args = parser.parse_args()
  main(args)
