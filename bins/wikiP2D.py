#coding: utf-8
import sys, os, random, copy, socket, time, re, argparse
from collections import OrderedDict
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

from core.models.wikiP2D.category.evaluation import print_example

BEST_CHECKPOINT_NAME = 'model.ckpt.best'

class ExperimentManager(ManagerBase):
  @common.timewatch()
  def __init__(self, args, sess):
    super().__init__(args, sess)
    self.config = config = self.load_config(args)
    self.port = args.port
    self.vocab = common.dotDict()
    self.model = None
    # Load pretrained embeddings.
    t = time.time()
    self.vocab.word = VocabularyWithEmbedding(
      config.embeddings_conf, config.vocab_size.word,
      lowercase=config.lowercase,
      normalize_digits=config.normalize_digits,
      normalize_embedding=True
    )
    self.vocab.char = PredefinedCharVocab(
      config.char_vocab_path, config.vocab_size.char,
      lowercase=False,
    )

    # Load Dataset.
    self.dataset = common.recDotDict()
    for k, v in config.tasks.items():
      if 'dataset' in v: # for tasks without data
        dataset_type = getattr(core.dataset, v.dataset.dataset_type)
      else:
        continue

      if dataset_type == core.dataset.CoNLL2012CorefDataset:
        self.dataset[k] = dataset_type(v.dataset, self.vocab)
        self.vocab.genre = self.dataset[k].genre_vocab
      else:
        self.dataset[k] = dataset_type(v.dataset, self.vocab)

    self.use_coref = 'coref' in self.config.tasks
    self.use_graph = 'graph' in self.config.tasks
    print(time.time()-t)
    t = time.time()

  def get_batch(self, batch_type):
    batches = common.recDotDict({'is_training': False})
    do_shuffle = False

    if batch_type == 'train':
      batches.is_training = True
      do_shuffle = True

    for task_name in self.config.tasks:
      #batches[task_name] = self.dataset[task_name][batch_type].get_batch(
      if not task_name in self.dataset:
        continue
      batches[task_name] = getattr(self.dataset[task_name], batch_type).get_batch(
        self.config.tasks[task_name].batch_size,
        do_shuffle=do_shuffle) 
    return batches

  @common.timewatch()
  def create_model(self, config, checkpoint_path=None):
    mtl_model_type = getattr(mtl_model, config.model_type)
    if not self.model:
      self.model = m = mtl_model_type(self.sess, config, self.vocab) # Define computation graph

    if not checkpoint_path or not os.path.exists(checkpoint_path + '.index'):
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.max_to_keep)
    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stdout.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      sys.stdout.write("Created model with fresh parameters.\n")
      tf.global_variables_initializer().run()

    # Store variable names and vocabulary for debug.
    variables_path = self.root_path + '/variables.list'
    if not os.path.exists(variables_path):
      with open(variables_path, 'w') as f:
        variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
        variable_names = [name for name in variable_names if not re.search('Adam', name)]
        f.write('\n'.join(variable_names) + '\n')
    vocab_path = self.root_path + '/vocab.word.list'
    if not os.path.exists(vocab_path):
      with open(vocab_path, 'w') as f:
        f.write('\n'.join(self.vocab.word.rev_vocab) + '\n')
    return self.model

  def debug(self):
    # coref = self.dataset.coref
    # for batch in self.dataset.coref.valid.get_batch(self.config.tasks.coref.batch_size, do_shuffle=False):
    #   pprint(batch)
    task_name = 'category'
    batches = self.dataset[task_name].train.get_batch(
      self.config.tasks[task_name].batch_size, do_shuffle=False)
    rels = []
    for i, batch in enumerate(batches):
      #print ('#####################################')
      #print(batch.category.raw, batch.category.label)
      for j, b in enumerate(common.flatten_batch(batch)):
        print('<%03d-%03d>' % (i,j))
        print_example(b, self.vocab)
        #exit(1)
        print('')
      

    common.dbgprint(self.dataset[task_name].valid.size)
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
    model = self.create_model(self.config)

    if isinstance(model, mtl_model.OneByOne):
      self.train_one_by_one(model)
    else:
      self.train_simultaneously(model)

    # Do final validation and testing with the best model.
    self.test()

    m = self.model
    self.logger.info("The model in epoch %d performs best." % m.epoch.eval())

  def train_simultaneously(self, model):
    m = model
    for epoch in range(m.epoch.eval(), self.config.max_epoch):
      batches = self.get_batch('train')
      self.logger.info("Epoch %d: Start" % epoch)
      epoch_time, train_loss, summary = m.train(batches)
      self.logger.info("Epoch %d (train): epoch-time %.2f, loss %s" % (epoch, epoch_time, " ".join(["%.3f" % l for l in train_loss])))

      batches = self.get_batch('valid')
      epoch_time, valid_loss, summary = m.valid(batches)
      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, loss %s" % (epoch, epoch_time, " ".join(["%.3f" % l for l in valid_loss])))

      save_as_best = self.test(m)
      self.save_model(m, save_as_best=save_as_best)
      m.add_epoch()

  def train_one_by_one(self, model):
    '''
    '''
    m = model
    def _run_task(m, task):
      epoch = m.epoch.eval()
      batches = self.get_batch('train')
      self.logger.info("Epoch %d: Start" % m.epoch.eval())
      epoch_time, train_loss, summary = m.train(task, batches)
      self.logger.info("Epoch %d (train): epoch-time %.2f, loss %.3f" % (epoch, epoch_time, train_loss))

      batches = self.get_batch('valid')
      epoch_time, valid_loss, summary = m.valid(task, batches)
      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, loss %.3f" % (epoch, epoch_time, valid_loss))
      m.add_epoch()

    # Train the model in a reverse order of important tasks.
    task = m.tasks.keys()[1]
    for i in range(m.epoch.eval(), int(self.config.max_epoch/2)):
      _run_task(m, task)
      save_as_best = self.test(model=m, target_tasks=[task])
      self.save_model(m, save_as_best=save_as_best)

    # Load a model with the best score of WikiP2D task. 
    best_ckpt = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
    m = self.create_model(self.config, checkpoint_path=best_ckpt)

    task = m.tasks.keys()[0]
    for epoch in range(m.epoch.eval(), self.config.max_epoch):
      _run_task(m, task)
      save_as_best = self.test(model=m, target_tasks=[task])
      self.save_model(m, save_as_best=save_as_best)

  def test(self, model=None, target_tasks=None):
    '''
    Args:
    - model:
    - target_tasks:
    '''
    m = model

    if m: # in training
      tasks = OrderedDict(
        [(k, v) for k, v in m.tasks.items() 
         if (not target_tasks or k in target_tasks) and k != 'adversarial'])
      epoch = m.epoch.eval()
      save_as_best = [False for t in tasks]
      for i, (task_name, task_model) in enumerate(tasks.items()):
        mode = 'valid'
        batches = self.get_batch(mode)[task_name]
        output_path = self.tests_path + '/%s_%s.%02d' % (task_name, mode, epoch)
        valid_score, valid_summary = task_model.test(batches, mode, 
                                                     self.logger, output_path)
        self.summary_writer.add_summary(valid_summary, m.epoch.eval())

        if valid_score > task_model.max_score.eval():
          save_as_best[i] = True
          self.logger.info("Epoch %d (valid): %s max score update (%.3f->%.3f): " % (m.epoch.eval(), task_name, task_model.max_score.eval(), valid_score))
          task_model.update_max_score(valid_score)

        mode = 'test'
        batches = self.get_batch(mode)[task_name]
        output_path = self.tests_path + '/%s_%s.%02d' % (task_name, mode, epoch)
        test_score, test_summary = task_model.test(batches, mode, 
                                                   self.logger, output_path)
        self.summary_writer.add_summary(test_summary, m.epoch.eval())

      # Currently update the best model by the score of the first task.
      return save_as_best[0] 

    else:
      mode = 'test'
      best_ckpt = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
      m = self.create_model(self.config, checkpoint_path=best_ckpt)
      tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if not target_tasks or k in target_tasks])
      for i, (task_name, task_model) in enumerate(tasks.items()):
        batches = self.get_batch(mode)[task_name]
        output_path = self.tests_path + '/%s_%s.best' % (task_name, mode)
        test_score, _ = task_model.test(batches, mode, self.logger, output_path)
      return False

  # def g_test_as_valid(self, m):
  #   save_as_best = False
  #   # Do validation.
  #   valid_graph_f1 = self.g_test(model=m, use_test_data=False, 
  #                                add_summary=True)
  #   valid_max_graph_f1 = m.tasks.graph.max_score.eval()
  #   if valid_graph_f1 > valid_max_graph_f1:
  #     self.logger.info("Epoch %d (valid): graph max f1 update (%.3f->%.3f): " % (m.epoch.eval(), valid_max_graph_f1, valid_graph_f1))
  #     m.tasks.graph.update_max_score(valid_graph_f1)
  #     save_as_best = True

  #   # Do testing. (only for analysis)
  #   test_graph_f1 = self.g_test(model=m, use_test_data=True, 
  #                               add_summary=True)
  #   return save_as_best

  # def c_test_as_valid(self, m):
  #   save_as_best = False
  #   # Do validation.
  #   valid_coref_f1 = self.c_test(model=m, use_test_data=False, add_summary=True)
  #   valid_coref_f1 = sum(valid_coref_f1) / len(valid_coref_f1)
  #   valid_max_coref_f1 = m.tasks.coref.max_score.eval()
  #   if valid_coref_f1 > valid_max_coref_f1:
  #     self.logger.info("Epoch %d (valid): coref max f1 update (%.3f->%.3f): " % (m.epoch.eval(), valid_max_coref_f1, valid_coref_f1))
  #     save_as_best = True
  #     m.tasks.coref.update_max_score(valid_coref_f1)

  #   # Do testing. (only for analysis)
  #   test_coref_f1 = self.c_test(model=m, use_test_data=True, add_summary=True)
  #   return save_as_best

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


  # def c_test(self, model=None, use_test_data=True, add_summary=False): 
  #   if 'coref' not in self.config.tasks:
  #     sys.stderr.write('No coreference model in config.\n')
  #     return None
  #   m = model
  #   mode = 'test' if use_test_data else 'valid'
  #   conll_eval_path = os.path.join(
  #     self.config.tasks.coref.dataset.source_dir, 
  #     self.config.tasks.coref.dataset['%s_gold' % mode]
  #   )
  #   if not m:
  #     best_ckpt = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
  #     m = self.create_model(self.config, mode, checkpoint_path=best_ckpt)
  #   else:
  #     best_ckpt = None

  #   batches = self.get_batch(mode)['coref']

  #   if best_ckpt and os.path.exists(best_ckpt + '.index'):
  #     output_path = self.tests_path + '/c_%s.best' % (mode)
  #   else:
  #     output_path = self.tests_path + '/c_%s.ep%02d' % (mode, m.epoch.eval())

  #   with open(output_path + '.stat', 'w') as f:
  #     sys.stdout = f
  #     eval_summary, f1, results = m.tasks.coref.test(
  #       batches, conll_eval_path, mode)
  #     sys.stdout = sys.__stdout__

  #   muc_f1, bcub_f1, ceaf_f1 = f1
  #   self.logger.info("Epoch %d (%s): MUC, B^3, Ceaf, ave_f1 = (%.3f, %.3f, %.3f, %.3f): " % (m.epoch.eval(), mode, muc_f1, bcub_f1, ceaf_f1, sum(f1)/len(f1)))

  #   if add_summary:
  #     self.summary_writer.add_summary(eval_summary, m.epoch.eval())
  #   with open(output_path + '.detail', 'w') as f:
  #     sys.stdout = f
  #     m.tasks.coref.print_results(results)
  #     sys.stdout = sys.__stdout__

  #   sys.stderr.write("Output the predicted and gold clusters to \'{}\' .\n".format(output_path))
  #   return f1


  def c_demo(self):
    port = self.port
    max_checkpoint_path = os.path.join(self.checkpoints_path, "model.cmax.ckpt")
    ckpt_path = None
    if os.path.exists(max_checkpoint_path + '.index'):
      sys.stderr.write('Found a checkpoint {}.\n'.format(max_checkpoint_path))
      ckpt_path = max_checkpoint_path
    m = self.create_model(self.config, checkpoint_path=ckpt_path)
    eval_data = [d for d in self.get_batch('valid')['coref']]
    run_model(m, eval_data, port)

  # def g_test(self, model=None, use_test_data=True, add_summary=False):
  #   if 'graph' not in self.config.tasks:
  #     sys.stderr.write('No wikiP2D model in config.\n')
  #     return 

  #   m = model
  #   mode = 'test' if use_test_data else 'valid'

  #   best_ckpt = None
  #   if not m:
  #     best_ckpt = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME)
  #     m = self.create_model(self.config, mode, checkpoint_path=best_ckpt)

  #   batches = self.get_batch(mode)['graph']
  #   if best_ckpt and os.path.exists(best_ckpt + '.index'):
  #     output_path = self.tests_path + '/g_%s.best' % (mode)
  #   else:
  #     output_path = self.tests_path + '/g_%s.ep%02d' % (mode, m.epoch.eval())
  #   (acc, precision, recall), summary = m.tasks.graph.test(
  #     batches, mode, output_path=output_path)
  #   self.logger.info("Epoch %d (%s): accuracy, precision, recall, f1 = (%.3f, %.3f, %.3f, %.3f): " % (m.epoch.eval(), mode, acc, precision, recall, (precision+recall)/2)) 
  #   if add_summary:
  #     self.summary_writer.add_summary(summary, m.epoch.eval())
  #   return (precision + recall) / 2
  
  def g_demo(self):
    m = self.create_model(self.self.config)

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


class IterativeTrainer(ExperimentManager):
  pass


def main(args):
  # if args.log_error:
  #   sys.stderr = open(os.path.join(args.model_root_path, '%s.err'  % args.mode), 'w')
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.set_random_seed(0)
    manager = ExperimentManager(args, sess)
    getattr(manager, args.mode)()

if __name__ == "__main__":
  # Common arguments are defined in base.py
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root_path', type=str, help ='')
  parser.add_argument('mode', type=str, help ='')
  parser.add_argument('--port', default=8080, type=int, help='for demo.')
  parser.add_argument('-ct','--config_type', default='main', 
                      type=str, help ='')
  parser.add_argument('-cp','--config_path', default='configs/experiments.conf',
                      type=str, help ='')
  parser.add_argument('-d', '--debug', default=False,
                      type=common.str2bool, help ='')
  parser.add_argument('--cleanup', default=False,
                      type=common.str2bool, help ='')
  #parser.add_argument('-le', '--log_error', default=False,
  #                    type=common.str2bool, help ='')
  args = parser.parse_args()
  main(args)
