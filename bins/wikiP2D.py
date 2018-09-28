#coding: utf-8
import sys, os, random, copy, socket, time, re, argparse
sys.path.append(os.getcwd())
from collections import OrderedDict
from pprint import pprint
import tensorflow as tf
import numpy as np
from bins.base import ManagerBase
from core.utils import common, tf_utils
import core.dataset 
import core.models.wikiP2D.mtl as mtl_model
from core.models.wikiP2D.coref.demo import run_model
from core.vocabulary.base import VocabularyWithEmbedding, PredefinedCharVocab

BEST_CHECKPOINT_NAME = 'model.ckpt.best'
class ExperimentManager(ManagerBase):
  @common.timewatch()
  def __init__(self, args, sess):
    super().__init__(args, sess)
    self.config = config = self.load_config(args)
    self.vocab = common.recDotDefaultDict()
    self.model = None

    # Load pretrained embeddings.
    self.vocab.encoder.word = VocabularyWithEmbedding(config.vocab.encoder.word)
    self.vocab.encoder.char = PredefinedCharVocab(config.vocab.encoder.char)
    self.vocab.decoder.word = VocabularyWithEmbedding(config.vocab.decoder.word)

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

  def get_batch(self, batch_type):
    batches = common.recDotDict({'is_training': False})
    do_shuffle = False

    if batch_type == 'train':
      batches.is_training = True
      do_shuffle = True

    for task_name in self.config.tasks:
      if not task_name in self.dataset:
        continue
      batches[task_name] = getattr(self.dataset[task_name], batch_type).get_batch(
        self.config.tasks[task_name].batch_size,
        do_shuffle=do_shuffle) 
    return batches

  @common.timewatch()
  def create_model(self, config, load_best=False):
    mtl_model_type = getattr(mtl_model, config.model_type)
    checkpoint_path = os.path.join(self.checkpoints_path, BEST_CHECKPOINT_NAME) if load_best else None

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
    task_name = [k for k in self.config.tasks][0]
    batches = self.dataset[task_name].train.get_batch(
      self.config.tasks[task_name].batch_size, do_shuffle=True)
    print(self.vocab.rel.rev_names)
    exit(1)
    rels = []
    for i, batch in enumerate(batches):
      for j, b in enumerate(common.flatten_batch(batch)):
        print('<%03d-%03d>' % (i,j))
        self.dataset[task_name].print_example(b)
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
    if model.epoch.eval() == 0:
      self.logger.info('<Dataset size>')
      for task_name, d in self.dataset.items():
        train_size, dev_size, test_size = d.size
        self.logger.info('%s: %d, %d, %d' % (task_name, train_size, dev_size, test_size))
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
      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (train): epoch-time %.2f, loss %s" % (epoch, epoch_time, " ".join(["%.3f" % l for l in train_loss])))

      batches = self.get_batch('valid')
      epoch_time, valid_loss, summary = m.valid(batches)
      self.summary_writer.add_summary(summary, m.epoch.eval())
      self.logger.info("Epoch %d (valid): epoch-time %.2f, loss %s" % (epoch, epoch_time, " ".join(["%.3f" % l for l in valid_loss])))

      save_as_best = self.test_for_valid(m)
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
      save_as_best = self.test_for_valid(model=m, target_tasks=[task])
      self.save_model(m, save_as_best=save_as_best)

    # Load a model with the best score of WikiP2D task. 
    m = self.create_model(self.config, load_best=True)

    task = m.tasks.keys()[0]
    for epoch in range(m.epoch.eval(), self.config.max_epoch):
      _run_task(m, task)
      save_as_best = self.test_for_valid(model=m, target_tasks=[task])
      self.save_model(m, save_as_best=save_as_best)

  @common.timewatch()
  def test_for_valid(self, model, target_tasks=None):
    '''
    This is a function to show the performance of the model in each epoch.
    If you'd like to run testing again in a different setting from terminal, execute test().

    <Args>
    - model:
    - target_tasks:
    <Return>
    - A boolean, which shows whether the score of the first task in this epoch becomes higher or not.
    '''
    m = model

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
      self.logger.info("Epoch %d (valid): %s score = (%.3f): " % (m.epoch.eval(), task_name, valid_score))

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

  @common.timewatch()
  def test(self):
    mode = 'test'
    target_tasks = []
    
    m = self.create_model(self.config, load_best=True)
    tasks = OrderedDict([(k, v) for k, v in m.tasks.items() if not target_tasks or k in target_tasks])
    for i, (task_name, task_model) in enumerate(tasks.items()):
      batches = self.get_batch(mode)[task_name]
      output_path = self.tests_path + '/%s_%s.best' % (task_name, mode)
      test_score, _ = task_model.test(batches, mode, self.logger, output_path)
      self.logger.info("Epoch %d (test): %s score = (%.3f): " % (m.epoch.eval(), task_name, test_score))

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


  # def c_demo(self):
  #   port = self.port
  #   max_checkpoint_path = os.path.join(self.checkpoints_path, "model.cmax.ckpt")
  #   ckpt_path = None
  #   if os.path.exists(max_checkpoint_path + '.index'):
  #     sys.stderr.write('Found a checkpoint {}.\n'.format(max_checkpoint_path))
  #     ckpt_path = max_checkpoint_path
  #   m = self.create_model(self.config, checkpoint_path=ckpt_path)
  #   eval_data = [d for d in self.get_batch('valid')['coref']]
  #   run_model(m, eval_data, port)



def main(args):
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    random.seed(0)
    tf.set_random_seed(0)
    manager = ExperimentManager(args, sess)
    getattr(manager, args.mode)()

if __name__ == "__main__":
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
  # parser.add_argument('-t', '--target_tasks', default=[],
  #                     type=common.str2arr, help ='The list of tasks to be tested.')
  #parser.add_argument('-le', '--log_error', default=False,
  #                    type=common.str2bool, help ='')
  args = parser.parse_args()
  main(args)
