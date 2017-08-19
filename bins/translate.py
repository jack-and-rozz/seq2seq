#coding: utf-8
import sys, os, random, math
import tensorflow as tf
from base import BaseManager, logger
import core.models.translate as translate_models
from core.utils import common
import core.models.wrappers as wrappers

# about dataset
tf.app.flags.DEFINE_string("source_data_dir", "dataset/translate/ASPEC-JE/source", "")
tf.app.flags.DEFINE_string("processed_data_dir", "dataset/translate/ASPEC-JE/processed", "")
tf.app.flags.DEFINE_string("model_type", "Baseline", "")
tf.app.flags.DEFINE_string("source_lang", "en", "")
tf.app.flags.DEFINE_string("target_lang", "ja", "")
tf.app.flags.DEFINE_integer("beam_size", 1, "")

tf.app.flags.DEFINE_string("vocab_data", "train", "")
tf.app.flags.DEFINE_string("train_data", "train", "")
tf.app.flags.DEFINE_string("dev_data", "dev", "")
tf.app.flags.DEFINE_string("test_data", "test", "")
tf.app.flags.DEFINE_integer("max_sequence_length", 64, "")
tf.app.flags.DEFINE_integer("max_train_rows", 2000000, 
                            "Maximum number of rows to be used as train data.")

# about hyperparameters
tf.app.flags.DEFINE_integer("source_vocab_size", 30000, "Vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 30000, "Vocabulary size.")
tf.app.flags.DEFINE_float("keep_prob", 0.5,
                          "the keeping probability of active neurons in dropout")
tf.app.flags.DEFINE_integer("num_samples", 512, "")
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Size of each token embedding.")
tf.app.flags.DEFINE_integer("max_epoch", 50, "")
tf.app.flags.DEFINE_float("init_scale", 0.1, "")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")

# for RNN LM
tf.app.flags.DEFINE_string("cell_type", "GRUCell", "Cell type")
tf.app.flags.DEFINE_string("seq2seq_type", "BasicSeq2Seq", "Cell type")
tf.app.flags.DEFINE_string("encoder_type", "RNNEncoder", "")
tf.app.flags.DEFINE_string("decoder_type", "RNNDecoder", "")
tf.app.flags.DEFINE_boolean("use_sequence_length", True, "If True, PAD_ID tokens are not input to RNN. (This option shouldn't be used when reversing encoder's inputs.)")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("trainable_source_embedding", True, "")
tf.app.flags.DEFINE_boolean("trainable_target_embedding", True, "")
tf.app.flags.DEFINE_boolean("share_embedding", False, "If true, a decoder uses encoder's embedding (for dialogue)")

from core.utils.vocabulary import Vocabulary
from core.utils.dataset import ASPECDataset, EOS_ID

class TranslateManager(BaseManager):
  def __init__(self, FLAGS, sess):
    super(TranslateManager, self).__init__(FLAGS, sess)
    self.TMP_FLAGS += ['beam_size']
    self.s_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, 
                              FLAGS.vocab_data, FLAGS.source_lang, 
                              FLAGS.source_vocab_size)
    self.t_vocab = Vocabulary(FLAGS.source_data_dir, FLAGS.processed_data_dir, 
                              FLAGS.vocab_data, FLAGS.target_lang, 
                              FLAGS.target_vocab_size)
    self.model_type = getattr(translate_models, FLAGS.model_type)

  @common.timewatch(logger)
  def create_model(self, FLAGS, forward_only, do_update, reuse=None):
    sess = self.sess

    with tf.variable_scope("Model", reuse=reuse):
      if do_update and len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        m = getattr(wrappers, "AverageGradientMultiGPUTrainWrapper")(sess, FLAGS, self.model_type)
        #m = getattr(wrappers, "AverageLossMultiGPUTrainWrapper")(sess, FLAGS, self.model_type)
      else:
        m = self.model_type(sess, FLAGS, forward_only, do_update)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path + '/checkpoints')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
  
    if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.index'):
      if reuse==None:
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      if reuse==None:
        logger.info("Created model with fresh parameters.")
      tf.global_variables_initializer().run()
      with open(FLAGS.checkpoint_path + '/variables/variables.list', 'w') as f:
        f.write('\n'.join([v.name for v in tf.global_variables()]) + '\n')
    return m

  def decode_interact(self):
    FLAGS = self.FLAGS
    mtest = self.create_model(FLAGS, True, False)
    while True:
      sys.stdout.write("> ",)
      source = sys.stdin.readline()
      source = source.split()
      raw_batch = [(None, self.s_vocab.to_ids(source), [])]
      _, outputs = mtest.decode(raw_batch)
      output = outputs[0]
      if EOS_ID in output:
        output = output[:output.index(EOS_ID)]
      output = " " .join(t_vocab.to_tokens(output))
      print (output)

  def decode_test(self):
    FLAGS = self.FLAGS
    sess = self.sess

    def get_gold_results(data):
      # read original test corpus (unknown words still remain)
      sources = [data.tokenizer(l) for l in open(data.s_source_path)]
      targets = [data.tokenizer(l) for l in open(data.t_source_path)]

      s_ref_path = '%s/%s.%s' % (self.TESTS_PATH, FLAGS.test_data, 
                                 FLAGS.source_lang)
      t_ref_path = '%s/%s.%s' % (self.TESTS_PATH, FLAGS.test_data, 
                                 FLAGS.target_lang)

      if not os.path.exists(s_ref_path):
        with open(s_ref_path, 'w') as f:
          for s in sources:
            f.write(" ".join([str(x) for x in s]) + '\n')
      if not os.path.exists(t_ref_path):
        with open(t_ref_path, 'w') as f:
          for t in targets:
            f.write(" ".join([str(x) for x in t]) + '\n')

      return sources, targets

    def get_decode_results(data):
      mtest = self.create_model(FLAGS, True, False)
      decode_path = '%s/%s.%s.decode.beam%d.ep%d' % (
        self.TESTS_PATH,
        FLAGS.test_data, 
        FLAGS.target_lang, 
        FLAGS.beam_size, 
        mtest.epoch.eval()
      )
      if not os.path.exists(decode_path):
        results = []
        for i, raw_batch in enumerate(data.get_batch(FLAGS.batch_size)):
          _, outputs = mtest.decode(raw_batch)
          for b, o in zip(raw_batch, outputs):
            idx, s, t = b
            if EOS_ID in o:
              o = o[:o.index(EOS_ID)]
            source = self.s_vocab.to_tokens(s)
            target = self.t_vocab.to_tokens(t)
            result = self.t_vocab.to_tokens(o)
            results.append(result)
            print "<%d>" % idx
            print (' '.join(source))
            print (' '.join(target))
            print (' '.join(result))
        with open(decode_path, 'w') as f:
          f.write("\n".join([' '.join(x) for x in results]) + "\n")
      else:
        results = [l.replace('\n', '').split() for l in open(decode_path)]
      return results

    data_path = os.path.join(FLAGS.source_data_dir, FLAGS.vocab_data)
    test = ASPECDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir, 
      FLAGS.test_data, self.s_vocab, self.t_vocab,
      max_sequence_length=None, max_rows=None)
    FLAGS.max_sequence_length = test.max_sequence_length
    sources, targets = get_gold_results(test)
    results = get_decode_results(test)
    logger.info("Number of tests: %d " % test.size)

    #bleu_score = nltk.translate.bleu_score.corpus_bleu(targets, results)
    #logger.info("BLEU Score: %f " % bleu_score)

  def train(self):
    FLAGS = self.FLAGS
    sess = self.sess

    logger.info("Reading dataset.")
    train = ASPECDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir, 
      FLAGS.train_data, self.s_vocab, self.t_vocab, 
      max_sequence_length=FLAGS.max_sequence_length,
      max_rows=FLAGS.max_train_rows)
    dev = ASPECDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir, 
      FLAGS.dev_data, self.s_vocab, self.t_vocab)
    test = ASPECDataset(
      FLAGS.source_data_dir, FLAGS.processed_data_dir, 
      FLAGS.test_data, self.s_vocab, self.t_vocab)
    logger.info("(train dev test) = (%d %d %d)" % (train.size, dev.size, test.size))

    with tf.name_scope('train'):
      mtrain = self.create_model(FLAGS, False, True)
      summary_writer = tf.summary.FileWriter(self.SUMMARIES_PATH, sess.graph) 

    with tf.name_scope('dev'):
      mvalid = self.create_model(FLAGS, False, False, 
                                 reuse=True)
    for epoch in xrange(mtrain.epoch.eval(), FLAGS.max_epoch):
      logger.info("Epoch %d: Start training." % epoch)
      epoch_time, step_time, train_loss = mtrain.run_batch(
        train, FLAGS.batch_size, do_shuffle=True)
      train_ppx = math.exp(train_loss)
      logger.info("Epoch %d (train): epoch-time %.2f, step-time %.2f, ppx %.4f" % (epoch, epoch_time, step_time, train_ppx))

      epoch_time, step_time, valid_loss = mvalid.run_batch(dev, FLAGS.batch_size)
      valid_ppx = math.exp(valid_loss)
      logger.info("Epoch %d (valid): epoch-time %.2f, step-time %.2f, ppx %.4f" % (epoch, epoch_time, step_time, valid_ppx))

      mtrain.add_epoch()
      checkpoint_path = self.CHECKPOINTS_PATH + "/model.ckpt"
      mtrain.saver.save(sess, checkpoint_path, global_step=mtrain.epoch)



def main(_):
  tf_config = tf.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )
  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    # 乱数シードの設定はgraphの生成後、opsの生成前 (http://qiita.com/yuyakato/items/9a5d80e6c7c41e9a9d22)
    tf.set_random_seed(0)
    FLAGS = tf.app.flags.FLAGS
    manager = TranslateManager(FLAGS, sess)

    manager.create_dir()

    if FLAGS.mode == "train":
      manager.save_config()
      manager.train()
    elif FLAGS.mode == "decode":
      manager.decode_test()
    elif FLAGS.mode == "decode_interact":
      manager.decode_interact()
    else:
      sys.stderr.write("Unknown mode.\n")
      exit(1)

if __name__ == "__main__":
  tf.app.run()
