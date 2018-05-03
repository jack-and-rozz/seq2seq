import tensorflow as tf
import os, time

class AverageGradientMultiGPUTrainWrapper(object):
  def __init__(self, sess, FLAGS, model_type):
    self.sess = sess
    self.model_type = model_type
    self.learning_rate = FLAGS.learning_rate
    self.max_gradient_norm = FLAGS.max_gradient_norm
    self.models = self.setup_models(sess, FLAGS)
    self.num_gpus = len(self.models)
    self.global_step = self.models[0].global_step
    self.epoch = self.models[0].epoch
    self.add_epoch = self.models[0].add_epoch
    with tf.name_scope('average_gradient'):
      #with tf.device('/cpu:0'):
      self.losses = tf.add_n([m.losses for m in self.models]) / self.num_gpus
      self.grad_and_vars = self.average_gradients([m.grad_and_vars for m in self.models])


      opt = tf.train.AdamOptimizer(self.learning_rate)
      self.updates = opt.apply_gradients(self.grad_and_vars, global_step=self.global_step)
    self.saver = tf.train.Saver(tf.global_variables())

  def setup_models(self, sess, FLAGS):
    if os.environ['CUDA_VISIBLE_DEVICES'] != '-1':
      num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
      num_gpus = 0
      raise ValueError("Set \'CUDA_VISIBLE_DEVICES\' to define which gpus to be used.")
    models = []
    for i in range(num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('model_%d' % (i)) as scope:
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          m = self.model_type(sess, FLAGS, False, True)
          models.append(m)
    return models

  def average_gradients(self, tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def get_input_feed(self, raw_batch):
    input_feed = {}
    for b, m in zip(raw_batch, self.models):
      input_feed.update(m.get_input_feed(b))
    return input_feed

  def step(self, raw_batch):
    sess = self.sess
    input_feed = self.get_input_feed(raw_batch)
    output_feed = [self.losses, self.updates]
    outputs = sess.run(output_feed, input_feed)
    return outputs[0]

  def run_batch(self, data, batch_size, do_shuffle=False):
    start_time = time.time()
    loss = 0.0
    for i, raw_batch in enumerate(data.get_batch(batch_size, do_shuffle=do_shuffle,n_batches=self.num_gpus)):
      step_loss = self.step(raw_batch)
      loss += step_loss 
    epoch_time = (time.time() - start_time)
    step_time = epoch_time / (i+1)
    loss = loss / (i+1)
    return epoch_time, step_time, loss 




class AverageLossMultiGPUTrainWrapper(AverageGradientMultiGPUTrainWrapper):
  def __init__(self, sess, FLAGS, model_type):
    self.sess = sess
    self.model_type = model_type
    self.learning_rate = FLAGS.learning_rate
    self.max_gradient_norm = FLAGS.max_gradient_norm
    self.models = self.setup_models(sess, FLAGS)
    self.num_gpus = len(self.models)
    self.global_step = self.models[0].global_step
    self.epoch = self.models[0].epoch
    self.add_epoch = self.models[0].add_epoch

    with tf.name_scope('average_loss'):
      with tf.device('/cpu:0'):

        self.losses = tf.add_n([m.losses for m in self.models]) / self.num_gpus
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = [grad for grad, _ in opt.compute_gradients(self.losses)]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
        params = tf.trainable_variables()

        self.grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]

        self.updates = opt.apply_gradients(self.grad_and_vars, global_step=self.global_step)
    self.saver = tf.train.Saver(tf.global_variables())
