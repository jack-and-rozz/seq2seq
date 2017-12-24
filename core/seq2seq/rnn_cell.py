#from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import * # tf == 1.1
from tensorflow.python.ops.rnn_cell_impl import * # tf >= 1.2
import core.models.wikiP2D.coref.util as coref_util



class SharingWrapper(RNNCell):
  def __init__(self, cell):
    self._cell = cell
    self.my_scope = None

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, _scope=None):
    # _scope is not used (Hold the scope where it was called first.)
    if self.my_scope == None:
      self.my_scope = tf.get_variable_scope() 
    else:
      self.my_scope.reuse_variables()
    return self._cell(inputs, state, self.my_scope)
    # print '------------'
    # print self
    # print self.scope
    # print self.reuse
    # if self.scope == None:
    #   self.scope = tf.get_variable_scope() 

    # if self.reuse:
    #   self.scope.reuse_variables()
    # else:
    #   self.reuse = True
    # return self._cell(inputs, state, self.scope)

# (from e2e-coref)
class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, dropout, reuse=None, scope=None):
    self._num_units = num_units
    self._dropout = dropout
    self._reuse = reuse

    with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
      self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
      initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
      initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
      self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  #def preprocess_input(self, inputs):
  #  return coref_util.projection(inputs, 3 * self.output_size)

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__, reuse=self._reuse): # "CustomLSTMCell"
      with tf.variable_scope('preprocess'):
        inputs = coref_util.projection(inputs, 3 * self.output_size)

      c, h = state
      #h *= self._dropout_mask
      h = tf.nn.dropout(h, self._dropout)
      with tf.variable_scope('projection'):
        projected_h = coref_util.projection(h, 3 * self.output_size, 
                                            initializer=self._initializer)
      concat = inputs + projected_h
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer
