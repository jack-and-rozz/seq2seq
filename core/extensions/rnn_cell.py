#coding: utf-8
# from __future__ import absolute_import
# from __future__ import division

# import collections
# import contextlib
# import hashlib
# import math
# import numbers

# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.framework import tensor_util
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import clip_ops
# from tensorflow.python.ops import embedding_ops
# from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import partitioned_variables
# from tensorflow.python.ops import random_ops
# from tensorflow.python.ops import variable_scope as vs

# from tensorflow.python.ops.math_ops import sigmoid
# from tensorflow.python.ops.math_ops import tanh
# from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

# from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import rnn_cell

# Shared rnn_cell classes.

#class SharedGRUCell(core_rnn_cell.GRUCell):


class GRUCell(core_rnn_cell.GRUCell):
  def __init__(self, num_units, input_size=None, activation=tf.nn.tanh,
               reuse=None):
    core_rnn_cell.GRUCell.__init__(self, num_units, input_size, activation)
    self.my_scope = None

  def __call__(self, inputs, state, _scope=None):
    # _scope is not used (Hold the scope that was called in the beginning)
    if self.my_scope == None:
      self.my_scope = tf.get_variable_scope()
    else:
      self.my_scope.reuse_variables()
    return core_rnn_cell.GRUCell.__call__(self, inputs, state, self.my_scope)


# class CustomLSTMCell(CustomLSTMCell):
#   def __init__(self, num_units, input_size=None, activation=tf.nn.tanh,
#                reuse=None):
#     rnn_cell.CustomLSTMCell.__init__(self, num_units, input_size, activation)
#     self.my_scope = None

#   def __call__(self, inputs, state, _scope=None):
#     # _scope is not used (Hold the scope that was called in the beginning)
#     if self.my_scope == None:
#       self.my_scope = tf.get_variable_scope()
#     else:
#       self.my_scope.reuse_variables()
#     return rnn_cell.CustomLSTMCell.__call__(self, inputs, state, self.my_scope)
