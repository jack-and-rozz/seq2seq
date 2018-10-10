# coding: utf-8
import copy
import tensorflow as tf
from . import rnn_cell

def setup_cell(cell_type, hidden_size, num_layers=1, keep_prob=None):
  cell_module = rnn_cell
  cell_class = getattr(cell_module, cell_type)

  def _get_cell(layer_id=None):
    scope = 'RNN%d' % layer_id if layer_id else tf.get_variable_scope()
    with tf.variable_scope(scope):
      if issubclass(cell_class, rnn_cell.CustomLSTMCell):
        cell = cell_class(hidden_size, keep_prob) 
      else:
        cell = cell_class(hidden_size) 
        cell = tf.contrib.rnn.DropoutWrapper(cell, 
                                             input_keep_prob=1.0,
                                             output_keep_prob=keep_prob)
    return cell


  if num_layers > 1:
    cells = [_get_cell(i) for i in range(num_layers)]
    cell = rnn_cell.MultiRNNCell(cells)
  else:
    cell = _get_cell()

  return cell
