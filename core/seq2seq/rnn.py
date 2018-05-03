# coding: utf-8
import copy
import tensorflow as tf
from . import rnn_cell

def setup_cell(cell_type, hidden_size, num_layers=1, keep_prob=None):
  cell_module = rnn_cell
  cell_class = getattr(cell_module, cell_type)

  def _get_cell():
    if issubclass(cell_class, rnn_cell.CustomLSTMCell):
      cell = cell_class(hidden_size, keep_prob,
                        reuse=tf.get_variable_scope().reuse) 
    else:
      cell = cell_class(hidden_size, 
                        reuse=tf.get_variable_scope().reuse)
      cell = tf.contrib.rnn.DropoutWrapper(cell, 
                                           input_keep_prob=1.0,
                                           output_keep_prob=keep_prob)
    return cell


  if num_layers > 1:
    cells = [_get_cell() for _ in range(num_layers)]
    cell = rnn_cell.MultiRNNCell(cells)
  else:
    cell = _get_cell()

  return cell
