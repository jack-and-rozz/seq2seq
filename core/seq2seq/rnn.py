# coding: utf-8
import copy
import tensorflow as tf
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
import rnn_cell
#from core.extensions import rnn_cell as shared_rnn_cell

def setup_cell(cell_type, hidden_size, num_layers=1, 
               shared=False,
               keep_prob=1.0, state_is_tuple=True):
  #cell_module = rnn_cell if not shared else shared_rnn_cell
 
  cell_module = rnn_cell
  cell_class = getattr(cell_module, cell_type)

  def _get_cell():
    if issubclass(cell_class, rnn_cell.CustomLSTMCell):
      cell = cell_class(hidden_size, keep_prob,
                        reuse=tf.get_variable_scope().reuse) 
    else:
      cell = cell_class(hidden_size, 
                        reuse=tf.get_variable_scope().reuse)
    return cell
  #if in_keep_prob < 1.0 or out_keep_prob < 1.0:
  #cell = rnn_cell.DropoutWrapper(cell, 
  #                               input_keep_prob=in_keep_prob,
  #                               output_keep_prob=out_keep_prob)

  if num_layers > 1:
    #cells = [copy.deepcopy(cell) for _ in xrange(num_layers)]
    cells = [_get_cell() for _ in xrange(num_layers)]
    cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  else:
    cell = _get_cell()

  # * 呼び出す際にEncoder側でscopeを保存しておけばSharingWrapperとか要らなかった
  #if shared: 
  #  cell = rnn_cell.SharingWrapper(cell)

  return cell
