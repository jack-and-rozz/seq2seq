import copy
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell

def setup_cell(cell_type, hidden_size, num_layers=1, 
               in_keep_prob=1.0, out_keep_prob=1.0, state_is_tuple=True):
  cell = getattr(rnn_cell, cell_type)(hidden_size, 
                                      reuse=tf.get_variable_scope().reuse) 
  if in_keep_prob < 1.0 or out_keep_prob < 1.0:
    cell = rnn_cell.DropoutWrapper(cell, 
                                   input_keep_prob=in_keep_prob,
                                   output_keep_prob=out_keep_prob)
  if num_layers > 1:
    cells = [copy.deepcopy(cell) for _ in xrange(num_layers)]
    cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  return cell
