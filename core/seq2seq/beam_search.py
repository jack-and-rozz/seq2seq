# coding: utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import copy, math
import numpy as np
# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell# tf == 1.1
from tensorflow.python.ops import rnn_cell_impl as rnn_cell # tf >= 1.2

#linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access
linear = rnn_cell._linear  # pylint: disable=protected-access

from core.vocabulary.base import EOS_ID

def _extract_beam_search(embedding, beam_size, batch_size,
                         output_projection=None,
                         update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  num_symbols, embedding_size = embedding.get_shape().as_list()

  if output_projection == None:
    raise ValueError('output_projection must not be None.')

  def tile_from_beam_to_vocab(t):
    # For tf.nn.top_k, do tiling information about beam for each word. 
    return tf.tile(tf.reshape(t, [-1, beam_size, 1]), [1, 1, num_symbols])


  def divide_index_by_batch(index_matrix):
    # For tf.gather_nd, transform the shape of indices 
    # from [batch_size, beam_size] to [batch_size, beam_size, 2]
    # ex) (batch_size=2, beam_size=3) : [[0,1,2], [2,1,0]] -> [[[0,0],[0,1],[0,2]], 
    #                                                          [[1,2],[1,1],[1,0]]]]
    replicated_first_indices = tf.tile(
      tf.expand_dims(tf.range(batch_size), dim=1), 
      [1, tf.shape(index_matrix)[1]])
    return tf.stack([replicated_first_indices, index_matrix], axis=2)

  def loop_function(i, prev, state, 
                    log_beam_probs, beam_path, beam_symbols,
                    path_lengthes, is_finished_beam):
    output_size = prev.get_shape().as_list()[-1]
    state_size = state.get_shape().as_list()[-1]

    if i == 1:
      # todo: prevだけではなくstateも分岐
      probs = nn_ops.xw_plus_b(
        prev, output_projection[0], output_projection[1])
      probs = tf.log(tf.nn.softmax(probs))

      best_probs, indices = tf.nn.top_k(probs, beam_size)

      # initialize length and EOS flags for each beam.
      path_lengthes = tf.fill([batch_size, beam_size], 1.0)
      is_finished_beam = tf.fill([batch_size, beam_size], False)
      # expand previous states to beams. (e.g. batch_size=beam_size=2: [a, b] -> [a, a, b, b])
      prev = tf.gather(prev, tf.tile(tf.expand_dims(tf.range(batch_size), dim=1), [1, beam_size]))
      # prev: [batch, beam, hidden] -> [batch * beam, hidden]
      prev = tf.reshape(prev, [-1, output_size])

      state = tf.gather(state, tf.tile(tf.expand_dims(tf.range(batch_size), dim=1), [1, beam_size]))
      state = tf.reshape(state, [-1, state_size])

    else:
      probs = nn_ops.xw_plus_b(
        prev, output_projection[0], output_projection[1])
      probs = tf.log(tf.nn.softmax(probs))
      probs = tf.reshape(probs, [-1, beam_size * num_symbols])

      # divide probs by the length of each beam (length penalty) and select top-k.
      pl = tf.reshape(tile_from_beam_to_vocab(path_lengthes), 
                      [-1, beam_size*num_symbols])
      best_probs, indices = tf.nn.top_k(probs / pl, beam_size)
    symbols = indices % num_symbols 
    beam_parent = indices // num_symbols

    beam_symbols.append(symbols)
    beam_path.append(beam_parent)
    log_beam_probs.append(best_probs)

    is_finished_beam = tf.logical_or(
      tf.gather_nd(is_finished_beam, divide_index_by_batch(beam_parent)),
      tf.equal(symbols, tf.constant(EOS_ID))
    )
    
    path_lengthes = tf.gather_nd(path_lengthes, divide_index_by_batch(beam_parent))
    path_lengthes += tf.to_float(tf.logical_not(is_finished_beam))

    beam_state = tf.gather_nd(
      tf.reshape(state, [batch_size, beam_size, state_size]), 
      divide_index_by_batch(beam_parent))
    emb_prev = embedding_ops.embedding_lookup(embedding, symbols)

    # [batch, beam, embedding] -> [batch * beam, embedding]
    beam_state = tf.reshape(beam_state, [-1, state_size])
    emb_prev  = tf.reshape(emb_prev, [-1, embedding_size])
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev, beam_state, path_lengthes, is_finished_beam
  return loop_function


def beam_rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None,output_projection=None, beam_size=10):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """

  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    prev = None
    log_beam_probs, beam_path, beam_symbols = [],[],[]
    path_lengthes, is_finished_beam = None, None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp, state, path_lengthes, is_finished_beam  = loop_function(
            i, prev, state,
            log_beam_probs, beam_path, beam_symbols,
            path_lengthes, is_finished_beam)

      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      if loop_function is not None:
        prev = output
  # from time-major to batch_major
  beam_path = tf.stack(beam_path, axis=1)
  beam_symbols = tf.stack(beam_symbols, axis=1)
  # [batch*beam, state] -> [batch, beam, state]
  state = tf.reshape(
    state, 
    [-1, beam_path.get_shape().as_list()[-1], state.get_shape().as_list()[-1]]
  )
  return beam_path, beam_symbols, state


def beam_attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False, output_projection=None, beam_size=10):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())

  if output_size is None:
    output_size = cell.output_size
  with variable_scope.variable_scope(scope or "attention_decoder"):
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(variable_scope.get_variable("AttnV_%d" % a,
                                           [attention_vec_size]))
      state_size = int(initial_state.get_shape().with_rank(2)[1]) 
    states =[]
    for kk in range(1):
        states.append(initial_state)
    state = tf.reshape(tf.concat(states, 0), [-1, state_size])
    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          # for c in range(ct):
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])

    if initial_state_attention:
       attns = []
       attns.append(attention(initial_state))
       tmp = tf.reshape(tf.concat(attns, 0), [-1, attn_size])
       attns = []
       attns.append(tmp)

    log_beam_probs, beam_path, beam_symbols = [],[],[]
    path_lengthes, is_finished_beam = None, None

    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None :
        with variable_scope.variable_scope("loop_function", reuse=True):
          if prev is not None:
            inp, state, path_lengthes, is_finished_beam  = loop_function(
              i, prev, state,
              log_beam_probs, beam_path, beam_symbols,
              path_lengthes, is_finished_beam)

            #inp, state, path_lengthes, is_finished_beam  = loop_function(
            #  prev, i, log_beam_probs, beam_path, beam_symbols,
            #  path_lengthes, is_finished_beam)


      input_size = inp.get_shape().with_rank(2)[1]
      x = linear([inp] + attns, input_size, True)
      cell_output, state = cell(x, state)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
          attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      if  i ==0:
        states =[]
        for kk in range(beam_size):
          states.append(state)
        state = tf.reshape(tf.concat(states, 0), [-1, state_size])
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)

    #beam_path = tf.reshape(tf.concat(beam_path, 0), [-1, beam_size])
    #beam_symbols = tf.reshape(tf.concat(beam_symbols, 0),[-1,beam_size])
  return beam_path, beam_symbols, state



def follow_path(path, symbol, beam_size):
  best_symbols = []
  curr = range(beam_size)
  num_steps = len(path)
  for i in range(num_steps-1, -1, -1):
    best_symbols.append(symbol[i][curr[0]]) 
    for j in range(beam_size):
      curr[j] = path[i][curr[j]] #親を更新

  best_symbols = best_symbols[::-1]
  # If there is an EOS symbol in outputs, cut them at that point.
  if EOS_ID in best_symbols:
    best_symbols = best_symbols[:best_symbols.index(EOS_ID)]
  return best_symbols

# old version
# def follow_path(path, symbol, beam_size):
#   paths = []
#   for _ in range(beam_size):
#     paths.append([])

#   curr = range(beam_size)
#   num_steps = len(path)
#   for i in range(num_steps-1, -1, -1):
#     for j in range(beam_size):
#       paths[j].append(symbol[i][curr[j]])
#       curr[j] = path[i][curr[j]]

#   for j in range(beam_size):
#     foutputs = [int(logit) for logit in paths[j][::-1]]
#   # If there is an EOS symbol in outputs, cut them at that point.
#   if EOS_ID in foutputs:
#     foutputs = foutputs[:foutputs.index(EOS_ID)]
#   rec = foutputs
#   return rec

