# coding: utf-8 
import math, time, sys
import tensorflow as tf
from core.utils import common
from core.models.base import ModelBase
from core.seq2seq import rnn
import numpy as np

class DescriptionGeneration(ModelBase):
  def __init__(self, config, encoder, w_vocab,
               activation=tf.nn.tanh):
    self.activation = activation
    self.encoder = encoder

    ## Seq2Seq for description generation
    with tf.variable_scope('Decoder') as scope:
      self.d_decoder_cell = rnn.setup_cell(
        config.cell_type, config.hidden_size,
        num_layers=config.num_layers, 
        in_keep_prob=config.in_keep_prob, 
        out_keep_prob=config.out_keep_prob,
        state_is_tuple=config.state_is_tuple)
      #self.d_decoder =decoders.RNNDecoder()

