# coding: utf-8 
import tensorflow as tf
from core.utils import common
from core.models.base import ModelBase
import numpy as np

class LanguageModel(ModelBase):
  def __init__(self, config, encoder, w_vocab,
               activation=tf.nn.tanh):
    self.activation = activation
    self.encoder = encoder
