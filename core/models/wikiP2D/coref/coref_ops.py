import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

kernel_path = "./core/models/wikiP2D/coref/coref_kernels.so"
coref_op_library = tf.load_op_library(kernel_path)

spans = coref_op_library.spans
tf.NotDifferentiable("Spans")

antecedents = coref_op_library.antecedents
tf.NotDifferentiable("Antecedents")

extract_mentions = coref_op_library.extract_mentions
tf.NotDifferentiable("ExtractMentions")

distance_bins = coref_op_library.distance_bins
tf.NotDifferentiable("DistanceBins")
