# coding: utf-8
import tensorflow as tf
import random
import numpy as np
from padding import padding
X = 100
N = 5 # num of examples
M = 3 # len of doc
L = 7 # len of sent
V = 5 # id range
batch_size = 3

tf_config = tf.ConfigProto(
  log_device_placement=False,
  allow_soft_placement=True,
  gpu_options=tf.GPUOptions(
    allow_growth=True, 
  )
)

sess = tf.InteractiveSession(config=tf_config)

# 2階以上のpaddingはしてくれない？
features = [np.array([[np.random.randint(1, V+1) for _ in range(np.random.randint(1, L+1))] for _ in range(np.random.randint(1, M+1))]) for _ in range(N)]
#features = np.array([[[np.random.randint(0, L) for _ in range(np.random.randint(1, L))] for _ in range(3)] for _ in range(N)])

padded_shapes = ([None, None], [])

labels = np.array([np.random.randint(1, X) for _ in range(N)])

# 確認した所、from_generatorはGPU使用時もCPUで実行されるのでデータに依るメモリ溢れは無さそう


# padded_batchはexample間でのpaddingはしてくれるが、example内でのpaddingはしてくれないのでgeneratorの時点でする必要がある。例えば、example間で文の数・単語の数が違っていてもいいが、あるexampleのそれぞれの文の単語の数は同じ（= numpy.arrayである）必要がある。generatorではなく初めにデータを読んでIDに変換する時にそれを行うべき？
def generator():
  for f, l in zip(features, labels):
    #print('padded_f', padding(f, minlen=[None], maxlen=[None]))
    yield (
      padding(f, minlen=[None], maxlen=[None]), 
      l)

dataset = tf.data.Dataset.from_generator(
  generator,
  output_types=(tf.int32, tf.int32))

train_dataset = dataset #.shuffle(2)
test_dataset = dataset

train_dataset = train_dataset.padded_batch(batch_size, padded_shapes)
test_dataset = test_dataset.padded_batch(batch_size, padded_shapes)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

next_element = iterator.get_next()
calc = tf.reduce_sum(next_element[0], axis=1)

i = 0
sess.run(train_init_op)
print('=== train ==')
while True:
  try:
    inp, res = sess.run([next_element[0], calc])
    print('input: ', inp, inp.shape)
    print('res  : ', res)
    #print(features[i*batch_size:(i+1)*batch_size])
  except tf.errors.OutOfRangeError:
    break
  i += 1
  break
exit(1)
print('=== test ==')
sess.run(test_init_op)
while True:
  try:
    inp, res = sess.run([next_element[0], calc])
    print('input: ', inp, inp.shape)
    print('res  : ', res)
    #print(features[i*batch_size:(i+1)*batch_size])
  except tf.errors.OutOfRangeError:
    break
  i += 1
  #sess.run(iterator.initializer)


exit(1)


#features_placeholder = tf.placeholder(features.dtype, [None, None])
labels_placeholder = tf.placeholder(labels.dtype, [None])
##placeholders = (features_placeholder, labels_placeholder)
#placeholders = features_placeholder
#dataset = tf.data.Dataset.from_tensor_slices(placeholders)

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
#dataset = tf.data.Dataset.from_tensor_slices([features])
#padded_shapes = (np.array([None, None]), np.array([None]))
padded_shapes = [tf.Dimension(None)]
dataset = dataset.padded_batch(batch_size, padded_shapes)
# [Other transformations on `dataset`...]
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
  # sess.run(iterator.initializer, feed_dict={features_placeholder: features,
  #                                           labels_placeholder: labels})
  #sess.run(iterator.initializer, feed_dict={features_placeholder: features})
  sess.run(iterator.initializer)





