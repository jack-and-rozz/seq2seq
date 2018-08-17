# coding: utf-8 
import copy, sys
from core.utils.common import dotDict, flatten_batch, dbgprint, RED, BLUE, RESET, UNDERLINE, BOLD, GREEN
from core.vocabulary.base import _PAD, _UNK

def decorate_text(text, vocab, link=None, word_ids=None):
  text = copy.deepcopy(text)
  num_words = len([w_id for w_id in text if w_id != vocab.word.token2id(_PAD)])
  text = text[:num_words]

  if link is not None:
    begin, end = link
    for j in range(begin, end+1):
      text[j] = BLUE + text[j] + RESET
  if word_ids is not None:
    for j in range(len(text)):
      if word_ids[j] == vocab.word.token2id(_UNK):
        text[j] = UNDERLINE + text[j] + RESET
  return ' '.join(text)

def print_example(example, vocab, prediction=None):
  '''
  Args:
  - example: An example in a batch obtained from 'flatten_batch(batch)'.
  '''
  if example.title:
    print('<Title>', example.title.raw)
  print('<Contexts>')
  for i in range(len(example.contexts.raw)):
    text = decorate_text(example.contexts.raw[i], vocab,
                         link=example.contexts.link[i],
                         word_ids=example.contexts.word[i])
    print(text)
  if example.desc:
    desc = decorate_text(example.desc.raw, vocab,
                         word_ids=example.desc.word)
    print ('<Desc>', desc)
  print ('<Category>', vocab.category.id2token(example.category.label))
  if prediction is not None:
    print ('<Prediction>', vocab.category.id2token(prediction))

# def print_batch(batch, prediction, vocab):
#   print ('<Context>')
#   for i in range(len(batch.contexts.raw)):
#     text = decorate_text(batch.contexts.raw[i], 
#                          vocab,
#                          link=batch.contexts.link[i],
#                          word_ids=batch.contexts.word[i])
#     print('- ' + text)
#   print ('<Gold>      : %s' % batch.category.raw)
#   print ('<Predicton> : %s' % vocab.category.id2token(prediction))

def evaluate_and_print(flat_batches, predictions, vocab):
  n_data = 0
  n_success = 0
  for i, (b, p) in enumerate(zip(flat_batches, predictions)):
    n_data += 1
    is_success = 'Failure'
    if b.category.label == p:
      is_success = 'Success' 
      n_success += 1
    _id = '<%04d:%s>' % (i, is_success)
    print (_id)
    print_example(b, vocab, prediction=p)
    print ('')

  acc = 1.0 * n_success / n_data
  print('Accuracy: %.3f (%d/%d)' % (acc, n_success, n_data))
  return acc
