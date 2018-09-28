# coding: utf-8 
import copy, sys
from core.utils.common import dotDict, flatten_batch, dbgprint, RED, BLUE, RESET, UNDERLINE, BOLD, GREEN
from core.vocabulary.base import _PAD, _UNK
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
    print('- Title :', example.title.raw)
  print('- Contexts :')
  for i in range(len(example.contexts.raw)):
    text = decorate_text(example.contexts.raw[i], vocab.encoder,
                         link=example.contexts.link[i],
                         word_ids=example.contexts.word[i])
    print(text)

  desc = decorate_text(example.desc.raw, vocab.decoder,
                       word_ids=example.desc.word)
  print ('- Reference :', desc)
  print ('- Hypothesis :',prediction)

def calc_bleu(reference, hypothesis):
  assert type(reference) == type(hypothesis)
  bleu = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method2) * 100.0
  return bleu

def evaluate_and_print(flat_batches, predictions, vocab):
  n_data = 0
  sum_bleu = 0
  for i, (b, p) in enumerate(zip(flat_batches, predictions)):
    n_data += 1
    hypothesis = vocab.decoder.word.ids2tokens(p)
    reference = ' '.join(b.desc.raw)
    bleu = calc_bleu(reference, hypothesis)
    sum_bleu += bleu
    _id = '[%04d]' % (i)
    print (_id)
    print_example(b, vocab, prediction=hypothesis)
    print ('')
  ave_bleu = 1.0 * sum_bleu / n_data
  print('Average BLEU: %.3f' % (ave_bleu))
  return ave_bleu
