#coding: utf-8

def get_rank(probs):
  '''
  r : a list of link connection probabilities where a correct one is inserted at the beginning of candidates.
  '''
  ranks = sorted([(idx, p) for idx, p in enumerate(probs)], 
                 key=lambda x: -x[1])
  return [rank for rank, (idx, _) in enumerate(ranks) if idx == 0][0] + 1


def mrr(ranks):
  return sum([1.0 / r for r in ranks]) / float(len(ranks))

def hits_k(ranks, k=10):
  return 100.0 * len([r for r in ranks if r <= 10]) / len(ranks)
  
