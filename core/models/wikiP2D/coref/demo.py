# coding: utf-8
#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np

import cgi
import BaseHTTPServer
import ssl

import tensorflow as tf
#import coref_model as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from metrics import ceafe

def load_eval_data(eval_path):
  with open(eval_path) as f:
    return [json.loads(jsonline) for jsonline in f]


def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  if 'gold_clusters' in example:
    #for i, cluster in enumerate(example["predicted_clusters"]):
    max_num_elems = max(len(example["predicted_clusters"]), 
                        len(example["gold_clusters"]))
    for i in xrange(max_num_elems):
      if i < len(example["predicted_clusters"]):
        cluster = example['predicted_clusters'][i]
        print(u"P{}: {}".format(i, [" ".join(words[m[0]:m[1]+1]) for m in cluster]))
      else:
        print(u"P{}: None".format(i))

      if i < len(example["gold_clusters"]):
        cluster = example['gold_clusters'][i]
        print(u"G{}: {}".format(i, [" ".join(words[m[0]:m[1]+1]) for m in cluster]))
      else:
        print(u"G{}: None".format(i))

  else:
    for i, cluster in enumerate(example["predicted_clusters"]):
      print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))
  

def get_aligned(predicted_clusters, gold_clusters, matching):
  # cluster_matching: gold[i] == predicted[j]
  max_num_clusters = max(len(predicted_clusters), len(gold_clusters))
  gold, predicted = [], []
  g_matched, p_matched = set(), set()
  for i, j in matching:
    gold.append(gold_clusters[i])
    predicted.append(predicted_clusters[j])
    g_matched.add(i)
    p_matched.add(j)
  if len(predicted_clusters) > len(gold_clusters):
    not_matched = set(xrange(max_num_clusters)) - p_matched
    for x in not_matched:
      predicted.append(predicted_clusters[x])
      gold.append(())
  else:
    not_matched = set(xrange(max_num_clusters)) - g_matched
    for x in not_matched:
      predicted.append(())
      gold.append(gold_clusters[x])
  return gold, predicted
  #   matched = [j for (_, j) in cluster_matching]
  #   not_matched = list(set(xrange(len(predicted_clusters))) - set(matched))
  #   predicted = [predicted_clusters[i] for i in matched + not_matched]
  #   return predicted, gold_clusters

def make_predictions(text, model, sample=None):
  #sampleがあればそちらを優先
  if not sample:
    example = create_example(text)
  else: 
    #tensorized_example = example = model.eval_data[sample_id]
    example = sample
  tensorized_example = model.tensorize_example(example, is_training=False)

  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)
  
  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)
  predicted_clusters, _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["predicted_clusters"] = predicted_clusters
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  if sample is not None:
    gold_clusters = sorted([tuple(sorted([tuple(m) for m in c], key=lambda x:x[0])) for c in example['clusters']], key=lambda x:x[0][0])
    _, _, _, _, cluster_matching = ceafe(predicted_clusters, gold_clusters)
    print 'matching', cluster_matching
    gold, predicted = get_aligned(predicted_clusters, gold_clusters, cluster_matching)
    example['predicted_clusters'] = predicted
    example['gold_clusters'] = gold
  return example

def run_model(model, eval_data, port=None):
  keyfile = None
  certfile = None

  if port is not None:
    class CorefRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
      def do_GET(self):
        idx = self.path.find("?")
        print 'path', (self.path)
        if idx >= 0:
          args = cgi.parse_qs(self.path[idx+1:])
          print 'args',args
          sample_id = int(args["sample_id"][0]) if "sample_id" in args else None
          sample = eval_data[sample_id] if sample_id >= 0 and sample_id < len(eval_data) else None
          if "text" in args:
            text_arg = args["text"]
            if len(text_arg) == 1 and len(text_arg[0]) <= 10000:
              text = text_arg[0].decode("utf-8")
              print(u"Document text: {}".format(text))
              print sample
              example = make_predictions(text, model, sample)
              print_predictions(example)
              self.send_response(200)
              self.send_header("Content-Type", "application/json")
              self.end_headers()
              self.wfile.write("jsonCallback({})".format(json.dumps(example)))
              return
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    server = BaseHTTPServer.HTTPServer(("", port), CorefRequestHandler)
    if keyfile is not None:
      server.socket = ssl.wrap_socket(server.socket, keyfile=keyfile, certfile=certfile, server_side=True)
    print("Running server at port {}".format(port))
    server.serve_forever()
  else:
    while True:
      text = raw_input("Document text: ")
      print_predictions(make_predictions(text, model))



if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  if len(sys.argv) > 2:
    port = int(sys.argv[2])
  else:
    port = None

  if len(sys.argv) > 3:
    # For https. See https://certbot.eff.org
    keyfile = sys.argv[3]
    certfile = sys.argv[4]
  else:
    keyfile = None
    certfile = None

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)
  eval_data = load_eval_data(config["eval_path"]) # not tensorized form.
  print 'Number of eval_data',len(eval_data)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)
    run_model(model, eval_data, port)
