#cofing:utf-8

import argparse, commands, os, re

import seaborn as sns
import matplotlib.pyplot as plt


def read_log(fp, max_epoch):
  train_pat = 'Epoch [0-9]+ \(train\): epoch-time (.+), step-time (.+), ppx (.+)'
  valid_pat = 'Epoch [0-9]+ \(valid\): epoch-time (.+), step-time (.+), ppx (.+)'
  times = []
  valid_log = []
  for l in open(fp):
    tm = re.search(train_pat, l)
    vm = re.search(valid_pat, l)
    if tm:
      epoch_time = float(tm.group(1)) / 3600
      times.append(epoch_time)
      tm = None
    if vm:
      valid_ppx = float(vm.group(3))
      valid_log.append(valid_ppx)
      vm = None
  times = [sum(times[:i+1]) for i, _ in enumerate(times)]
  res = [(t, v) for t,v in zip(times, valid_log)]
  if max_epoch:
    res = res[:int(max_epoch)]
  return res

def main(args):
  if args.grep_query:
    file_path = [m for m in commands.getoutput('ls -d %s' % args.proj_dir + '/*/train.log | grep %s' % args.grep_query).split()]
  else:
    file_path = [m for m in commands.getoutput('ls -d %s' % args.proj_dir + '/*/train.log').split()]
  logs = [read_log(fp, args.max_epoch) for fp in file_path]
  logs = [(fp, l) for fp, l in zip(file_path, logs) if len(l) > 0]

  for fp, log in logs:
    print fp
    fp = re.search('.+/(.+?)/train\.log', fp).group(1)
    X, Y = map(list, zip(*log))
    print X,Y
    plt.plot(X, Y, label=fp, marker='o')
  #plt.xlim(0)
  #plt.set_ylabel
  plt.xlabel('time (hour)')
  plt.ylabel('ppx')
  plt.legend()
  plt.savefig('graph.png')


if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('proj_dir', help ='')
  parser.add_argument('--grep_query', default='gpu', help ='')
  parser.add_argument('--max_epoch', default=None, help ='')
  args = parser.parse_args()
  main(args)
