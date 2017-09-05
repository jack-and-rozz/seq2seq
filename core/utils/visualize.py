#coding:utf-8
import matplotlib.pyplot as plt
from collections import Counter

def histgram(freqs, titles, file_path='plot.eps'):
  def plot_hist(ax, freq, title=None):
    f = freq
    #f = [(k,v) for k,v in Counter(freq).items()]
    #f = [x[1] for x in f]
    #width = 100
    #bins = int((max(f) - min(f))/width)
    #bins = 50
    bins = 500
    #ax.hist(f, bins=bins)
    ax.hist(f, bins=bins)
    ax.tick_params(labelcolor='b', top='off', bottom='on', left='on', right='off')
    if title:
      ax.set_title(title)

  # https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
  plt.rcParams["font.size"] = 10
  fig = plt.figure()
  plt.subplots_adjust(wspace=0.4, hspace=0.3)
  ax = fig.add_subplot(111) 
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
  ax.set_xlabel('freq')
  ax.set_ylabel('N')
  n_plots = len(freqs)
  for i, (freq, title) in enumerate(zip(freqs, titles)):
    sub_ax = fig.add_subplot(n_plots, 1, i+1)
    #sub_ax.set_xscale('log')
    #sub_ax.set_yscale('log')
    plot_hist(sub_ax, freq, title)
  fig.savefig(file_path)
