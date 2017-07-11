from nltk.corpus import wordnet as wn

def func1():
  while True:
    word = raw_input('Enter a word: ')
    print 'Synsets:'
    for i, w in enumerate(wn.synsets(word)):
      print "{0}) '{1}' -- definition: '{2}'".format(i,w.name(),w.definition())

    print 'Lemmas:'
    for i, w in enumerate(wn.lemmas(word)):
      print "{0}) '{1}'".format(i, str(w))
    print ''
print wn.VERB
func1()

def func2():
  while True:
    word = raw_input('Enter a word: ')
    print "You entered: '%s'" % word
    
    wn_word = wn.synsets(word)
    print "Choose a Synset of your entry, '%s'" % (word,) 
    for i,w in enumerate(wn_word):
      print "{0}) '{1}' -- definition: '{2}'".format(i,w.name(),w.definition())

    p = raw_input('Your selection: ')
    synset = wn.synsets(word)[int(p)]
    wn_word = synset.name()
    print ("lemmas:")
    for i, l in enumerate(synset.lemmas()):
      print ("{0}) '{1}'".format(i, l.name()))
    print ("")
