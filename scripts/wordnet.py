from nltk.corpus import wordnet as wn
import re

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
    #continue
    lem_id = int(raw_input('Enter an lemma ID: '))
    l = wn.lemmas(word)[lem_id]
    print l, l.definition()
    continue
    syn_id = int(raw_input('Enter an synset ID: '))

    synset = wn.synsets(word)[syn_id]
    print "Lemmas in %s:" % str(synset)
    if len(wn.synsets(word)) > 0:
      for i, l in enumerate(synset.lemmas()):
        print "{0}) '{1}'".format(i, l)
    lem_id = int(raw_input('Enter an lemma ID: '))
    lemma = synset.lemmas()[lem_id]
    print [x for x in dir(lemma) if not re.match('^_.+', x)]
    print "<%s> :" % str(lemma)
    print 'synset - ', lemma.synset()
    print 'hypernyms - ', lemma.hypernyms()
    print 'hyponyms - ', lemma.hyponyms()
    print 'pertainyms - ', lemma.pertainyms()
    print 'antonyms - ', lemma.antonyms()
    print 'usage_domains - ', lemma.usage_domains()

    #for i, l in enumerate(lemma.synsets()):
    #  print "{0}) '{1}'".format(i, l)
    
    


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
