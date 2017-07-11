#coding: utf-8
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

while True:
  word = raw_input('Enter a word: ')
  print word
  print lemmatizer.lemmatize(word)
