from __future__ import unicode_literals
import random
from hazm import *
import pandas as pd
from text_processor import *
from config import TRAIN_CONFIG

pd.options.display.max_colwidth = 200
tagger = POSTagger(model='models/postagger.model')

# TODO: Implement tips on this: https://www.niu.edu/writingtutorial/style/formal-and-informal-style.shtml

def lexical_diversity(tokens):
  lexical_diversity = len(set(tokens)) / len(tokens)
  return lexical_diversity

def pos_diversity(tokens):
  tags = tagger.tag(tokens)
  return tags

def read_txt(path, shuffle=False, lim=-1):
  data = []
  with open(path, "r") as f:
    for i, line in enumerate(f):
      data.append(line)
  if shuffle:
    random.shuffle(data)
  return data[0:lim]

def load_data(input):
  data = []
  for i in range(len(input)):
    lxd = 0
    for j in range(2):
      line = input[j][i]
      tokenized = word_tokenize(line)
      ending_tag = tagger.tag(tokenized)[-1][1] # Replace with above post div function maybe
      # if ending_tag is not "V" or ending_tag is not "PUNC":
      #   print(tagger.tag(tokenized)[-1])
      #   break
      lxd += lexical_diversity(tokenized)
  # Sum of scores. Score of each line (original & paraphrase) best results when it's above .85. Duplicate sentences
  # could be probelmatic of course!
    if lxd >= 1.7: 
      # print("lexical diversity:", lxd)
      data.append((input[0][i], input[1][i]))

  return data[:TRAIN_CONFIG["dataset_limit"]]

# TODO: replace numbers and normalize + other stuff
def preprocess_text(data):
  text = []
  for i in data:
    i.replace(r"")