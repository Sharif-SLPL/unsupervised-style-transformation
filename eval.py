# Evaluation Script - Fluency and Semantic Similarity
from __future__ import unicode_literals
import torch
from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForMaskedLM
)
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm, trange
# from hazm import *
import pandas as pd
import os
from torch.utils.data import DataLoader
import random
import argparse
import numpy as np
from transformers.utils import logging
logging.set_verbosity(40) # Suppress warnings


parser = argparse.ArgumentParser(description='Run evaluation script.')
parser.add_argument('--input', type=str, help='Name of text file to evaluate', default='input')

args = parser.parse_args()

Sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
Bert_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base") 
Bert_model = AutoModelForMaskedLM.from_pretrained("HooshvareLab/bert-fa-zwnj-base")

def sentence_similarity(i, t):
    emb1 = Sbert.encode(i, convert_to_tensor=True)
    emb2 = Sbert.encode(t, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb1, emb2)
    return cosine_scores

class Eval:
    def __init__(self):
        self.probs = []
        self.threshold_probs = []

    def mask_words(self, input_tokens, idx):
        """Mask each token and compute its probability

        Args:
            input_tokens (list): A tokenized sentence
            idx (int): Position of current token to be masked
        """
        label = input_tokens[idx]
        input_tokens[idx] = "[MASK]"
        masked_sentence = " ".join(input_tokens)
        mask_filler = pipeline("fill-mask", model=Bert_model, tokenizer=Bert_tokenizer)
        mask_filler = mask_filler(masked_sentence, targets = [label])
        # mask_filler = mask_filler(masked_sentence)[0]["sequence"]
        self.probs.append(mask_filler[0]["score"])

    def threshold_finder(self, data):
        """Text corpus

        Args:
            data (list): List of text

        Returns:
            int: Average probability of each token in the whole text
        """
        rn = 1000
        for i in tqdm(range(rn)):
            inp = data["text2"][i]
            tokenized = inp.split()
            sen_len = len(tokenized)
            self.probs = []
            for j in range(len(inp.split())):
                self.mask_words(inp.split(), j)
            # Avg probability of each token in a sentence
            self.threshold_probs.append(sum(self.probs)/sen_len)
        # Avg probability of each token in the whole text
        print(sum(self.threshold_probs)/rn)

        return sum(self.threshold_probs)/rn

    def score(self, data):
        """Evaluate fluency and semantic similarity of test data 

        Args:
            data (list): Test data
        """
        sim = []
        fl = []
        for i in tqdm(range(len(data))):
            inp = data["text"][i]
            target = data["text2"][i]
            if isinstance(inp, float):
              continue
            # print(len(target.split(" ")), len(target))
            cos_s = sentence_similarity(inp, target)
            cos_s = cos_s[0].detach().cpu().numpy()[0]

            tokenized = inp.split()
            sen_len = len(tokenized)
            self.probs = []
            for j in range(len(inp.split())):
                self.mask_words(inp.split(), j)

            # th 0.0.1631687008989672
            threshold = sen_len * 0.1631
            score = round((sum(self.probs)/threshold), 2)
            score = 1 if score >= threshold else score
            sent_score = (100 * score)/threshold
            # sent_score = 1 if score >= threshold else score
            sim.append(round(cos_s * 100, 2))
            fl.append(sent_score)
            # fl.append(round(sent_score * 100, 2))

        print(len(sim), len(fl))
        print("Sim:", sum(sim)/len(sim))
        print("Fl:", sum(fl)/len(fl))

evaluation = Eval()

df = pd.read_csv(args.input.lower()+'.txt', sep=",", header=None)
df.columns = ["text", "text2"]

evaluation.score(df)

