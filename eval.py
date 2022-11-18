# Evaluation Script - Style Accuracy, Fluency, and Semantic Similarity
from __future__ import unicode_literals
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer
import torch
from torch import cuda
from torch import nn
from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForMaskedLM
)
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm, trange
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import numpy as np
from transformers.utils import logging
import logging as logg
logg.basicConfig(level=logging.ERROR)
logging.set_verbosity(40) # Suppress warnings

device = 'cuda' if cuda.is_available() else 'cpu'

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

class ClassifyData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("xlm-roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class Eval:
    def __init__(self):
        self.probs = []
        self.threshold_probs = []
        # Load RoBERTa model checkpoints
        base_checkpoint = "xlm-roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, truncation=True, do_lower_case=True)
        self.roberta = RobertaClass()
        self.roberta.to(device)
        self.roberta = torch.load('roberta_style_classifier.pt')
        self.roberta.eval()

    def predict_class(self, input):
        inp = self.tokenizer(input, return_tensors='pt', return_token_type_ids=True).to(device)
        t = self.roberta(**inp)
        m = nn.Sigmoid()
        output = m(t).detach().cpu().numpy()
        predicted_class = np.argmax(output)

        return predicted_class, output

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

    def style_accuracy(self, input):
        cumulative_prob = 0
        formals_idx = []
        paraphrase_informal = []
        l = []

        for idx, i in tqdm(enumerate(input)):
            try:
                pred = self.predict_class(i)
                res = pred[0]
                if res == 0:
                    formals_idx.append(idx+1)
                    paraphrase_informal.append((input[idx], input[idx]))
                cumulative_prob += pred[1][0][1]
                l.append(res)
            except Exception as e:
                pass

        style_acc = l.count(1) / len(l)

        return style_acc, l

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
        print("="*200)
        st_acc, pred = self.style_accuracy(data["text"])
        print(f"Accuracy: {'%.2f' % st_acc}% of data ({pred.count(1)} out of {len(pred)}) were formalized.")
        print("Similarity:", sum(sim)/len(sim))
        print("Fluency:", sum(fl)/len(fl))

evaluation = Eval()

df = pd.read_csv(args.input.lower()+'.csv', sep=",", header=None, on_bad_lines='skip')
df.columns = ["text", "text2"]
df = df.drop_duplicates(subset=['text2'], keep='last')

evaluation.score(df)