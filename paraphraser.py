from __future__ import unicode_literals
import torch
from transformers import (
    Adafactor,
    T5ForConditionalGeneration,
    AutoTokenizer
)
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm, trange
from hazm import *
import pandas as pd
from utils import get_batch, cleanup, load_model, draw_loss_plot
from style_classifier import *
import os
from config import BASE_CONFIG, GENERATION_CONFIG, TRAIN_CONFIG
from text_processor import read_txt
from style_classifier import *

pd.options.display.max_colwidth = 200

class Paraphraser:
    def __init__(self, num_outputs=1, conditional_generation=False, device="cuda"):
        self.device = torch.device(device)
        self.para_model_path = BASE_CONFIG["para_model_path"]
        # Model's path if exist to be used for inference. Save the trained model in this path
        self.local_para_model_path = BASE_CONFIG["local_para_model_path"]
        if os.path.exists(self.local_para_model_path):
            print('Loading model from local checkpoint')
            self.para_model_path = self.local_para_model_path
        self.model = T5ForConditionalGeneration.from_pretrained(self.para_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.para_model_path)
        self.model.to(self.device)
        self.conditional_generation = conditional_generation
        self.num_outputs = num_outputs
        self.loss_history = []

    def train(self, data):
        # TODO: Remove
        # qq = [0.5, 0.75, 0.9, 0.99, 1]
        # print(pd.Series(len(self.tokenizer(get_batch(data, mult=3)[0], padding=True)['input_ids'][0]) for _ in range(1000)).quantile(qq))
        # print(pd.Series(len(self.tokenizer(get_batch(data, mult=3)[1], padding=True)['input_ids'][0]) for _ in range(1000)).quantile(qq))

        optimizer = Adafactor(
            self.model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"], #1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=True,
            # scale_parameter=False,
            warmup_init=TRAIN_CONFIG["warmup_init"],
        )
        self.model.train()
        mult = 4
        batch_size = mult * 8
        max_len = 128 
        epochs = 500
        accumulation_steps = 32
        save_steps = 2000

        window = 2000
        ewm = 0
        errors = 0

        # tq = trange(int(1000 * epochs / mult)) #1_000_000
        tq = trange(int(len(data)/2 * epochs)) #1_000_000
        cleanup()

        # TODO: Freeze layers during training if needed
        # for i, m in enumerate(model.encoder.block): 
        #   #Only un-freeze the last n transformer blocks in the decoder
        #   # if i+1 > 12:
        #   for parameter in m.parameters():
        #       parameter.requires_grad = True # False to freeze layers

        # for name, param in model.named_parameters():
        #   print(name, param.requires_grad)

        # Freeze layer(s) by name
        # for name, param in model.named_parameters():
        #   if name.startswith("bert.encoder.layer.1"):
        #     param.requires_grad = False
        #   if name.startswith("bert.encoder.layer.2"):
        #     param.requires_grad = False

        for i in tq:
            xx, yy = get_batch(data, mult=mult)
            try:
                # Swap x & y. Actually should be batched like this from the start
                x = self.tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(self.device)
                y = self.tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(self.device)

                y.input_ids[y.input_ids==0] = -100  #to make sure we have correct labels for T5 text generation
                loss = self.model(
                    input_ids=x.input_ids,
                    attention_mask=x.attention_mask,
                    labels=y.input_ids,
                    decoder_attention_mask=y.attention_mask,
                    return_dict=True
                ).loss
                loss.backward()
                
            except RuntimeError as e:
                print(e)
                errors += 1
                loss = None
                cleanup()
                continue

            w = 1 / min(i+1, window)
            ewm = ewm * (1-w) + loss.item() * w
            tq.set_description(f'loss: {ewm}')

            # Save history when an epoch is completed
            if i > 0 and i % (len(data)/mult) == 0:
              self.loss_history.append((i/len(data), ewm))
            if i%10 ==0:# redundant maybe but save loss every 10 steps TODO: revise loss_history
                self.loss_history.append(loss.item())

            if i % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                cleanup()
            
            if i % window == 0 and i > 0:
                print(ewm, errors)
                errors = 0
                cleanup()
                # optimizer.param_groups[0]['lr'] *= 0.999
            if i % save_steps == 0 and i > 0:
                self.save_model()
                print('saving...', i, optimizer.param_groups[0]['lr'])
                # self.loss_history.append((round(i/len(data)), ewm))
                draw_loss_plot(self.loss_history)
            # Early stopping:
            if ewm < TRAIN_CONFIG["loss_threshold"]:
                # self.loss_history.append((round(i/len(data)), ewm))
                draw_loss_plot(self.loss_history)
                self.save_model()
                print('saving...', i, optimizer.param_groups[0]['lr'])
                print("Stopping Training...")
                break  # early stop criterion is met, we can stop now
        
    def save_model(self):
        self.model.save_pretrained(self.local_para_model_path)
        self.tokenizer.save_pretrained(self.local_para_model_path)

    def paraphrase_text(self, text):
        self.model.eval()
        self.encode_text(text)
        self.decode_text(text)
        print(self.decoded_text)

    def batch_paraphrase_text(self, path):
        self.model.eval()
        data = read_txt(path)
        for text in data:
            self.encode_text(text)
            self.decode_text(text)
            print(text, self.decoded_text, "\n")

    def encode_text(self, input_text):
        self.input_tokenized = self.tokenizer(input_text, return_tensors='pt', padding=True).to(self.model.device)
        sent_len = self.input_tokenized.input_ids.shape[1]
        max_size = int(sent_len)
        #TODO: Add temperature=0.9, top_p=0.6, top_k=50 with default values
        # self.encoded_text = self.model.generate(**self.input_tokenized, encoder_no_repeat_ngram_size=4, no_repeat_ngram_size=4, min_length=sent_len, num_return_sequences=self.num_outputs, do_sample=GENERATION_CONFIG["do_sample"], num_beams=GENERATION_CONFIG["num_beams"], max_length=max_size)
        self.encoded_text = self.model.generate(**self.input_tokenized, temperature=GENERATION_CONFIG["temperature"], no_repeat_ngram_size=2, min_length=sent_len, early_stopping=True, num_return_sequences=self.num_outputs, do_sample=GENERATION_CONFIG["do_sample"], num_beams=GENERATION_CONFIG["num_beams"], max_length=max_size, top_k=GENERATION_CONFIG["top_k"], top_p=GENERATION_CONFIG["top_p"])
        # self.encoded_text = self.model.generate(**self.input_tokenized, min_length=sent_len, early_stopping=True, max_length=max_size)

    def decode_text(self, input_text):
        if len(self.encoded_text) <= 1:
            self.decoded_text = self.tokenizer.decode(self.encoded_text[0], skip_special_tokens=True)
        else:
            if self.conditional_generation:
                Sbert = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
                temp = []
                for text in self.encoded_text:
                    generated_output = self.tokenizer.decode(text, skip_special_tokens=True)
                    emb1 = Sbert.encode(input_text, convert_to_tensor=True)
                    emb2 = Sbert.encode(generated_output, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(emb1, emb2)
                    if cosine_scores > GENERATION_CONFIG["text_similarity"]:# and is_formal:
                        # print(generated_output)
                        temp.append(generated_output)
                self.decoded_text = temp
                if len(self.decoded_text) < 1:
                    self.decoded_text = "No output for this sentence! Changing the condition might help..."
            else:
                temp = []
                for text in self.encoded_text:
                    temp.append(self.tokenizer.decode(text, skip_special_tokens=True))
                self.decoded_text = temp
