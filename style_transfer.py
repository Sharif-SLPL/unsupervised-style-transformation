from __future__ import unicode_literals
import torch
from transformers import (
    Adafactor,
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq
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
from torch.utils.data import DataLoader
from transformers import get_scheduler

pd.options.display.max_colwidth = 200
# km = KMeansAlgorithm(2, iter=500, n_init=100)

class StyleTransfer:
    def __init__(self, num_outputs=1, conditional_generation=False, device="cuda"):
        self.device = torch.device(device)
        self.st_model_path = BASE_CONFIG["base_model_path"] # Or use base model
        # Model's path if exist to be used for inference. Save the trained model in this path
        self.local_st_model_path = BASE_CONFIG["trained_model_path"]
        if os.path.exists(self.local_st_model_path):
            print('Loading model from local checkpoint')
            self.st_model_path = self.local_st_model_path
        self.model = T5ForConditionalGeneration.from_pretrained(self.st_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.st_model_path)
        self.model.to(self.device)
        self.conditional_generation = conditional_generation
        self.num_outputs = num_outputs
        self.loss_history = []

    def train(self, dataset):
        max_input_length = 64
        max_target_length = 64
        num_epochs = TRAIN_CONFIG["epochs"]

        def preprocess_function(examples):
            inputs = [examples['sentence'] for ex in examples]
            inputs = inputs[0]
            targets = [examples['labels'] for ex in examples]
            targets = targets[0]
            model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True)
            # Tokenize labels 
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=max_target_length, truncation=True, padding=True)
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets.set_format("torch")
        train_dataloader = DataLoader(
            tokenized_datasets, shuffle=True, batch_size=TRAIN_CONFIG["batch_size"], collate_fn=data_collator
        )
        # TODO: Remove
        # qq = [0.5, 0.75, 0.9, 0.99, 1]
        # print(pd.Series(len(self.tokenizer(get_batch(data, mult=3)[0], padding=True)['input_ids'][0]) for _ in range(1000)).quantile(qq))
        # print(pd.Series(len(self.tokenizer(get_batch(data, mult=3)[1], padding=True)['input_ids'][0]) for _ in range(1000)).quantile(qq))
        # optimizer = torch.optim.Adam(params = [p for p in self.model.parameters() if p.requires_grad], lr=1e-5)
        # optimizer.param_groups[0]['lr'] = 1e-5
        optimizer = Adafactor(
            self.model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"], #1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=TRAIN_CONFIG["weight_decay"],
            relative_step=TRAIN_CONFIG["relative_step"],
            # scale_parameter=False,
            warmup_init=TRAIN_CONFIG["warmup_init"],
        )
        self.model.train()
        # mult = 4
        # batch_size = mult * 8
        # max_len = 128 
        # accumulation_steps = 32
        # save_steps = 2000
        # window = 2000
        # ewm = 0
        # errors = 0
        # tq = trange(int(1000 * epochs / mult)) #1_000_000
        cleanup()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        num_training_steps = num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))

        batches_loss = []
        for epoch in range(num_epochs):
          for batch in train_dataloader:
              batch = {k: v.to(device) for k, v in batch.items()}
              outputs = self.model(**batch)
              loss = outputs.loss
              loss.backward()
              progress_bar.set_description(f'loss: {loss}')
              batches_loss.append(loss.detach().cpu().numpy())
              optimizer.step()
              # lr_scheduler.step()
              optimizer.zero_grad()
              progress_bar.update(1)
          try:
            print("Epoch Loss: ", sum(batches_loss) / len(batches_loss))
          except:
            print("Failed to epoch batch loss")
        print("Saving...")
        self.model.save_pretrained(self.local_st_model_path)
        self.tokenizer.save_pretrained(self.local_st_model_path)
        draw_loss_plot(batches_loss)

        return
        # for i in tq:
        #     xx, yy = get_batch(data, mult=mult)
        #     try:
        #         # Swap x & y. Actually should be batched like this from the start
        #         y = self.tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(self.device)
        #         x = self.tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(self.device)

        #         y.input_ids[y.input_ids==0] = -100  #to make sure we have correct labels for T5 text generation
        #         loss = self.model(
        #             input_ids=x.input_ids,
        #             attention_mask=x.attention_mask,
        #             labels=y.input_ids,
        #             decoder_attention_mask=y.attention_mask,
        #             return_dict=True
        #         ).loss
        #         loss.backward()
                
        #     except RuntimeError as e:
        #         print(e)
        #         errors += 1
        #         loss = None
        #         cleanup()
        #         continue

        #     w = 1 / min(i+1, window)
        #     ewm = ewm * (1-w) + loss.item() * w
        #     tq.set_description(f'loss: {ewm}')

        #     # Save history when an epoch is completed
        #     if i > 0 and i % (len(data)/mult) == 0:
        #       self.loss_history.append((i/len(data), ewm))
        #     if i%10 ==0:# redundant maybe but save loss every 10 steps TODO: revise loss_history
        #         self.loss_history.append(loss.item())

        #     if i % accumulation_steps == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         cleanup()
            
        #     if i % window == 0 and i > 0:
        #         print(ewm, errors)
        #         errors = 0
        #         cleanup()
        #         # optimizer.param_groups[0]['lr'] *= 0.999
        #     if i % save_steps == 0 and i > 0:
        #         self.save_model()
        #         print('saving...', i, optimizer.param_groups[0]['lr'])
        #         # self.loss_history.append((round(i/len(data)), ewm))
        #         draw_loss_plot(self.loss_history)
        #     # Early stopping:
        #     if ewm < TRAIN_CONFIG["loss_threshold"]:
        #         # self.loss_history.append((round(i/len(data)), ewm))
        #         draw_loss_plot(self.loss_history)
        #         self.save_model()
        #         print('saving...', i, optimizer.param_groups[0]['lr'])
        #         print("Stopping Training...")
        #         break  # early stop criterion is met, we can stop now
        
    def save_model(self):
        self.model.save_pretrained(self.local_st_model_path)
        self.tokenizer.save_pretrained(self.local_st_model_path)

    def formalize_text(self, text):
        self.model.eval()
        self.encode_text(text)
        self.decode_text(text)
        print(self.decoded_text)

    def batch_formalize_text(self, dataset):
        self.model.eval()
        def preprocess_function(examples):
            inputs = [examples['sentence'] for ex in examples]
            inputs = inputs[0]
            model_inputs = self.tokenizer(inputs, max_length=32, truncation=True, padding=True)

            return model_inputs

        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets.set_format("torch")
        test_dataloader = DataLoader(
            tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
        )
        with open("output.csv", "w") as f:
          f.write("text,text2\n")
          for i, batch in tqdm(enumerate(tokenized_datasets)):
            original_text = dataset[i]["sentence"]
            original_text = original_text.replace("1 paraphrase: ", "")
            original_text = original_text.replace("{transfer:} 1 ", "")
            original_text = original_text.replace(" </s>", "")
            original_text_size = len(original_text.split())
            batch = {k: torch.unsqueeze(v.to(self.model.device), 0) for k, v in batch.items()}
            batch = batch['input_ids']
            if GENERATION_CONFIG["decode_method"] == "beam":
              outputs = self.model.generate(batch, temperature=GENERATION_CONFIG["temperature"], no_repeat_ngram_size=2, min_length=len(dataset[i]), early_stopping=True, num_return_sequences=self.num_outputs, do_sample=GENERATION_CONFIG["do_sample"], num_beams=GENERATION_CONFIG["num_beams"], max_length=32, top_k=GENERATION_CONFIG["top_k"], top_p=GENERATION_CONFIG["top_p"])
            else:
              outputs = self.model.generate(batch, min_length=original_text_size, max_length=original_text_size)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(dataset[i])
            # print(decoded)
            # print("-"*100)
            f.write(decoded+","+original_text+"\n")
            
        # data = read_txt(dataset)
        # for text in data:
        #     self.encode_text(text)
        #     self.decode_text(text)
        #     print(text, self.decoded_text, "\n")

    def encode_text(self, input_text):
        self.input_tokenized = self.tokenizer(input_text, return_tensors='pt', padding=True).to(self.model.device)
        print(self.input_tokenized)
        sent_len = self.input_tokenized.input_ids.shape[1]
        max_size = int(sent_len)
        #TODO: Add temperature=0.9, top_p=0.6, top_k=50 with default values
        # self.encoded_text = self.model.generate(**self.input_tokenized, encoder_no_repeat_ngram_size=4, no_repeat_ngram_size=4, min_length=sent_len, num_return_sequences=self.num_outputs, do_sample=GENERATION_CONFIG["do_sample"], num_beams=GENERATION_CONFIG["num_beams"], max_length=max_size)
        # self.encoded_text = self.model.generate(**self.input_tokenized, encoder_no_repeat_ngram_size=2, temperature=GENERATION_CONFIG["temperature"], no_repeat_ngram_size=2, min_length=sent_len, early_stopping=True, num_return_sequences=self.num_outputs, do_sample=GENERATION_CONFIG["do_sample"], num_beams=GENERATION_CONFIG["num_beams"], max_length=max_size, top_k=GENERATION_CONFIG["top_k"], top_p=GENERATION_CONFIG["top_p"])
        self.encoded_text = self.model.generate(**self.input_tokenized, min_length=sent_len, early_stopping=True, max_length=max_size)

    def decode_text(self, input_text):
        print(self.encoded_text)
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
                    is_formal = km.predict_instance([generated_output])
                    if cosine_scores > GENERATION_CONFIG["text_similarity"]:# and is_formal:
                        # print(generated_output)
                        temp.append((generated_output, is_formal))
                self.decoded_text = temp
                if len(self.decoded_text) < 1:
                    self.decoded_text = "No output for this sentence! Changing the condition might help..."
            else:
                temp = []
                for text in self.encoded_text:
                    temp.append(self.tokenizer.decode(text, skip_special_tokens=True))
                self.decoded_text = temp
