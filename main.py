from text_processor import load_data, read_txt
import pandas as pd
from config import TRAIN_CONFIG
from paraphraser import Paraphraser
from style_transfer import StyleTransfer
from style_classifier import KMeansAlgorithm
import argparse
from datasets import Dataset, load_dataset, Features, Value, ClassLabel

parser = argparse.ArgumentParser(description='Run style transfer/paraphraser/classifier.')
parser.add_argument('task', type=str, help='Model to run, paraphraser, style transfer, or classifier')
parser.add_argument('mode', type=str, help='Mode to run the model, training or inference')
parser.add_argument('--input', type=str, help='Input text to transfer/paraphrase/classify', default='من میرم با بچه‌ها بازی کنم')

args = parser.parse_args()
# para = Paraphraser(num_outputs=5, conditional_generation=True)
st = StyleTransfer(num_outputs=5, conditional_generation=True)
km = KMeansAlgorithm(2, iter=500, n_init=100)
task_prefix = "paraphrase" if args.mode.lower() == "paraphrase" else "transfer"

if args.task.lower() == "train":
  print("Training Phase")
  df = pd.read_csv("data/"+args.input.lower()+'.txt', sep=",", header=None)
  df.columns = ["sentence", "labels"]
  print("Selected task prefix: "+task_prefix)
  df['sentence'] = '{'+task_prefix+'}: ' + df['sentence'].astype(str) + " </s>"
  df['labels'] = df['labels'].astype(str) + " </s>"
  dataset = Dataset.from_pandas(df)
  st.train(dataset)
elif args.task.lower() == "test":
  print("Testing Phase")
  df = pd.read_csv("data/"+args.input.lower()+'.txt', sep=",", header=None)
  df.columns = ["sentence"]
  df['sentence'] = '{'+task_prefix+':} ' + df['sentence'].astype(str) + " </s>"
  dataset = Dataset.from_pandas(df)
  st.batch_formalize_text(dataset)
else:
  print("Probably one or more arguments are missing.")