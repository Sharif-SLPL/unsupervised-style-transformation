from text_processor import load_data, read_txt
import pandas as pd
from config import TRAIN_CONFIG
from paraphraser import Paraphraser
from style_transfer import StyleTransfer
from style_classifier import KMeansAlgorithm
import argparse

parser = argparse.ArgumentParser(description='Run style transfer/paraphraser/classifier.')
parser.add_argument('task', type=str, help='Model to run, paraphraser, style transfer, or classifier')
parser.add_argument('mode', type=str, help='Mode to run the model, training or inference')
parser.add_argument('--input', type=str, help='Input text to transfer/paraphrase/classify', default='من میرم با بچه‌ها بازی کنم')

args = parser.parse_args()
para = Paraphraser(num_outputs=5, conditional_generation=True)
st = StyleTransfer(num_outputs=5, conditional_generation=True)
km = KMeansAlgorithm(2, iter=500, n_init=100)

if args.task.lower() == "transfer":
  print("Selected Task: Style Transfer")
  if args.mode.lower() == "train":
    print("Selected Mode: Training")
    raw_data = pd.read_csv(TRAIN_CONFIG["paraphrase_dataset_path"], sep=',', header=None, on_bad_lines='skip')
    data = load_data(raw_data)
    st.train(data)
  elif args.mode.lower() == "test":
    print("Selected Mode: Testing")
    if args.input.endswith(".txt"):
      st.batch_formalize_text("data/"+args.input) #digi.txt
    else:
      st.formalize_text(args.input)
if args.task.lower() == "paraphrase":
  print("Selected Task: Paraphrasing")
  if args.mode.lower() == "train":
    print("Selected Mode: Training")
    raw_data = pd.read_csv(TRAIN_CONFIG["paraphrase_dataset_path"], sep=',', header=None, on_bad_lines='skip')
    data = load_data(raw_data)
    para.train(data)
  elif args.mode.lower() == "test":
    print("Selected Mode: Testing")
    if args.input.endswith(".txt"):
      para.batch_paraphrase_text("data/"+args.input) #digi.txt
    else:
      para.paraphrase_text(args.input)
elif args.task.lower() == "classify":
  print("Selected Task: Style Classification")
  if args.mode.lower() == "train":
    print("Selected Mode: Training")
    formal_data = read_txt(TRAIN_CONFIG["formal_dataset_path"], True)
    informal_data = read_txt(TRAIN_CONFIG["informal_dataset_path"], True)
    train_data = formal_data[:TRAIN_CONFIG["dataset_limit"]] + informal_data[:TRAIN_CONFIG["dataset_limit"]]
    km.train(train_data)
  elif args.mode.lower() == "test":
    print("Selected Mode: Testing")
    if args.input.endswith(".txt"):
      km.batch_predict("data/"+args.input) #digi.txt
    else:
      km.predict_instance([args.input])
else:
  print("Probably one or more arguments are missing.")
