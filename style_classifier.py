from __future__ import unicode_literals
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from collections import Counter
from hazm import *
from text_processor import read_txt
from utils import save_model, load_model
import seaborn as sns
import numpy as np
import pickle
import sys

# TODO: Stylometry project. Check it out for ideas
# !git clone git@github.com:jpotts18/stylometry.git

np.set_printoptions(threshold=sys.maxsize)

def tfidf_vectorizer_fit(input_data, analyzer="word"):
  tfidfvectorizer = TfidfVectorizer(analyzer=analyzer)
  tfidfvectorizer.fit(input_data)
  save_model(tfidfvectorizer, "tfidf.pkl")

  return tfidfvectorizer

def tfidf_vectorizer_fit_transform(input_data, vectorizer=None, vectorizer_path=None):
  if vectorizer:
    vectorizer = vectorizer
  elif vectorizer_path:
    vec = load_model(vectorizer_path)
    vectorizer = TfidfVectorizer(vocabulary = vec.vocabulary_)
  else:
    raise ValueError('No vectorizer object/path was given')

  tfidf_tokens = vectorizer.get_feature_names_out()
  term_vectors  = vectorizer.fit_transform(input_data)

  return term_vectors

def tfidf_fit(input_data, analyzer="word"):
  """
  Returns TFIDF matrix of input data.
  analyzer {‘word’, ‘char’, ‘char_wb’}
  """
  tfidf = TfidfTransformer(norm='l2')
  tfidf.fit(input_data)

  return tfidf

def tfidf_transform(transformer, input_vectors):
  tf_idf_matrix = transformer.transform(input_vectors)
  
  return tf_idf_matrix

# %matplotlib inline
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def scatter(x, labels):
    n_label_colors = len(set(labels))
    palette = np.array(sns.color_palette("hls", n_label_colors))
    f = plt.figure(figsize=(24, 24))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[labels.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    # Labels for each cluster.
    txts = []
    for i in range(n_label_colors):
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

class KMeansAlgorithm:
  def __init__(self, k, seed=1, iter=300, n_init=100):
    self.k = k
    self.iter = iter
    self.n_init = n_init
    self.seed = seed
    self.model = KMeans(n_clusters=self.k, init='k-means++', max_iter=self.iter, n_init=self.n_init)
  
  def model_fit(self, input_data):
    self.model.fit(input_data)
    
    return self.model

  def predict_style(self, input_data):
    predicted_style = self.model.predict(input_data)
    
    return predicted_style

  def train(self, input, save_plot=True):
    tfidf_vectorizer = tfidf_vectorizer_fit(input)
    tfidf_vectors = tfidf_vectorizer_fit_transform(input, vectorizer_path="tfidf")
    tfidf_transformer = tfidf_fit(tfidf_vectors)
    tf_idf_matrix = tfidf_transform(tfidf_transformer, tfidf_vectors)
    save_model(tfidf_transformer, "tfidf_model.pkl")

    # TSNE is better for presentation rather than feature extraction
    # digits_proj = TSNE(random_state=self.seed)
    # digits_proj.fit(tf_idf_matrix)
    # digits_proj.transform(tf_idf_matrix)

    svd = TruncatedSVD(n_components=2, random_state=42)
    digits_proj = svd.fit(tf_idf_matrix) 
    digits_proj = svd.transform(tf_idf_matrix) 
    save_model(svd, "svd_model.pkl")

    kmeans = self.model_fit(digits_proj)
    pred = kmeans.predict(digits_proj)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    save_model(self, "kmeans_model.pkl")

    print(Counter(pred))
    # print(order_centroids)

    Y = kmeans.labels_

    if save_plot:
      scatter(digits_proj, Y)
      plt.savefig('outputs/tfidf_{}_cluster.png'.format("svd"), dpi=120) # svd, pca, tsne
      print("Saved the plot!")
    # for i in range(len(pred)):
    #   print(train[i], "=>", pred[i])
  
  def predict_instance(self, input):
    # Predicts a single input
    tfidf_transformer = load_model('tfidf_model')
    tfidf_vectors = tfidf_vectorizer_fit_transform(input, vectorizer_path="tfidf")
    tf_idf_matrix = tfidf_transform(tfidf_transformer, tfidf_vectors)
    svd = load_model('svd_model')
    svd_matrix = svd.transform(tf_idf_matrix)
    KM = load_model('kmeans_model')
    pred_style = list(KM.predict_style(svd_matrix))[0]

    # Might need to change between 0 & 1
    if pred_style == 1:
      print(input[0], " - Stlye: Informal")
    if pred_style == 0:
      print(input[0], " - Stlye: Formal")

    # test_raw_data = pd.read_csv('data.txt', sep=',', header=None, on_bad_lines='skip')

    # test_data = []
    # for i in range(len(test_raw_data)):
    #   lxd = 0
    #   for j in range(2):
    #     line = test_raw_data[j][i]
    #     tokenized = word_tokenize(line)
    #     lxd += lexical_diversity(tokenized)
    #   if lxd >= 1.7:
    #     test_data.append(test_raw_data[1][i])

  def batch_predict(self, path):
    # Predicts a set of sentences in a file
    input = read_txt(path, True)
    tfidf_transformer = load_model('tfidf_model')
    svd = load_model('svd_model')
    km = load_model('kmeans_model')
    tfidf_vectors = tfidf_vectorizer_fit_transform(input, vectorizer_path="tfidf")
    tf_idf_matrix = tfidf_transform(tfidf_transformer, tfidf_vectors)
    svd_matrix = svd.transform(tf_idf_matrix)
    pred_style = list(km.predict_style(svd_matrix))
    for i, pred in enumerate(pred_style):
      if pred == 0:
        print(input[i], " - Stlye: Informal")
      if pred == 1:
        print(input[i], " - Stlye: Formal")
# Clustering data with KMeans, reduced dim with t-SNE / SVD, and plotting data






