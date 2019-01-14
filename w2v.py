import pandas as pd
import numpy as np
import nltk
import os

# TODO load a pre-trained model (gloVe word vectors):
import gensim.downloader as api

info = api.info()  # show info about available models/datasets
model = api.load("glove-twitter-25")  # download the model and return as object ready for use
model.most_similar("cat")

# TODO load a corpus and use it to train a Word2Vec model:
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

corpus = api.load('text8')  # download the corpus and return it opened as an iterable
model = Word2Vec(corpus)  # train a model from the corpus
model.most_similar("car")

# TODO Save model and load KeyedVectors
from gensim.models import KeyedVectors

model.wv.save(os.path.join('w2v', 'model.wv'))
wv = KeyedVectors.load(os.path.join('w2v', 'model.wv'), mmap='r')
vector = wv['computer']  # numpy vector of a word