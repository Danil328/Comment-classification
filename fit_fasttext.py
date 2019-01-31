import numpy as np
import pandas as pd
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import tqdm
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
# from bpemb import BPEmb

PATH_TO_DATA = 'data/'

stop_words = stopwords.words('russian')
stop_words.append('http')
table = str.maketrans('', '', string.punctuation)
symbols = (u"abvgdeejzijklmnoprstufhzcss_y_eua",
           u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
transliterate = {ord(a): ord(b) for a, b in zip(*symbols)}
table.update(transliterate)

# tokenizer = BPEmb(lang='ru', vs=1000000, dim=100)

def clean_and_tokenize_text(text):
    tokens = word_tokenize(text)
    #tokens = tokenizer.encode(text)
    words = [word.lower().translate(table) for word in tokens if
             word.isalpha() and word not in stop_words and len(word) < 30]
    if len(words) == 0:
        words = ['пусто']
    return words

def read_data(path):
    print("Loading data...")
    data = pd.read_csv(os.path.join(path, 'union_df.csv'))
    return data

# class Generator():
#     def __init__(self, data):
#         self.data = data
#
#     def __iter__(self):
#         for i, row in enumerate(self.data):
#             yield clean_and_tokenize_text(row)
#
#
# class W2VTrainer:
#     def __init__(self, data):
#         self.generator = Generator(data)
#
#     def train_and_save(self, path_to_save_w2v):
#         model_w2v = Word2Vec(min_count=5, workers=8)
#         model_w2v.build_vocab(self.generator)
#
#         model_w2v.train(self.generator, total_examples=model_w2v.corpus_count, epochs=10)
#
#         model_w2v.save(path_to_save_w2v)

if __name__ == "__main__":
    data = read_data(PATH_TO_DATA)
    tqdm.tqdm.pandas()
    data['tokens'] = data['TEXT'].progress_apply(clean_and_tokenize_text)

    model = FastText(data['tokens'].values.tolist(), size=100, window=5, min_count=5, workers=8, iter=10)
    # model.wv.vocab
    model.wv.most_similar('привет')