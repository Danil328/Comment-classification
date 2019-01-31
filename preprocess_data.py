import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from config import PATH_TO_DATA, LENGTH_W2V_VECTOR, REPLACE_SMILE
import gensim.downloader as api
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
import pickle
import emot

class TextPreprocessing():
    def __init__(self, path_to_w2v_model='w2v_models/ru/ru.bin'):
        self.stop_words = stopwords.words('russian')
        self.stop_words.append('http')
        self.w2v_model = FastText.load_fasttext_format(path_to_w2v_model)
        self.table = str.maketrans('', '', string.punctuation)

        symbols = (u"abvgdeejzijklmnoprstufhzcss_y_eua",
                   u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        transliterate = {ord(a): ord(b) for a, b in zip(*symbols)}
        self.table.update(transliterate)

    def clean_and_tokenize_text(self, text):
        # if REPLACE_SMILE:
        #     smiles = emot.emoticons(text)
        #     if smiles['flag']:
        #         last_location = [-10, -10]
        #         last_mean = ''
        #         for value, mean, location in zip(smiles['value'], smiles['mean'], smiles['location']):
        #             if location[0] - last_location[1] > 10 and mean == last_mean:
        #                 text = text.replace(value, mean, 1)
        #                 last_location = location
        #                 last_mean = mean
        #             elif mean != last_mean:
        #                 text = text.replace(value, mean, 1)
        #                 last_location = location
        #                 last_mean = mean

        tokens = word_tokenize(text)
        words = [word.lower().translate(self.table) for word in tokens if
                 word.isalpha() and word not in self.stop_words and len(word) < 30]
        if len(words) == 0:
            words = ['пусто']
        return np.array(words)

    def get_w2v_vector(self, tokens):
        vector = []
        for token in tokens:
            try:
                vector.append(self.w2v_model.wv.get_vector(token))
            except Exception:
                pass
                print(f'Error token - {token}')
            if len(vector) == 0:
                vector.append(self.w2v_model.wv.get_vector('пусто'))
        return vector

    def preprocessing(self, text):
        return self.get_w2v_vector(self.clean_and_tokenize_text(text))


if __name__ == '__main__':
    print("Loading data...")
    data = pd.read_csv(os.path.join(PATH_TO_DATA, 'union_df.csv'))
    data['result'] = data['result'].map({'clean': 0, 'spam': 1})

    print("Load class...")
    processor = TextPreprocessing()

    print('Cleaning and tokenize data...')
    tqdm.pandas()
    # data['clean_comment_tokens'] = data['TEXT'].progress_apply(processor.clean_and_tokenize_text)

    train, val = train_test_split(data, test_size=0.15, random_state=17, stratify=data.result.values)

    vectorized_train_data = train['TEXT'].progress_apply(processor.preprocessing).values
    vectorized_val_data = val['TEXT'].progress_apply(processor.preprocessing).values

    print('Saving preprocess data...')
    with open(os.path.join(PATH_TO_DATA, 'X_train_w2v_v2.pickle'), 'wb') as f:
        pickle.dump(vectorized_train_data, f)

    with open(os.path.join(PATH_TO_DATA, 'y_train_v2.pickle'), 'wb') as f:
        pickle.dump(train.result.values, f)

    with open(os.path.join(PATH_TO_DATA, 'X_val_w2v_v2.pickle'), 'wb') as f:
        pickle.dump(vectorized_val_data, f)

    with open(os.path.join(PATH_TO_DATA, 'y_val_v2.pickle'), 'wb') as f:
        pickle.dump(val.result.values, f)

# v2 without http in tokens