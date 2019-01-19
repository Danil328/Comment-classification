import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from config import PATH_TO_DATA, CLASSES, LENGTH_W2V_VECTOR, USE_IDF, REPLACE_SMILE
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import emot

stop_words = stopwords.words('english')
porter = PorterStemmer()

# IF we can see several emoticons, we pass identical emoticons with step = 10
def clean_and_tokenize_text(text):
    if REPLACE_SMILE:
        smiles = emot.emoticons(text)
        if smiles['flag']:
            last_location = [-10, -10]
            last_mean = ''
            for value, mean, location in zip(smiles['value'], smiles['mean'], smiles['location']):
                if location[0] - last_location[1] > 10 and mean == last_mean:
                    text = text.replace(value, mean, 1)
                    last_location = location
                    last_mean = mean
                elif mean != last_mean:
                    text = text.replace(value, mean, 1)
                    last_location = location
                    last_mean = mean

    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    words = [word.lower().translate(table) for word in tokens if word.isalpha() and word not in stop_words and len(word)<30]
    if len(words) == 0:
        words = ['nan']
    return np.array(words)

# def clean_and_tokenize_text(text):
#     if REPLACE_SMILE:
#         smiles = emot.emoticons(text)
#         if smiles['flag']:
#             for value, mean in zip(smiles['value'], smiles['mean']):
#                 text = text.replace(value, mean, 1)
#     tokens = word_tokenize(text)
#     table = str.maketrans('', '', string.punctuation)
#     words = [word.lower().translate(table) for word in tokens if word.isalpha() and word not in stop_words and len(word)<30]
#     if len(words) == 0:
#         words = ['nan']
#     return np.array(words)

def get_w2v_vector(tokens):
    vector = []
    for token in tokens:
        try:
            if USE_IDF:
                vector.append(w2v_model.get_vector(token) * weights_dict[token]/len(tokens))
            else:
                vector.append(w2v_model.get_vector(token))
        except Exception:
            pass
            #print(f'Error token - {token}')
        if len(vector) == 0:
            vector.append(w2v_model.get_vector('nan'))
    return vector

def get_idf_dict(train_text):
    print("Calculating idf...")
    tf_vect = TfidfVectorizer(min_df=2, tokenizer=clean_and_tokenize_text, preprocessor=None, stop_words=stop_words, smooth_idf=False)
    tf_vect.fit(train_data)
    weights_dict = dict(zip(tf_vect.get_feature_names(), tf_vect.idf_))
    return weights_dict

if __name__ == '__main__':
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv'))
    test_data = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

    # TF-IDF weights
    if USE_IDF:
        weights_dict = get_idf_dict(train_data['comment_text'])

    print('Cleaning and tokenize data...')
    tqdm.pandas()
    train_data['clean_comment_tokens'] = train_data['comment_text'].progress_apply(clean_and_tokenize_text)
    test_data['clean_comment_tokens'] = test_data['comment_text'].progress_apply(clean_and_tokenize_text)
    target = train_data[CLASSES].values

    # TODO Load w2v
    print("Loading w2v...")
    # info = api.info()  # show info about available models/datasets
    w2v_model = api.load(f"glove-twitter-{LENGTH_W2V_VECTOR}")  # download the model and return as object ready for use

    train = train_data['clean_comment_tokens'].progress_apply(get_w2v_vector).values
    test = test_data['clean_comment_tokens'].progress_apply(get_w2v_vector).values

    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=17, stratify=target.sum(axis=1))

    print('Saving preprocess data...')
    with open(os.path.join(PATH_TO_DATA, 'X_train_w2v.pickle'), 'wb') as f:
        pickle.dump(X_train, f)

    with open(os.path.join(PATH_TO_DATA, 'y_train.pickle'), 'wb') as f:
        pickle.dump(y_train, f)

    with open(os.path.join(PATH_TO_DATA, 'X_val_w2v.pickle'), 'wb') as f:
        pickle.dump(X_val, f)

    with open(os.path.join(PATH_TO_DATA, 'y_val.pickle'), 'wb') as f:
        pickle.dump(y_val, f)

    with open(os.path.join(PATH_TO_DATA, 'X_test_w2v.pickle'), 'wb') as f:
        pickle.dump(dict(zip(test_data['id'].values, test)), f)
