import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from config import PATH_TO_DATA, CLASSES, LENGTH_W2V_VECTOR
import gensim.downloader as api
from sklearn.model_selection import train_test_split

stop_words = stopwords.words('english')
porter = PorterStemmer()

def clean_and_tokenize_text(text):
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    words = [word.lower().translate(table) for word in tokens if word.isalpha() and word not in stop_words and len(word)<30]
    if len(words) == 0:
        words = ['nan']
    return np.array(words)

def get_w2v_vector(tokens):
    vector = []
    for token in tokens:
        try:
            vector.append(w2v_model.get_vector(token))
        except Exception:
            pass
            #print(f'Error token - {token}')
        if len(vector) == 0:
            vector.append(w2v_model.get_vector('nan'))
    return vector

if __name__ == '__main__':
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv'))
    test_data = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

    print('Cleaning and tokenize data...')
    tqdm.pandas()
    train_data['clean_comment_tokens'] = train_data['comment_text'].progress_apply(clean_and_tokenize_text)
    test_data['clean_comment_tokens'] = test_data['comment_text'].progress_apply(clean_and_tokenize_text)

    # TODO Load w2v
    info = api.info()  # show info about available models/datasets
    w2v_model = api.load(f"glove-twitter-{LENGTH_W2V_VECTOR}")  # download the model and return as object ready for use

    train = train_data['clean_comment_tokens'].progress_apply(get_w2v_vector).values
    test = test_data['clean_comment_tokens'].progress_apply(get_w2v_vector).values
    target = train_data[CLASSES].values

    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=17, stratify=target.sum(axis=1))

    print('Saving preprocess data...')
    np.save(os.path.join(PATH_TO_DATA, 'X_train_w2v'), X_train)
    np.save(os.path.join(PATH_TO_DATA, 'y_train'), y_train)

    np.save(os.path.join(PATH_TO_DATA, 'X_val_w2v'), X_val)
    np.save(os.path.join(PATH_TO_DATA, 'y_val'), y_val)

    np.save(os.path.join(PATH_TO_DATA, 'X_test_w2v'), dict(zip(test_data['id'].values, test)))