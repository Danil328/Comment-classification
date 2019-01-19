import os
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import PATH_TO_DATA, CLASSES

stop_words = stopwords.words('english')
porter = PorterStemmer()

def clean_and_tokenize_text(text):
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    words = [word.lower().translate(table) for word in tokens if word.isalpha() and word not in stop_words and len(word)<30]
    if len(words) == 0:
        words = ['nan']
    return np.array(words)

if __name__ == '__main__':
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv'))
    test_data = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

    X_train, X_val = train_test_split(train_data, test_size=0.2, random_state=17, stratify=train_data[CLASSES].sum(axis=1))

    tf_vect = TfidfVectorizer(min_df=2, tokenizer=clean_and_tokenize_text, preprocessor=None, stop_words=stop_words)
    train_data_features = tf_vect.fit_transform(X_train['comment_text'])
    val_data_features = tf_vect.transform(X_val['comment_text'])
    test_data_features = tf_vect.transform(test_data['comment_text'])

    logreg = LogisticRegression(n_jobs=1, C=0.5)
    score = {}
    logreg_dict = {}
    for clas in tqdm(CLASSES):
        logreg_dict[clas] = logreg.fit(train_data_features, X_train[clas])
        pred = logreg.predict_proba(val_data_features)[:, 1]
        score[clas] = roc_auc_score(X_val[clas], pred)
    print(f'Mean ROC_AUC score = {np.mean(list(score.values()))}')

    predict_test = np.zeros((153164, 6))
    for i, lr in tqdm(enumerate(CLASSES)):
        predict_test[:, i] = logreg_dict[clas].predict_proba(test_data_features)[:, 1]

    predict_df = pd.DataFrame(columns=['id'] + CLASSES)
    predict_df['id'] = test_data.id
    predict_df[CLASSES] = predict_test
    predict_df.to_csv(os.path.join('submissions', 'sub_baseline.csv'), index=False)  # 0.94 on private LB

