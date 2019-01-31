import pickle
import os
import numpy as np
from config import PATH_TO_DATA, LENGTH_W2V_VECTOR
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, f1_score

from baseline import PlotConfusionMatrix, calculate_err
if __name__ == '__main__':

    with open(os.path.join(PATH_TO_DATA, 'X_train_w2v.pickle'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(PATH_TO_DATA, 'y_train.pickle'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(PATH_TO_DATA, 'X_val_w2v.pickle'), 'rb') as f:
        X_val = pickle.load(f)
    with open(os.path.join(PATH_TO_DATA, 'y_val.pickle'), 'rb') as f:
        y_val = pickle.load(f)

    X_train_array = np.empty((X_train.shape[0], LENGTH_W2V_VECTOR))
    X_val_array = np.empty((X_val.shape[0], LENGTH_W2V_VECTOR))

    for i, line in enumerate(tqdm(X_train)):
        X_train_array[i] = np.array(line).mean(axis=0)

    for i, line in enumerate(tqdm(X_val)):
        X_val_array[i] = np.array(line).mean(axis=0)

    lr = LogisticRegression()
    lr = RandomForestClassifier(max_depth=20, n_estimators=20)
    lr.fit(X_train_array, y_train)

    pred = lr.predict_proba(X_val_array)[:,1]

    roc_score = roc_auc_score(y_val, pred)
    print(f'ROC_AUC score = {roc_score}')

    f1 = f1_score(y_val, np.round(pred))
    print(f'F1 score = {f1}')

    eer, thres = calculate_err(y_val, pred)
    print(f'EER score - {eer}')

    y_test_legit = -(y_val - 1).sum()
    y_test_fraud = int(y_val.sum())

    PlotConfusionMatrix(y_val, np.round(pred), y_test_legit, y_test_fraud)