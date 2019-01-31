import torch
import numpy as np
import pandas as pd
from DAN_with_attention import load_model, MODEL_NAME, LENGTH_W2V_VECTOR
import os
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('..')
from preprocess_data import TextPreprocessing


class StatusAnalizer():
    def __init__(self, path_to_w2v_model='../w2v_models/ru/ru.bin', path_to_classifier='../pretrained models'):
        self.preprocessing = TextPreprocessing(path_to_w2v_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classifier = load_model(os.path.join(path_to_classifier, MODEL_NAME))

    def predict(self, text):
        tokens = self.preprocessing.clean_and_tokenize_text(text)
        wv_vectors = self.preprocessing.get_w2v_vector(tokens)
        wv_vectors_torch = torch.from_numpy(np.squeeze(wv_vectors).reshape(-1, LENGTH_W2V_VECTOR)).unsqueeze(0).to(self.device)
        output, attention_score = self.classifier(wv_vectors_torch)
        return output.cpu().item(), tokens, attention_score.cpu().detach().numpy().squeeze()


def view_attention():
    print("Loading data...")
    data = pd.read_csv(os.path.join('../data', 'union_df.csv'))
    data['result'] = data['result'].map({'clean': 0, 'spam': 1})
    train, val = train_test_split(data, test_size=0.15, random_state=17, stratify=data.result.values)
    status_analizer = StatusAnalizer()

    predictions = []
    list_tokens = []
    list_attention = []

    for text in tqdm(val['TEXT'].values):
        prediction, tokens, attention_score = status_analizer.predict(text)
        list_tokens.append(tokens.tolist())
        predictions.append(prediction)
        list_attention.append(attention_score.tolist())

    val.columns = ['TEXT', 'label']
    val['text'] = list_tokens
    val['prediction'] = predictions
    val['attention'] = list_attention
    val['id'] = list(map(lambda x: str(x), val.index))

    val[val.label == 1].drop(columns=['TEXT'])[:100].to_json('val.json', orient='index', force_ascii=False)

    return val

if __name__ == '__main__':
    #val = view_attention()

    status_analizer = StatusAnalizer()
    predict, tokens, attention_matrix = status_analizer.predict('Анонимные знакомства для секса в вашем городе http://c.twnt.ru/s5HV/3333 5')
