import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import shutil
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve
import gc
import click
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tools.pytorch_lr_finder.lr_finder import LRFinder

import sys
sys.path.append('..')
from config import LENGTH_W2V_VECTOR, NUM_CLASSES, PATH_TO_DATA, EPOCHS, BATCH_SIZE, N_HEADS_ATTENTION, POSITION_ENCODING
from baseline import PlotConfusionMatrix, calculate_err
MODEL_NAME = f'DAN_with_attention_{LENGTH_W2V_VECTOR}_{NUM_CLASSES}_v2_three_head_ps'
PATH_TO_DATA = '../' + PATH_TO_DATA

class StatusDataset(Dataset):

    def __init__(self, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            with open(os.path.join(PATH_TO_DATA, 'X_train_w2v_v2.pickle'), 'rb') as f:
                self.X = pickle.load(f)
            with open(os.path.join(PATH_TO_DATA, 'y_train_v2.pickle'), 'rb') as f:
                self.y = pickle.load(f)
        elif self.mode == 'val':
            with open(os.path.join(PATH_TO_DATA, 'X_val_w2v_v2.pickle'), 'rb') as f:
                self.X = pickle.load(f)
            with open(os.path.join(PATH_TO_DATA, 'y_val_v2.pickle'), 'rb') as f:
                self.y = pickle.load(f)

    def __len__(self):
        if self.mode in ['train', 'val']:
            return self.X.shape[0]
        else:
            return len(self.X_keys)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            sample = {'vector': torch.from_numpy(np.squeeze(self.X[idx]).reshape(-1, LENGTH_W2V_VECTOR)),
                      'labels': torch.from_numpy(np.array(self.y[idx])).type(torch.float)}
            return sample['vector'], sample['labels']
        else:
            pass

def position_encoding_init(n_position, emb_dim=LENGTH_W2V_VECTOR):
    ''' Init the sinusoid position encoding table '''

    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out

class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
        scored_x = x_in * attention_score

        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)

        return condensed_x, attention_score


class Net(nn.Module):

    def __init__(self, n_heads=N_HEADS_ATTENTION):
        super(Net, self).__init__()

        self.n_heads = n_heads
        self.attention = list()
        for i in range(n_heads):
            self.attention.append(Attention(LENGTH_W2V_VECTOR))

        self.fc1 = nn.Linear(LENGTH_W2V_VECTOR*n_heads, int(LENGTH_W2V_VECTOR / 2))
        self.fc2 = nn.Linear(int(LENGTH_W2V_VECTOR / 2), int(LENGTH_W2V_VECTOR / 8))
        self.fc3 = nn.Linear(int(LENGTH_W2V_VECTOR / 8), NUM_CLASSES)

        if POSITION_ENCODING:
            self.position_dict = dict()
            for i in range(1, 250):
                self.position_dict[i] = position_encoding_init(i)

    def get_position_matrix(self, n_tokens):
        if n_tokens < 250:
            return self.position_dict[n_tokens]
        else:
            return position_encoding_init(n_tokens)

    def forward(self, x):
        if POSITION_ENCODING:
            x = x + self.get_position_matrix(x.size(1))

        attention_out = list()
        attention_score = list()
        for i in range(self.n_heads):
            a, s = self.attention[i](x)
            attention_out.append(a)
            attention_score.append(s)
        concat_x = torch.cat((attention_out), 1)

        concat_x = F.relu(self.fc1(concat_x))
        concat_x = F.relu(self.fc2(concat_x))
        concat_x = torch.sigmoid(self.fc3(concat_x))

        return concat_x, torch.cat(attention_score)


def load_model(path):
    model = Net()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def save_checkpoint(state, is_best, path, min_loss):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print(f"=> Validation Loss improve from {min_loss}")
        torch.save(state, path)  # save checkpoint
    else:
        print(f"=> Validation Loss did not improve from {min_loss}")


def train_model(lr, n_iter_stat=10000):
    print('Training model...')
    train_dataset = StatusDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    val_dataset = StatusDataset(mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    net.train()

    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2, verbose=True, min_lr=1e-8, threshold=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    try:
        shutil.rmtree('../logs')
    except Exception:
        pass
    os.mkdir('../logs')
    writer = SummaryWriter(log_dir='../logs')
    min_loss = 1
    is_best = False
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        print(f'Number epochs - {epoch + 1}')
        running_loss = 0.0
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            # optimizer.zero_grad()  # zero the parameter gradients

            outputs, attention_score = net(X)  # forward + backward + optimize
            loss = criterion(outputs.squeeze(), y.squeeze())
            loss.backward()
            # optimizer.step()

            # Batch accumulation
            if i % BATCH_SIZE == BATCH_SIZE-1:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            running_loss += loss.item()
            if i % n_iter_stat == n_iter_stat-1:  # print every n_iter_stat mini-batches
                writer.add_scalar(tag='loss', scalar_value=running_loss / n_iter_stat,
                                  global_step=i + train_loader.__len__() * epoch)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n_iter_stat))
                running_loss = 0.0

        val_loss, val_score, val_f1 = evaluate_model(net, val_loader)
        writer.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=epoch)
        writer.add_scalar(tag='val_roc_auc', scalar_value=val_score, global_step=epoch)
        writer.add_scalar(tag='val_f1', scalar_value=val_f1, global_step=epoch)
        if val_loss < min_loss:
            min_loss = val_loss
            is_best = True

        scheduler.step(val_loss)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'val_loss': val_loss}, is_best,
                        path=os.path.join('../pretrained models', MODEL_NAME), min_loss=min_loss)
        is_best = False

        optimizer.step()

    print('Finished Training')


def evaluate_model(net=None, val_loader=None, calculate_score=False):
    print('Evaluating model...')
    if net is None:
        net = load_model(os.path.join('../pretrained models', MODEL_NAME))
        val_dataset = StatusDataset(mode='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        net.eval()
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    y_true, predict = list(), list()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device).squeeze()
            outputs, attention_score = net(X)
            loss = criterion(outputs.squeeze(), y)
            running_loss += loss.item()
            y_true.append(y.squeeze().cpu().numpy().item())
            predict.append(outputs.squeeze().cpu().numpy().item())

    score = roc_auc_score(y_true, predict)
    running_loss /= val_loader.__len__()

    print(f'Evaluate loss = {running_loss}')
    print(f'ROC_AUC score = {score}')

    f1 = f1_score(y_true, np.round(predict))
    print(f'F1 score = {f1}')

    if calculate_score:
        eer, thres = calculate_err(y_true, predict)
        print(f'EER score - {eer}')

        y_test_legit = int(np.sum(list(map(lambda x: -x+1, y_true))))
        y_test_fraud = int(np.sum(y_true))

        PlotConfusionMatrix(y_true, np.round(predict), y_test_legit, y_test_fraud)

        # PRECISION RECALL CURVE
        plt.figure()
        plt.title('PRECISION RECALL CURVE')
        precision, recall, thresholds = precision_recall_curve(y_true, predict)
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')
        # plot the roc curve for the model
        plt.plot(recall, precision)
        # show the plot
        plt.show(block=False)

        # ROC AUC CURVE
        plt.figure()
        plt.title('ROC AUC CURVE')
        fpr, tpr, thresholds = roc_curve(y_true, predict)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr)
        plt.show(block=False)

        print(f'Class balance :\n0  {y_test_legit}\n1  {y_test_fraud}')
        plt.figure()
        sns.countplot(y_true)
        plt.show(block=False)

    return running_loss, score, f1

def find_lr():
    print('Finding best learning rate...')
    train_dataset = StatusDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    net.train()

    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    lr_finder = LRFinder(net, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=1000)
    lr_finder.plot()


def make_predict():
    print("Make prediction...")
    dataset = StatusDataset(mode='val')
    trainloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    net = load_model(os.path.join('../pretrained models', MODEL_NAME))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, data in enumerate(trainloader):
            X, ids = data
            X = X.to(device)
            outputs, attention_score = net(X)

    out_numpy = outputs.cpu().numpy()
    predict_df = pd.DataFrame(columns=['id', 'result'])
    predict_df['id'] = ids
    predict_df['result'] = np.squeeze(out_numpy)
    predict_df.to_csv(os.path.join('submissions', 'sub.csv'), index=False)
    print('Finished prediction')

@click.command()
@click.option('--action', default='train', help='train/eval/test/find_lr')
@click.option('--lr', default=1e-3)
def main(action, lr):
    if action == 'train':
        train_model(lr)
        gc.collect()
    elif action == 'eval':
        val_loss, val_score, val_f1 = evaluate_model(calculate_score=True)
        gc.collect()
    elif action == 'test':
        make_predict()
        gc.collect()
    elif action == 'find_lr': #prease comment return second varible in forward method in Net class
        find_lr()
        gc.collect()


if __name__ == "__main__":
    main()


""" DAN with one head attention
Evaluate loss = 0.01695090573578194
ROC_AUC score = 0.9902580810737787
F1 score = 0.9560059317844785
FAR - [0.         0.         0.         ... 0.99972003 0.99989501 1.        ]
FRR - [1.         0.99706745 0.99511241 ... 0.         0.         0.        ]
EER score - 0.024437927663734114
---Classification Report---
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     28574
         1.0       0.97      0.95      0.96      1023

   micro avg       1.00      1.00      1.00     29597
   macro avg       0.98      0.97      0.98     29597
weighted avg       1.00      1.00      1.00     29597

Class balance :
0  28574
1  1023
"""

""" DAN with two head attention
Evaluate loss = 0.014180506520915825
ROC_AUC score = 0.9924255766150157
F1 score = 0.9579256360078278
FAR - [0.         0.         0.         ... 0.99968503 0.99986001 1.        ]
FRR - [1.         0.99804497 0.99608993 ... 0.         0.         0.        ]
EER score - 0.022482893450635377
---Classification Report---
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     28574
         1.0       0.96      0.96      0.96      1023

   micro avg       1.00      1.00      1.00     29597
   macro avg       0.98      0.98      0.98     29597
weighted avg       1.00      1.00      1.00     29597

Class balance :
0  28574
1  1023
"""

"""
Evaluate loss = 0.01610769263971332
ROC_AUC score = 0.9941342473703272
F1 score = 0.9523809523809524
FAR - [0.         0.         0.         ... 0.99975502 0.99993001 1.        ]
FRR - [1.         0.99315738 0.97849462 ... 0.         0.         0.        ]
EER score - 0.020527859237428888
---Classification Report---
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     28574
         1.0       0.96      0.95      0.95      1023

   micro avg       1.00      1.00      1.00     29597
   macro avg       0.98      0.97      0.98     29597
weighted avg       1.00      1.00      1.00     29597

Class balance :
0  28574
1  1023
"""