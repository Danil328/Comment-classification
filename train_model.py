import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import LENGTH_W2V_VECTOR, NUM_CLASSES, CLASSES, PATH_TO_DATA, EPOCHS, WORD_DROPOUT
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import shutil
from  sklearn.metrics import roc_auc_score
import gc
import click

MODEL_NAME = f'DAN_{LENGTH_W2V_VECTOR}_{NUM_CLASSES}'

def word_dropout(matrix):
    if not WORD_DROPOUT:
        return matrix
    nrows = matrix.shape[0]
    if nrows < 5:
        return matrix
    elif nrows >= 5 and nrows <= 10:
        skiprow = np.random.randint(0, nrows-1)
    else:
        skiprow = np.random.randint(0, nrows-1, 2)
    return np.delete(matrix, skiprow, axis=0)


class CommentDataset(Dataset):

    def __init__(self, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            self.X = np.load(os.path.join(PATH_TO_DATA, 'X_train_w2v.npy'))
            self.y = np.load(os.path.join(PATH_TO_DATA, 'y_train.npy'))
        elif self.mode == 'val':
            self.X = np.load(os.path.join(PATH_TO_DATA, 'X_val_w2v.npy'))
            self.y = np.load(os.path.join(PATH_TO_DATA, 'y_val.npy'))
        else:
            self.X = np.load(os.path.join(PATH_TO_DATA, 'X_test_w2v.npy')).item()
            self.X_keys = list(self.X.keys())
            self.X_values = list(self.X.values())

    def __len__(self):
        if self.mode in ['train', 'val']:
            return self.X.shape[0]
        else:
            return len(self.X_keys)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            sample = {'vector': torch.from_numpy(word_dropout(np.squeeze(self.X[idx]).reshape(-1, LENGTH_W2V_VECTOR)).mean(axis=0).reshape(1, -1)),
                      'labels': torch.from_numpy(self.y[idx]).reshape(1, -1).type(torch.float)}
            return sample['vector'], sample['labels']
        else:
            sample = {'vector': torch.from_numpy(np.squeeze(self.X_values[idx]).reshape(-1, LENGTH_W2V_VECTOR).mean(axis=0).reshape(1, -1)),
                      'id': self.X_keys[idx]}
            return sample['vector'], sample['id']


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(LENGTH_W2V_VECTOR, LENGTH_W2V_VECTOR)
        self.fc2 = nn.Linear(LENGTH_W2V_VECTOR, int(LENGTH_W2V_VECTOR/2))
        self.fc3 = nn.Linear(int(LENGTH_W2V_VECTOR/2), NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = Net()
    model.load_state_dict(torch.load(path))
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def train_model():
    print('Training model...')
    dataset = CommentDataset(mode='train')
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    # dataiter = iter(trainloader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    shutil.rmtree('logs')
    os.mkdir('logs')
    writer = SummaryWriter(log_dir='logs')
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        print(f'Number epochs - {epoch}')
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()  # zero the parameter gradients

            outputs = net(X)  # forward + backward + optimize
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 500 mini-batches
                writer.add_scalar(tag='loss', scalar_value=running_loss/500, global_step=i+trainloader.__len__()*epoch)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
    save_model(os.path.join('pretrained models', MODEL_NAME), net)
    print('Finished Training')


def evaluate_model():
    dataset = CommentDataset(mode='val')
    trainloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=4)

    net = load_model(os.path.join('pretrained models', MODEL_NAME))
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    score = dict(zip(CLASSES, [0.]*NUM_CLASSES))
    with torch.no_grad():
        for i, data in enumerate(trainloader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            outputs = net(X)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            for ax, key in enumerate(list(score.keys())):
                score[key] += roc_auc_score(y[..., ax].cpu().numpy(), outputs[..., ax].cpu().numpy())

    print(f'Evaluate loss = {running_loss/trainloader.__len__()}')
    for ax, key in enumerate(list(score.keys())):
        score[key] /= trainloader.__len__()
    print(f'ROC_AUC score = {score}')
    print(f'Mean ROC_AUC score = {np.mean(list(score.values()))}')


def make_predict():
    print("Make prediction...")
    dataset = CommentDataset(mode='test')
    trainloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=4)

    net = load_model(os.path.join('pretrained models', MODEL_NAME))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, data in enumerate(trainloader):
            X, ids = data
            X = X.to(device)
            outputs = net(X)

    out_numpy = outputs.cpu().numpy()
    predict_df = pd.DataFrame(columns=['id'] + CLASSES)
    predict_df['id'] = ids
    predict_df[CLASSES] = np.squeeze(out_numpy)
    predict_df.to_csv(os.path.join('submissions', 'sub.csv'), index=False)
    print('Finished prediction')

@click.command()
@click.option('--action', default='train', help='train/val/test')
def main(action):
    if action == 'train':
        train_model()
        gc.collect()
    elif action == 'val':
        evaluate_model()
        gc.collect()
    elif action == 'test':
        make_predict()
        gc.collect()

if __name__ == "__main__":
    main()
