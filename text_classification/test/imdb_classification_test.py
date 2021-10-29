# coding=utf-8
import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.enabled = False
from src.models.TextCNN import TextCNN, TextCNNConfig
from src.models.RnnModel import RnnConfig, RnnModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class IMDBDataset(Dataset):
    def __init__(self, path='imdb.npz', num_words=None, skip_top=0, maxlen=None,
                 seed=113, start_char=1, oov_char=2, index_from=3, type="train", valid_size=0.2, random_state=0):
        assert type in ["train", "valid", "test"]


        # 套娃，封装Keras IMDB数据读取
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(path=path,
                                                                                num_words=num_words, skip_top=skip_top, maxlen=maxlen,
                                                                                seed=seed, start_char=start_char, oov_char=oov_char, index_from=index_from)


        if type == "test":
            x_test = [np.pad(x, (maxlen - len(x), 0), "constant", constant_values=(0, 0)) for x in x_test]
            self.x, self.y = [torch.tensor(x) for x in x_test], torch.tensor(y_test)
            return

        x_train = [np.pad(x, (maxlen - len(x), 0), "constant", constant_values=(0, 0)) for x in x_train]
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

        if type == "train":
            self.x, self.y = [torch.from_numpy(x) for x in x_train], torch.from_numpy(y_train)
        if type == "valid":
            self.x, self.y = [torch.from_numpy(x) for x in x_valid], torch.from_numpy(y_valid)



    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return len(self.x)


def train(model, optimizer, criterion, num_epochs, train_loader, valid_loader):
    model.train()
    loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    tqdm_iterator = tqdm.tqdm(range(num_epochs), maxinterval=10, mininterval=2, ncols=120, unit='epoch', bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]', nrows=10,smoothing=0.1)
    for epoch in tqdm_iterator:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss_list.append(loss.cpu().detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        valid_acc, valid_loss = valid(model, valid_loader, criterion)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)
        tqdm_iterator.set_description('epoch %d' %epoch)
        tqdm_iterator.set_postfix_str('train_loss={:^7.3f}, valid acc={:^7.3f}'.format(loss, valid_acc))
    return loss_list, valid_acc_list, valid_loss_list

def valid(model, valid_loader, criterion):# -> tuple(float, float):
    correct = 0
    total = 0
    loss_list = []
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss_list.append(criterion(y_hat, y).cpu().detach().item())
            correct += torch.sum((y == y_hat.max(1)[1]))
            total += y.shape[0]
    # return accuracy and mean loss
    return correct / total, torch.mean(torch.tensor(loss_list)).item()


def test(model, test_loader, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    measure = []
    y_hat_list = []
    y_list = []
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_loader):
            x, y = x.to(device), y
            y_hat = torch.softmax(model(x), dim=1)
            y_hat_list.append((y_hat[:, 1]).cpu())
            y_list.append(y == 1)
    # 二分类计算，多分类后面再改
    for threshold in thresholds:
        y = torch.cat(y_list)
        y_hat = torch.cat(y_hat_list, 0) >= threshold
        accuracy = accuracy_score(y, y_hat)
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        measure.append([accuracy, precision, recall, f1])

    measure = np.array(measure).T
    return measure  # 例： measure[0] 为各阈值下的accuracy值


if __name__ == '__main__':

    train_loader = DataLoader(dataset=IMDBDataset(type="train", num_words=20000, maxlen=512), batch_size=128,
                              shuffle=True)
    valid_loader = DataLoader(dataset=IMDBDataset(type="valid", num_words=20000, maxlen=512), batch_size=128,
                              shuffle=False)
    test_loader = DataLoader(dataset=IMDBDataset(type="test", num_words=20000, maxlen=512), batch_size=128,
                             shuffle=False)

    measure_list = []
    for i in range(5):
        # config = TextCNNConfig(n_vocab=20000, n_embed=300, num_classes=2, embedding_pretrained=None, num_filters=600, filter_sizes=[3, 4, 5], dropout=0.2)
        # model = TextCNN(config).to(device)

        config = RnnConfig(20000, 300, 2, None, hidden_size=512,  layer_num=1, bidirectional=True)
        model = RnnModel(config).to(device)

        num_epochs = 5
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        train(model, optimizer, criterion, 5, train_loader, valid_loader)

        measure = test(model, test_loader)
        measure_list.append(measure)
        plt.xkcd()
        measrue_name = ["accuracy", "precision", "recall", "f1 score"]
        for i in range(len(measure)):
            plt.plot([0.5, 0.6, 0.7, 0.8, 0.9], measure[i], label=measrue_name[i])
        plt.legend()
        plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        plt.xlabel("threshold")
        plt.title("test result")
        plt.show()