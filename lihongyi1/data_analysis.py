import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
# 讀取測資

def getfeature():
    train_path = './covid.train.csv'
    test_path = './covid.test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    corr = train_data.iloc[:, 41:].corr().iloc[-1]
    features = corr[abs(corr) > 0.45]
    features_col = features.index.to_list()[:-1]
    features_id = np.array([train_data.columns.to_list().index(i) for i in features_col]) - 1
    return features_id

class COVID19Dataset(Dataset):
    def __init__(self, path, mode, valid_rate, mean, std):
        with open(path, "r") as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)#去头去id
        choose_features = []
        choose_features.extend(getfeature())
        if mode == "test":
            self.x = torch.FloatTensor(data[:, choose_features])
        else:
            train_indices, valid_indices = train_test_split([i for i in range(data.shape[0])], test_size=valid_rate,
                                                            random_state=1)
            #训练验证分割
            if mode == "train":
                self.x = torch.FloatTensor(data[train_indices, :])
                self.y = torch.FloatTensor(data[train_indices, 93])
            elif mode == "valid":
                self.x = torch.FloatTensor(data[valid_indices, :])
                self.y = torch.FloatTensor(data[valid_indices, 93])
            self.x = self.x[:, choose_features]

        self.mean = mean
        self.std = std
        self.x[:, :] = (self.x[:, :] - self.mean) / self.std
        self.mode = mode
        self.features_num = self.x.shape[1]

    def __getitem__(self, index):
        if self.mode in ["train", "valid"]:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

    def __len__(self):
        return len(self.x)

if __name__=='__main__':
    getfeature()