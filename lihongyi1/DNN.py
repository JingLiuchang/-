import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_analysis import COVID19Dataset,getfeature
from utils import get_device
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
train_path = './covid.train.csv'
test_path = './covid.test.csv'
my_seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(my_seed)
torch.manual_seed(my_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(my_seed)

def prepare_dataloader(path, mode, valid_rate, batch_size, jobs_num, mean, std):
    dataset = COVID19Dataset(path, mode, valid_rate, mean, std)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode=="train"), drop_last=False, num_workers=jobs_num, pin_memory=True)
    return dataloader

class NeuralNetwork(nn.Module):
    def __init__(self, features_num):
        super(NeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(features_num, 64),#features_num个神经元,64个输出
            nn.ReLU(),
            nn.Linear(64, 1)#64个神经元
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.network(x).squeeze(1)#降维

    def cal_loss(self, y, y_hat):
        loss = torch.sqrt(self.criterion(y, y_hat))
        l1 = 0
        for i in self.parameters():
            l1 += torch.sum(abs(i))
        return loss + 0.0001 * l1, loss

def train(train_dataloader,valid_dataloader,model, config, device):
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_hparas"])#设置优化方法和参数

    rmse_min = float("Inf")
    not_better_cnt = 0
    epoch = 0

    while epoch < config["epochs_num"]:
        model.train()
        train_loss = 0
        for x, y in train_dataloader:
            optimizer.zero_grad()#清空过往梯度
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            rmse_l1_loss, rmse_loss = model.cal_loss(y, y_hat)
            rmse_l1_loss.backward()#反向传播
            optimizer.step()#更新
            train_loss += rmse_loss.detach().cpu().item() * x.shape[0]
        #一次全批训练完毕
        train_loss /= len(train_dataloader.dataset)

        model.eval()
        valid_loss = 0
        for x, y in valid_dataloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():#无需反向传播，无需构建计算图
                y_hat = model(x)
                rmse_l1_loss, rmse_loss = model.cal_loss(y, y_hat)
            valid_loss += rmse_loss.detach().cpu().item() * x.shape[0]
        valid_loss /= len(valid_dataloader.dataset)

        print("Epoch: {:4d}, Train Loss: {:.4f}, Valid Loss: {:.4f}".format(epoch + 1, train_loss, valid_loss))

        if valid_loss < rmse_min:
            rmse_min = valid_loss
            torch.save(model.state_dict(), config["save_path"])
            not_better_cnt = 0
        else:
            not_better_cnt += 1

        if not_better_cnt > config["early_stop"]:
            print("Early stop at epoch {:4d}.".format(epoch + 1))
            break
        epoch += 1


def test(test_dataloader, model, device,file):
    model.eval()
    y_hats = []
    for x in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            y_hat = model(x)
            y_hats.append(y_hat.detach().cpu())
    y_hats = torch.cat(y_hats, dim=0).numpy()
    print("Saving results to {}".format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, y in enumerate(y_hats):
            writer.writerow([i, y])








if __name__=="__main__":
    train_flag=int(input())
    test_flag=int(input())
    device = get_device()
    print('device:',device)
    os.system("mkdir models")
    config = {
        "epochs_num": 20000,
        "batch_size": 270,
        "optimizer": "SGD",
        "optimizer_hparas": {
            "lr": 0.0001,
            "momentum": 0.9
        },
        "early_stop": 200,
        "save_path": "models/model.pth"
    }

    with open(train_path, "r") as fp:
        train_data = list(csv.reader(fp))
        train_data = np.array(train_data[1:])[:, 1:-1].astype(float)
    with open(test_path, "r") as fp:
        test_data = list(csv.reader(fp))
        test_data = np.array(test_data[1:])[:, 1:].astype(float)
    all_data = np.vstack([train_data, test_data])

    id = getfeature()
    mean = torch.FloatTensor(train_data[:, id]).mean(dim=0, keepdim=True)
    std = torch.FloatTensor(train_data[:, id]).std(dim=0, keepdim=True)

    train_dataloader = prepare_dataloader(train_path, "train", 0.1, config["batch_size"], 0, mean, std)
    valid_dataloader = prepare_dataloader(train_path, "valid", 0.1, config["batch_size"], 0, mean, std)
    test_dataloader = prepare_dataloader(test_path, "test", None, config["batch_size"], 0, mean, std)

    model = NeuralNetwork(train_dataloader.dataset.features_num).to(device)

    if train_flag:
        train(train_dataloader, valid_dataloader, model, config, device)
    if test_flag:
        test(test_dataloader,model,device,file="y_hat.csv")



