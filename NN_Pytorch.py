"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@desc: Use Neural Networks with Pytorch
"""
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn

# Set parameters
EPOCH = 10000
BATCH_SIZE = 5


class Net(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# Build Dataset
source_path = './NN_data/in_data/'
target_path = './NN_data/out_data/'
train_data, test_data = [], []
test_ratio = 1.
sta_test = ['000', '010', '027', '037', '044', '056']
for data_id in sta_test:
    test_data.append([source_path + data_id + '.npy', target_path + data_id + '_out.npy'])
for each in os.listdir(target_path):
    match = re.search(pattern=r'\d{3}_out.npy', string=each)
    if match:
        data_id = match.group(0)[:3]
        if np.random.randn() < test_ratio and data_id not in sta_test:
            train_data.append([source_path + data_id + '.npy', target_path + data_id + '_out.npy'])
        else:
            test_data.append([source_path + data_id + '.npy', target_path + data_id + '_out.npy'])

x_train, y_train = np.array([]), np.array([])
for source, target in train_data:
    x_train = np.append(x_train, np.load(source).reshape(-1, 100))
    y_train = np.append(y_train, np.load(target).reshape(-1, 100))
x, y = torch.from_numpy(x_train).type(torch.float32).reshape(-1, 100), \
       torch.from_numpy(y_train).type(torch.float32).reshape(-1, 100)
train_dataset = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

x_test, y_test = np.array([]), np.array([])
for source, target in test_data:
    x_test = np.append(x_test, np.load(source).reshape(-1, 100))
    y_test = np.append(y_test, np.load(target).reshape(-1, 100))
x, y = torch.from_numpy(x_test).type(torch.float32).reshape(-1, 100), \
       torch.from_numpy(y_test).type(torch.float32).reshape(-1, 100)
test_dataset = Data.TensorDataset(x, y)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=len(test_data),
    shuffle=True,
)

model = Net(in_dim=100, n_hidden_1=300, n_hidden_2=500, out_dim=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_func = torch.nn.MultiLabelSoftMarginLoss()
acc_saver, time_saver = [], []

# Training part
for epoch in range(EPOCH):
    start_time = time.time()
    for step, (batch_x, batch_y) in enumerate(train_loader):
        y = model.forward(batch_x)
        loss = loss_func(y, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    time_saver.append(round(end_time - start_time, 6) * 1e3)

    if epoch % 100 == 0:
        print("Epoch: {}".format(epoch))
    #     for _, (test_x, test_y) in enumerate(test_loader):
    #         y = model.forward(test_x)
    #         k_num = []
    #         for each in test_x:
    #             k_num.append((torch.nonzero(each).size()[0] // 25) + 1)
    #         out, out_ = [], []
    #         for i in range(len(k_num)):
    #             out.append(torch.topk(y[i], k_num[i], largest=True, sorted=True, out=None)[1])
    #             out_.append(torch.topk(test_y[i], k_num[i], largest=True, sorted=True, out=None)[1])
    #         acc = 0
    #         for i in range(len(out)):
    #             for j in range(out_[i].size()[0]):
    #                 if out[i][j] in out_[i]:
    #                     acc += 1
    #         acc_ratio = round(acc / sum(k_num), 4)
    #         acc_saver.append(acc_ratio)
    #         print("Epoch: {} Accuracy: {}".format(epoch, acc_ratio))

# Plot part
# plt.plot(acc_saver)
plt.plot(time_saver)
plt.show()

# Save part
# save = pd.DataFrame(acc_saver)
# save.to_csv("./logs/acc_.csv")
save = pd.DataFrame(time_saver)
save.to_csv("./logs/time.csv")
