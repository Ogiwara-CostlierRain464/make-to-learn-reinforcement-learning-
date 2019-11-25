import torch
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

mnist = fetch_openml("mnist_784", data_home=".", cache=True)

X = mnist.data / 255
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=1/7, random_state=0
)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = y_train.astype(np.float32)
y_train = torch.LongTensor(y_train)
y_test = y_test.astype(np.float32)
y_test = torch.LongTensor(y_test)

ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

model = nn.Sequential()
# 28*28*1 to 100
model.add_module("fc1", nn.Linear(28*28*1, 100))
model.add_module("relu1", nn.ReLU())
model.add_module("fc2 ", nn.Linear(100, 100))
model.add_module("relu2", nn.ReLU())
model.add_module("fc3", nn.Linear(100, 10))

print(model)

# 重みを学習する際の最適化手法
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

