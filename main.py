import numpy as np
from tqdm import tqdm

import nn
from model import Model
from dataset import DataLoader
from optim import SGD
from engine import Module

num_epochs = 2
train_dataset = DataLoader(train=True)
test_dataset = DataLoader(train=False)

    
fcl = Model()
optimizer = SGD(fcl, lr=1e-3)
loss_fn = nn.CrossEntropy()

for e in range(num_epochs):
    for i, (x, y_hat) in enumerate(train_dataset):
        y = x.reshape(-1, 784)
        y = fcl(y)
        loss = loss_fn(y, y_hat)
        dL = loss_fn.backward()
        optimizer.step(dL)
        optimizer.zero_grad()
        print(loss.mean())
print('done')

