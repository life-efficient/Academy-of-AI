import torch
import torchvision
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lr = 0.1
epochs = 100
batch_size = 4
steps = 100

X, y = sklearn.datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=4, random_state=1)
#X, y = sklearn.datasets.make_moons(n_samples=100, noise=0, random_state=1)
#X, y = sklearn.datasets.make_circles(n_samples=100, random_state=1)

x1 = X[:, 0]
x2 = X[:, 1]
print(X.shape)

#plt.ion()
#plt.scatter(x1, x2, c=y)
#plt.show()


class Dataset():

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, idx):
        return X[idx], y[idx]

    def __len__(self):
        return X.shape[0]


data = Dataset(X, y)

dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)


class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(2, 2)
        self.l3 = torch.nn.Linear(2, 2)
        self.l4 = torch.nn.Linear(2, 2)
        self.l5 = torch.nn.Linear(2, 1)
        self.g = torch.nn.Sigmoid()

    def forward(self, x):
        features = []
        features.append(self.l1(x))
        features.append(self.g(features[-1]))

        # removeable layers
        features.append(self.l2(features[-1]))
        features.append(self.g(features[-1]))
        features.append(self.l3(features[-1]))
        features.append(self.g(features[-1]))
        features.append(self.l4(features[-1]))
        features.append(self.g(features[-1]))

        features.append(self.l5(features[-1]))
        features.append(self.g(features[-1]))

        return features

nn = NN().double()

optimiser = torch.optim.SGD(nn.parameters(), lr=lr)
criterion = torch.nn.BCELoss()

def train(model, epochs=epochs):
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            #print(x.data, y.data)

            optimiser.zero_grad()

            y_hat = model(x)[-1]
            #print(type(y_hat), type(y))
            loss = criterion(y_hat, y.double())
            print('Epoch:', epoch, '\tBatch idx:', batch_idx, 'Loss:', loss.data.item())
            #print('\tPrediction:', y_hat.data.numpy(), '\tLabel:', y.data)
            loss.backward()
            optimiser.step()

train(nn)
'''
a3 = nn(torch.tensor(X))
a3 = a3[-3].data.numpy()
print(a3)
plt.scatter(a3[:, 0], a3[:, 1], c=y)
plt.show()
'''

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
plt.show()
from time import sleep

datas = nn(torch.tensor(X))     # get transformed features

for idx in range(len(datas)):   #

    #print('idx:', idx)
    if idx % 2 == 0:
        print('Stretching and shifting')
    else:
        print('Activating')

    if idx >= len(datas) - 2:
        break

    data = datas[idx].data.numpy()
    next_data = datas[idx + 1].data.numpy()

    for step in range(steps):
        plt.cla()
        now = data + step * (next_data - data) / steps

        #if len(now.shape) == 2:
        #    now = np.c_[now, np.zeros(now.shape[0])]

        ax.scatter(now[:, 0], now[:, 1], c=y)
        fig.canvas.draw()


