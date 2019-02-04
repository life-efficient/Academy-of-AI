import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np
import random

m = 50
x = np.linspace(-10, 10, m)
print(x)

y = 3*x**2 + 0.1 * x ** 3 + 30*np.cos(x) + 30*np.random.rand((m))
#y = x**2 + 10*np.sin(x)
#y = x**2
#y = 3 * x + 10

class Dataset(torch.utils.data.Dataset):

    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        #if self.transform:
         #   x, y = self.transform(x, y)
        return x, y

    def __len__(self):
        return len(x)

epochs = 50
batch_size = 1
lr = 0.000001

dataset = Dataset(x, y)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=m)


units1 = 5
units2 = 5
units3 = 5

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(1, units1),
            torch.nn.ReLU(),
            torch.nn.Linear(units1, units1),
            torch.nn.ReLU(),
            torch.nn.Linear(units1, units1),
            torch.nn.ReLU(),
            torch.nn.Linear(units1, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

nn = NN().double()

criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.SGD(params=nn.parameters(), lr=lr, momentum=0.9, weight_decay=0)

def train(model, epochs=epochs):
    train_loss = []
    val_loss = []

    fig = plt.figure(figsize=(10, 20))

    h_ax = fig.add_subplot(121)
    h_ax.grid()
    h_ax.scatter(x, y, s=5)

    loss_ax = fig.add_subplot(122)
    loss_ax.grid()

    plt.ion()
    plt.show()

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_dataloader):

            batch_losses = []
            features, labels = batch
            features = features.unsqueeze(0)
            features = features.transpose(1, 0)
            prediction = model(features)
            #print('Prediction:', prediction)
            loss = criterion(prediction, labels)
            batch_losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print()
            print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
            if batch_idx % 10 == 0:
                train_loss.append(np.mean(batch_losses))
        for batch in val_dataloader:
            features, labels = batch
            features = features.unsqueeze(0)
            features = features.transpose(1, 0)
            prediction = model(features)
            val_loss.append(criterion(prediction, labels))
            loss_ax.plot(train_loss, c='b')
        fig.canvas.draw()



train(nn)
#plt.ion()
#plt.show()
plt.close()
plt.ioff()
plt.scatter(x, y, s=5)

nn.eval()

x = torch.Tensor(x)
x = x.unsqueeze(0)
x = x.transpose(1, 0)

predictions = nn(x.double())
print()
print(x)
print(predictions)
print()
x = x.detach().numpy()
predictions = predictions.detach().numpy()
plt.plot(x, predictions, c='r')

print()
for x, y in val_dataloader:
    print(x)
    print(y)
    plt.scatter(x, y, c='purple', marker='x')

plt.show()
