from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from genderDataset import GenderDataset
import myTransforms

csv = 'data/genders.csv'
data_dir = 'data/faces'

thetransforms = []
thetransforms.append(myTransforms.Resize((224, 224)))
thetransforms.append(myTransforms.ToTorchTensor())
#thetransforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225]))
thetransforms = transforms.Compose(thetransforms)

dataset = GenderDataset(csv, data_dir, transform=thetransforms)

'''
img, g = dataset[10]
print(img)
print(img.shape)
img = transforms.ToPILImage()(img)
img.show()
print(g)

k
'''

epochs = 10
batch_size = 2
units1 = 128
units2 = 32
dropout_p = 0.5
m = len(dataset)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=m)

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.features = models.vgg11(pretrained=True).features
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, units1),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(units1, units2),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(units2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.features(x)).reshape(-1, 512 * 7 * 7)
        x = self.regressor(x)
        return x

nn = NN()
criterion = torch.nn.BCELoss(reduction='mean')
optimiser = torch.optim.SGD(params=nn.parameters(), lr=0.7, momentum=0.1)


def train(model, fig, ax, idx):
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        #print(batch)
        h = model(x)
        #print('Prediction:', h.item(), 'Label:', y.item())     # for batch size == 1

        loss = criterion(h, y.float())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        #print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
        #print()
        #batch_losses.append(loss.item())
        ax.scatter(idx, loss.item(), c='b', s=5)
        #losses.append(np.mean(batch_losses))
        fig.canvas.draw()
        idx += 1
    return idx


def test(model, fig, ax, idx):
    for batch_idx, batch in enumerate(val_loader):
        x, y = batch
        h = model(x)
        loss = criterion(h, y.float())
        ax.scatter(idx, loss.item(), c='r')
        fig.canvas.draw()
    pass

def execute(model, epochs=epochs):

    # SET UP PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    plt.ion()
    plt.show()

    losses = []
    batch_losses = []

    idx = 0

    for epoch in range(epochs):
        train(model, fig, ax, idx)
        losses.append()
        print(idx)
        test(model, fig, ax, idx)


    plt.ioff()

execute(nn)

import time
torch.save(nn.state_dict(), str(time.time()))
