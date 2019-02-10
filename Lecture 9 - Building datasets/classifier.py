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


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.features = models.vgg11(pretrained=True).features
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, units1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(units1, units2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(units2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.features(x)).reshape(-1, 512 * 7 * 7)
        x = self.regressor(x)
        return x

nn = NN()
criterion = torch.nn.BCELoss()
optimiser = torch.optim.SGD(params=nn.parameters(), lr=0.5, momentum=0.1)


def train():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    plt.ion()
    plt.show()
    losses = []
    for epoch in range(epochs):
        batch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            #print(batch)
            h = nn(x)
            #print('Prediction:', h.item(), 'Label:', y.item())     # for batch size == 1

            loss = criterion(h, y.float())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
            print()
            batch_losses.append(loss.item())
            ax.plot(losses, c='b')
            fig.canvas.draw()
        losses.append(np.mean(batch_losses))
    plt.ioff()

train()
