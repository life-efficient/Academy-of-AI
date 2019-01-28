from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models

from genderDataset import GenderDataset
import myTransforms

csv = 'data/genders.csv'
data_dir = 'data/faces'

thetransforms = []
thetransforms.append(myTransforms.Resize((224, 224)))
thetransforms.append(myTransforms.ToTorchTensor())
thetransforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
thetransforms = transforms.Compose(thetransforms)

dataset = GenderDataset(csv, data_dir, transform=thetransforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=16,
                          shuffle=True)

class NN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.features = models.vgg11(pretrained=True).features
        self.regressor = torch.nn.sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4096, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.features(x)).reshape(-1, 512 * 7 * 7)
        x = self.regressor(x)
        return x

epochs = 1

nn = NN()
criterion = torch.nn.BCELoss()
optimiser = torch.optim.SGD(params=nn.parameters(), momentum=0.9)

for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        h = nn(x)
        loss = criterion(h, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
