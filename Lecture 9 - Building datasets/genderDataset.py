import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from time import sleep

class GenderDataset(Dataset):

    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.imgs = self.data['Filename'].tolist()

        self.label_map = {'m': 0, 'w': 1}
        self.genders = self.data['Label'].tolist()
        self.genders = [self.label_map[gender] for gender in self.genders]

        self.transform = transform

    def __getitem__(self, idx):

        img_name = os.path.join(self.data_dir, self.imgs[idx])
        img = Image.open(img_name)

        gender = self.genders[idx]

        return img, gender

    def __len__(self):
        return len(self.data)

