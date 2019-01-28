import torch
import numpy as np

class Resize():

    def __init__(self, out_size):
        assert isinstance(out_size, (tuple))
        self.output_size = out_size

    def __call__(self, example):
        img, gender = example
        new_h, new_w = self.output_size
        img = img.resize((new_h, new_w))
        return img, gender

class ToTorchTensor():

    def __call__(self, example):
        img, gender = example
        img = np.array(img) / 255
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)

        img = img.transpose((2, 0, 1))

        return torch.Tensor(img), torch.Tensor(gender)


resize = Resize((20, 20))

