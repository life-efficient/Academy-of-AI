import numpy as np

import matplotlib.pyplot as plt

def relu(x):
    return max(0, x)

class NN():
    """Neural network with 2 hidden layers"""

    def __init__(self, l1_width=3, l2_width=2):

        self.w1 = np.random.randn(l1_width, 1)
        self.b1 = np.random.randn(l1_width, 1)

        self.w2 = np.random.randn(l2_width, l1_width)
        self.b2 = np.random.randn(l2_width, l1_width)
        
        self.w3 = np.random.randn(1, l2_width)
        self.b3 = np.random.randn(1, l2_width)


    def forward(self, x):
        print(self.w1)
        print(self.b1)
        x = relu(self.w1 * x + self.b1)
        x = relu(self.w2 * x + self.b2)
        x = self.w3 * x + self.b3           # no activation here
        return x

    def step(self, x):

nn = NN()
nn.forward(x=5)