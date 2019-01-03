from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
print(x)
print(y)
print(y.shape)

l
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='winter')
plt.show()