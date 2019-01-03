import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.datasets.samples_generator import make_blobs

class SVM():

    def __init__(self, lambda_=0, kernel):
        self.alpha = np.random.random((2))          # weight vector - has dimensions = num features used
        self.b = np.random.random((1))          # bias
        self.lambda_ = lambda_                          # lambda
        self.lr = 0.1                           # learning rate
		self.k = kernel

    def forward(self, batch, support_coeffs, support_vectors):
        """Predict the class of each vector in the supplied batch"""
		print(X.shape())
		print(X[0].shape())
        predictions = []
        for sample in batch:
            x, y = sample
			
			prediction = self.b				# initialise prediction with bias
			for idx in support_idxs:		# get attribution from each support
				prediction += self.alpha[idx] * k(X[idx], x)	# add to prediction
            prediction = np.sign(np.dot(self.w, s) + self.b)	# hypothesis = sign

            predictions.append(prediction)
        return predictions

	def get_support(X):
		support_idxs = [idx for idx, is_support in enumerate(self.alpha > self.threshold) if is_support]
		support_coeffs = self.alpha[support_idxs]
		support_vectors = X[support_idxs]
		return support_coeffs, support_vectors

    def _loss_and_gradient(self, batch_X, batch_y):                  # estimate the gradient of the loss wrt the weights for this batch
        m = len(batch_X)
        loss = 0                                                             # initialise batch loss
        grad_alpha = 0
        grad_b = 0
        for sample in range(m):                                                 # for each sample in the minibatch
            x, y = batch_X[sample], batch_y[sample] #sample[:-1], sample[-1]                          #  unpack the batch
            #print(x)
            #print(y)
            this_loss = max([0, 1 - y * (np.dot(self.w, x) + self.b)])       # compute example loss with current weights
            loss += this_loss                                                # add this examples loss to the batch loss

            if this_loss > 0:                                                # if misclassified
                grad_w -= y * x
                grad_b -= y

        avg_example_loss = loss / m                                 # avg per example in minibatch
        weight_decay_loss = self.lambda_ * np.dot(self.w.transpose(), self.w)               # add weight decay penalty
        loss = avg_example_loss + weight_decay_loss

        # GRADIENTS
        avg_example_grad_w = grad_w / m                         # average gradient for weight decay
        weight_decay_grad = 2 * self.lambda_ * self.w           # compute grad from weight decay
        grad_w = avg_example_grad_w + weight_decay_grad         # add grad from weight decay
        grad_b /= m
        print('grad_w', grad_w, 'grad_b', grad_b, 'weight_decay_grad', weight_decay_grad)
        print(type(loss))
        loss = float(loss)
        return loss, grad_w, grad_b

    def fit(self, epochs, batch_size, data, loss_plot, hypothesis_plot, threshold):
        X, y = data
		
        m = len(X)		
		self.alpha = np.random.randn()
        losses = []
        for epoch in range(epochs):
            print(y)

            #SHUFFLE
            s = np.arange(m)       # list of example indices
            np.random.shuffle(s)            # shuffle list of example indices
            X, y = X[s], y[s]

			# convert all this list shit into numpy arrays
            batch_idx = 0
            while m - (batch_idx * batch_size) > 0:                         # num examples remaining in this epoch
                print('Samples remaining', m - (batch_idx * batch_size))
                if m - (batch_idx * batch_size) > batch_size:          # if num examples remaining > batch_size
                    batch_X = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    batch_y = y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:                               # if num examples remaining < batch_size
                    batch_X = X[batch_idx * batch_size: -1]
                    batch_y = y[batch_idx * batch_size: -1]
                if len(batch_X) == 0:
                    continue
                batch_idx += 1

                #print('Batch:', batch)
                loss, grad_alpha, grad_b = self._loss_and_gradient(batch_X, batch_y)
                losses.append(loss)
                #loss_plot.update(losses)
                print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss)

                self.alpha -= self.lr * grad_alpha
                self.b -= self.lr * grad_b

                print(self.w)
                print(self.b)
                print('\n')

                #print(max(dataset[:, 0]))
                #print(min(dataset[:, 0]))
                domain = np.linspace(0, 8)
                #domain = np.linspace(min(dataset[:, 0], max(dataset[:, 0])))
                hypothesis_plot.update((domain, (- self.b / self.w[1]) - ((self.w[0] / self.w[1]) * domain)))
                #print((self.w[0] / self.w[1]) * domain)

            loss_plot.update(losses)

	return self.support_vectors

class LinearKernel():
	
	def __call__(x1, x2):
		return np.dot(x1.transpose(), x2)

class PolynomialKernel():

	def __init__(order):
		self.order = order

	def __call__(x1, x2):
		return(1 + np.dot(x1.transpose(), x2)) ** order

class RBFKernel():

	def __call__():
		return np.exp( - np.power( (x1 - x2) , 2))
   

# MAKE DATA
X, y = make_blobs(n_samples=6, centers=2, random_state=0, cluster_std=0.6)
y[y == 0] = -1					# make labels -1 or +1 (not 0 and 1)

lambda_ = 0               # lambda to weight weight decay loss        # set this equal to zero to train unregularised
kernel = LinearKernel()       # how do we want to transform the data

# SET UP PLOTS
myLossPlot = LossPlot()                     
myHypothesisPlot = HypothesisPlot(X[:, 0], X[:, 1], colours=y)           # initialise plot to show data and hypothesis

mySVM = SVM(lambda_=lambda_, kernel)
mySVM.fit(epochs=20, batch_size=16, data=(X, y), loss_plot=myLossPlot, hypothesis_plot=myHypothesisPlot, support_threshold=0.1)
