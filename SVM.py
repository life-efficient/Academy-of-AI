import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.datasets.samples_generator import make_blobs


class SVM():

    def __init__(self, lambda_=0):
        self.w = np.random.random((2))          # weight vector - has dimensions = num features used
        self.b = np.random.random((1))          # bias
        self.lambda_ = lambda_                          # lambda
        self.lr = 0.1                           # learning rate

    '''
    def _update_weights(self, sample):           # nudge the parameterising weights in the right direction
        for s in sample:                        # for each sample
            x, y = s

            if y*(np.dot(x, self.w) - self.b) >= 1:
                grad = 2 * self.l * self.w
            else:
                grad = 2 * self.l * self.w - y * x

            self.w -= self.lr * grad
    '''
    def predict(self, batch):
        """Predict the class of each vector in the list called batch"""
        predictions = []
        for sample in batch:
            x, y = sample
            prediction = np.dot(self.w, s)
            if prediction < 0:
                prediction = -1
            else:
                prediction = 1
            predictions.append(prediction)
        return predictions

    def _loss_and_gradient(self, batch_X, batch_y):                  # estimate the gradient of the loss wrt the weights for this batch
        m = len(batch_X)
        loss = 0                                                             # initialise batch loss
        #gradient = 0                                                        # initialise batch gradient
        grad_w = 0
        grad_b = 0
        for sample in range(m):                                                 # for each sample in the minibatch
            x, y = batch_X[sample], batch_y[sample] #sample[:-1], sample[-1]                          #  unpack the batch
            #print(x)
            #print(y)
            this_loss = max([0, 1 - y * (np.dot(self.w, x) - self.b)])       # compute example loss with current weights
            loss += this_loss                                                # add this examples loss to the batch loss

            if this_loss > 0:                                                # if misclassified
                #gradient -= y * x                                            #
                grad_w -= y * x
                grad_b += y

        #loss /= m                                                  # avg per example in minibatch
        avg_example_loss = loss / m                                 # avg per example in minibatch
        weight_decay_loss = self.lambda_ * np.dot(self.w.transpose(), self.w)               # add weight decay penalty
        loss = avg_example_loss + weight_decay_loss
        #print('Avg example loss', avg_example_loss, 'Weight decay loss', weight_decay_loss)
        #        gradient /= m

        # WEIGHT GRADIENTS
        avg_example_grad_w = grad_w / m                         # average gradient for weight decay
        weight_decay_grad = 2 * self.lambda_ * self.w           # compute grad from weight decay
        grad_w = avg_example_grad_w + weight_decay_grad         # add grad from weight decay
        grad_b /= m
        #print('Avg example grad', avg_example_grad, 'Weight decay grad', weight_decay_grad)
        #print(type(loss))
        loss = float(loss)
        return loss, grad_w, grad_b

    def fit(self, epochs, batch_size, data, loss_plot, hypothesis_plot):
        X, y = data
        losses = []
        for epoch in range(epochs):
            n = len(X)
            print(y)

            #SHUFFLE
            s = np.arange(X.shape[0])       # list of example indices
            np.random.shuffle(s)            # shuffle list of example indices
            X, y = X[s], y[s]
            #np.random.shuffle(dataset)                          # shuffle order of examples

            batch_idx = 0
            while n - (batch_idx * batch_size) > 0:                         # num examples remaining in this epoch
                #print('Samples remaining', n - (batch_idx * batch_size))
                batch_idx += 1
                if n - (batch_idx * batch_size) > batch_size:          # if num examples remaining > batch_size
                    batch_X = X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    batch_y = y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:                               # if num examples remaining < batch_size
                    batch_X = X[batch_idx * batch_size: -1]
                    batch_y = y[batch_idx * batch_size: -1]
                if len(batch_X) == 0:
                    continue
                #print('Batch:', batch)
                loss, grad_w, grad_b = self._loss_and_gradient(batch_X, batch_y)
                losses.append(loss)
                #loss_plot.update(losses)
                print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss)

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

                print(self.w)
                print(self.b)
                print('\n')

                #print(max(dataset[:, 0]))
                #print(min(dataset[:, 0]))
                domain = np.linspace(0, 8)
                #domain = np.linspace(min(dataset[:, 0], max(dataset[:, 0])))
                hypothesis_plot.update((domain, (self.b / self.w[1]) - ((self.w[0] / self.w[1]) * domain)))
                #print((self.w[0] / self.w[1]) * domain)

            loss_plot.update(losses)

    def show_decision_boundary(self, ax, newdata):
        ax.plot(newdata)

'''
# GET DATA
data = pd.read_csv(file)
features = np.array(data.iloc[:99, 1:3])
label_dict = {'Iris-setosa': 1, 'Iris-versicolor': -1, 'Iris-virginica': 'b'}    # create specie -> colour char dict
labels = np.array([label_dict[data.iloc[i, -1]] for i in range(99)])           # convert first 99 (2 classes) labels into +-1
labels.resize((99, 1))
dataset = np.hstack((features, labels))         # np array of examples
print(dataset)
'''

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)

lambda_ = 0               # lambda to weight weight decay loss        # set this equal to zero to train unregularised
#kernel = 'linear'       # how do we want to transform the data

label_dict = {1: 'g', -1: 'r'}

#
myLossPlot = LossPlot()                     #
#myHypothesisPlot = HypothesisPlot((dataset[:50, 0], dataset[:50, 1]))           # initialise plot to show data and hypothesis
myHypothesisPlot = HypothesisPlot(X[:, 0], X[:, 1], colours=y)           # initialise plot to show data and hypothesis

#myHypothesisPlot.plot((dataset[50:, 0], dataset[50:, 1]), colour='r')           # add the second class's data in a red
mySVM = SVM(lambda_=lambda_)
mySVM.fit(epochs=100, batch_size=16, data=(X, y), loss_plot=myLossPlot, hypothesis_plot=myHypothesisPlot)
