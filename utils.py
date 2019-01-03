import matplotlib.pyplot as plt

class LossPlot():

    def __init__(self, figsize=(10, 20)):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        self.ax.grid()
        plt.ion()
        plt.show()

    def update(self, newdata):
        self.ax.plot(newdata, marker='x', c='r')
        self.fig.canvas.draw()

class HypothesisPlot():

    def __init__(self, x, y, colours, label_dict=None, domian=None, range=None):
        self.x, self.y = x, y
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)

        print(colours)

        self.ax.scatter(x, y, c=colours)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        self.hypothesis = self.ax.plot(0, 0, c='g')         # plot returns list of artists
        print(self.hypothesis)
        print(self.hypothesis[0])
        plt.ion()
        plt.show()

    def update(self, newdata):
        x, y = newdata
        self.hypothesis[0].remove()                  # remove first artist from list of artists
        self.hypothesis = self.ax.plot(x, y, 'b')
        self.fig.canvas.draw()

    def plot(self, data, colour):
        x, y = data
        #self.ax.scatter(x, y, c=colour)
        #self.fig.canvas.draw()