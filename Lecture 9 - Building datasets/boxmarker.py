import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.widgets import Cursor
import glob
import pandas as pd
import sys, os

root = 'data/faces/'
csv = 'data/face_crops.csv'

class BoxMarker():

    def __init__(self, root, csv):

        self.bndboxes = []
        self.locations =['x', 'y', 'w', 'h']
        self.root = root
        self.csv = csv
        self.got_data = False

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key) # link this name to this function
        self.ax = self.fig.add_subplot(111)
        cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        #rec = RectangleSelector(self.ax, onselect=self.onclick, drawtype='box', useblit=True, spancoords='pixels',
        #                        interactive=True)#, minspanx=5, minspany=5)
        if os.path.isfile(self.csv):
            self.csv_already_exists = True
            print('The csv already exists')
            # files is a list of the full filepath from the working directory to each file in the root directory
            self.files = [file for file in glob.iglob(self.root + '*') if file[len(self.root):] not
                          in pd.read_csv(self.csv)['Filename'].tolist()]
        else:
            self.csv_already_exists = False
            self.files = [file for file in glob.iglob(self.root + '*')]

        if len(self.files) == 0:
            print('Make sure all of your images are in a folder called \'{}\' within this one.'.format(self.root),
            '\nIf you\'re sure they are, then you\'ve already labelled all of these images.')

        self.set_up_image()

        plt.ion()
        plt.show(block=True)

    def set_up_image(self):
        print('Moving to next image')
        plt.cla()
        print(len(self.files), 'images remaining')
        file = self.files.pop(0)
        self.im = image.imread(file)
        print(self.im.shape)
        self.ax.imshow(self.im)
        self.label_filename = file[len(self.root):] # slices off the 'data' (folder) prefix and '.jpg' suffix to get the filename
        print('Image name:', self.label_filename, '\t{} files remaining'.format(len(self.files)))
        print('Mark the top left of the box and then the bottom right')

        self.image_box = [self.label_filename] # this will be appended to when landmarks are marked

        self.fig.canvas.draw() # update the plot

    def finish_with_image(self):

        # change xmin, ymin, xmax, ymin -> x, y, w, h
        xmin, ymin, xmax, ymax = self.image_box[1:]
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w / 2
        y = ymin + h / 2
        self.image_box[1:] = [x, y, w, h]
        self.bndboxes.append(self.image_box)
        self.got_data = True

    def onclick(self, event):
        if len(self.image_box) < 4:

            print(len(self.image_box))
            x, y = float(event.xdata), float(event.ydata)   # event.x gives coordinates of x in data visualised
            print('X', x, 'Y', y)
            self.image_box.append(x)
            self.image_box.append(y)
            self.cross = plt.plot(event.xdata, event.ydata, marker='x', color='r') # xdata gives pixel coordinates
            self.fig.canvas.draw()
        else:
            print('Youre done, press enter. If youve made a mistake, press backspace and remark this image')

    def on_key(self, event):
        print('\nYou pressed', event.key)

        if event.key == 'backspace':    # use backspace to remove previously placed crosses
            plt.cla()
            self.ax.imshow(self.im)
            self.image_box = [self.label_filename]

        if event.key == 'enter' or event.key == 'ctrl+w': # press enter when youre done with an image
            self.finish_with_image()
            if len(self.files) > 0:
                print('Images remaining:', len(self.files))
                self.set_up_image()
            else:
                print('All images done')
                self.close()

        if event.key == 'escape':
            self.close()

    def close(self):
        if self.got_data:
            print('Writing collected data')
            self.bndboxes= pd.DataFrame(self.bndboxes)
            print(len(self.bndboxes))
            print(self.bndboxes)
            self.bndboxes.columns = ['Filename'] + self.locations
            self.bndboxes = self.bndboxes.set_index('Filename')
            if self.csv_already_exists: # if the csv already exists then append to it
                print('The csv already exists')
                #print('old shit:', pd.read_csv(self.root + self.csv))
                self.bndboxes = pd.read_csv(self.csv, index_col=0).append(self.bndboxes)
            self.bndboxes.to_csv(self.csv)
            print(self.bndboxes)
        print('Closing ImageMarker')
        sys.exit(0)


mymarker = BoxMarker(root=root, csv=csv)


