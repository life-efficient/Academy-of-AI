import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.widgets import Cursor
import glob
import pandas as pd
import sys, os

root = 'data/faces/'
csv = 'data/genders.csv'

class Labeller():

    def __init__(self, root, csv):

        self.labels = []
        self.root = root
        self.csv = csv
        self.got_data = False

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key) # link this name to this function
        self.ax = self.fig.add_subplot(111)
        if os.path.isfile(self.csv):
            self.csv_already_exists = True
            print('The csv already exists')
            # files is a list of the full filepath from the working directory to each file in the root directory
            self.files = [file for file in glob.iglob(self.root + '*')
                          if file[len(self.root):] not in pd.read_csv(self.csv)['Filename'].tolist()]
        else:
            self.files = [file for file in glob.iglob(self.root + '*')]
            self.csv_already_exists = False

        self.files = [file for file in self.files if file[-4:] == '.jpg']
        self.files = [file[len(self.root):] for file in self.files]
        print(self.files)

        self.all_files = []

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
        try:
            self.file = self.files.pop(0)
        except IndexError:
            self.close()
        file = os.path.join(self.root, self.file)
        self.im = image.imread(file)
        print(self.im.shape)
        self.ax.imshow(self.im)
        self.label_filename = self.file[len(self.root):] # slices off the 'data' (folder) prefix and '.jpg' suffix to get the filename
        print('Image name:', self.label_filename, '\t{} files remaining'.format(len(self.files)))
        print('Label the gender of the subject by pressing m for man or w for woman')
        print('Then press enter to continue')
        print('Press escape to save and quit')

        self.image_box = [self.label_filename] # this will be appended to when landmarks are marked
        self.label = None
        self.fig.canvas.draw() # update the plot

    def finish_with_image(self):

        # change xmin, ymin, xmax, ymin -> x, y, w, h
        self.labels.append(self.label)
        self.all_files.append(self.file)
        self.got_data = True

    def on_key(self, event):
        print('\nYou pressed', event.key)

        if event.key == 'm' or event.key == 'w':
            self.label = event.key

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
            #self.all_files = [file[len(self.root):] for file in self.all_files]
            if self.csv_already_exists: # if the csv already exists then append to it
                print('The csv already exists')
                #print('old shit:', pd.read_csv(self.root + self.csv))
                old_data = pd.read_csv(self.csv)
                print(old_data)

                self.all_files += old_data['Filename'].tolist()
                self.labels += old_data['Label'].tolist()
            print(self.all_files)
            print(self.labels)
            df = pd.DataFrame({'Filename': self.all_files, 'Label': self.labels})
            df.to_csv(self.csv)
        print('Closing ImageMarker')
        sys.exit(0)


mymarker = Labeller(root=root, csv=csv)


