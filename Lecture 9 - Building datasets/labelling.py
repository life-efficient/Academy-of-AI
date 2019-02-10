import pandas as pd
import os
from PIL import Image

data_dir = 'data/faces/'
file = data_dir + 'gender.csv'
data = pd.read_csv(file)
print(data)

img_names = data['img_name'].tolist()
print(img_names)
genders = data['gender'].tolist()
print(genders)

imgs_not_labelled = [img_name for img_name in os.listdir(data_dir)
                     if img_name not in img_names
                     and img_name[-4:] == '.jpg']
print(imgs_not_labelled)

for img_name in imgs_not_labelled:
    img_name = os.path.join(data_dir, img_name)
    print(img_name)
    img = Image.open(img_name)
    img.show()
    gender = input('Gender (m/f)')
    img_names.append(img_name)
    genders.append(gender)
    df = pd.DataFrame({'img_name': img_names, 'genders': genders})
    df.to_csv('lab.csv')
