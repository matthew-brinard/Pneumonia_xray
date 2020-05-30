import os
import random
import shutil

try:
    train_NORMAL_dir = './chest_xray/train/NORMAL'
except Exception as e:
    print(e)
    print('Error finding image directory.')

try:
    train_PNEUMONIA_dir = './chest_xray/train/PNEUMONIA'
except Exception as e:
    print(e)
    print('Error finding image directory.')

# If validation dataset has not been created then make one
if os.path.exists('./chest_xray/val'):
    print('The validation dataset folder already exists.')
else:
    os.mkdir('./chest_xray/val')
    os.mkdir('./chest_xray/val/NORMAL')
    os.mkdir('./chest_xray/val/PNEUMONIA')
    # Take ten percent of each of the classes images
    count_normal = int(len(os.listdir(train_NORMAL_dir)) * 0.1)
    count_pneumonia = int(len(os.listdir(train_PNEUMONIA_dir)) * 0.1)
    # Move training images to the validation folders.
    for file in random.sample(os.listdir(train_NORMAL_dir), count_normal):
        path_normal = os.path.join(train_NORMAL_dir, file)
        shutil.move(path_normal, './chest_xray/val/NORMAL')
    print("The validation dataset for the class 'NORMAL' now has {} images.".format(
        len(os.listdir('./chest_xray/val/NORMAL'))))
    for file in random.sample(os.listdir(train_PNEUMONIA_dir), count_pneumonia):
        path_pneumonia = os.path.join(train_PNEUMONIA_dir, file)
        shutil.move(path_pneumonia, './chest_xray/val/PNEUMONIA')
    print("The validation dataset for the class 'PNEUMONIA' now has {} images.".format(
        (len(os.listdir('./chest_xray/val/PNEUMONIA')))))
