import os
import random
import shutil

random.seed(0)

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

if os.path.exists('./chest_xray/val'):
    print('The validation dataset already exists.')
else:
    # Take 10% of each class (rounded to the nearest batch) from the training dataset and create the validation dataset.
    os.mkdir('./chest_xray/val')
    os.mkdir('./chest_xray/val/NORMAL')
    os.mkdir('./chest_xray/val/PNEUMONIA')

    train_normal = os.listdir(train_NORMAL_dir)
    num_normal_batches = int((len(train_normal)*0.1)/16)
    for file in random.sample(train_normal, (16 * num_normal_batches)):
        path_normal = os.path.join(train_NORMAL_dir, file)
        shutil.move(path_normal, './chest_xray/val/NORMAL')

    train_pneumonia = os.listdir(train_PNEUMONIA_dir)
    num_pneumonia_batches = int((len(train_pneumonia) * 0.1) / 16)
    for file in random.sample(train_pneumonia, (16 * num_pneumonia_batches)):
        path_pneumonia = os.path.join(train_PNEUMONIA_dir, file)
        shutil.move(path_pneumonia, './chest_xray/val/PNEUMONIA')

    print('The validation dataset for the class "NORMAL" now contains {} images.'.format(
        len(os.listdir('./chest_xray/val/NORMAL'))))
    print('The validation dataset for the class "PNEUMONIA" now contains {} images.'.format(
        len(os.listdir('./chest_xray/val/PNEUMONIA'))
    ))
