import os
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

try:
    test_NORMAL_dir = './chest_xray/test/NORMAL'
except Exception as e:
    print(e)
    print('Error finding image directory.')

try:
    test_PNEUMONIA_dir = './chest_xray/test/PNEUMONIA'
except Exception as e:
    print(e)
    print('Error finding image directory.')

# If master image folders do not exist, make them and populate them.
if os.path.exists('./chest_xray/NORMAL') and os.path.exists('./chest_xray/PNEUMONIA'):
    print('The datasets have already been created.')
else:
    os.mkdir('./chest_xray/NORMAL')
    os.mkdir('./chest_xray/PNEUMONIA')
    for image in os.listdir(train_NORMAL_dir):
        path_normal = os.path.join(train_NORMAL_dir, image)
        shutil.move(path_normal, './chest_xray/NORMAL')
    for image in os.listdir(test_NORMAL_dir):
        path_normal = os.path.join(test_NORMAL_dir, image)
        shutil.move(path_normal, './chest_xray/NORMAL')
    for image in os.listdir(train_PNEUMONIA_dir):
        path_pneumonia = os.path.join(train_PNEUMONIA_dir, image)
        shutil.move(path_pneumonia, './chest_xray/PNEUMONIA')
    for image in os.listdir(test_PNEUMONIA_dir):
        path_pneumonia = os.path.join(test_PNEUMONIA_dir, image)
        shutil.move(path_pneumonia, './chest_xray/PNEUMONIA')
    print('The image folder for the class "NORMAL" now has {} images.'.format(
        len(os.listdir('./chest_xray/NORMAL'))))
    print('The image folder for the class "PNEUMONIA" now has {} images.'.format(
        len(os.listdir('./chest_xray/PNEUMONIA'))))
    # Remove unneeded files
    os.rmdir(train_NORMAL_dir)
    os.rmdir(train_PNEUMONIA_dir)
    os.rmdir(test_NORMAL_dir)
    os.rmdir(test_PNEUMONIA_dir)
    directory_list = ['./chest_xray/train/', './chest_xray/test/']
    for directory in directory_list:
        file = os.path.join(directory, ".DS_STORE")
        os.remove(file)
    os.rmdir('./chest_xray/train')
    os.rmdir('./chest_xray/test')