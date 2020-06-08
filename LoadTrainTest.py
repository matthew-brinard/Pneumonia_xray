import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from NeuralNet import Net
from Functions import train_model, test_model, plot_history, class_accuracy, plot_class_accuracy

try:
    train_dir = './chest_xray/train'
except Exception as e:
    print(e)
    print('Error finding image directory.')

try:
    val_dir = './chest_xray/val'
except Exception as e:
    print(e)
    print('Error finding image directory.')

try:
    test_dir = './chest_xray/test'
except Exception as e:
    print(e)
    print('Error finding image directory.')

# Hyper parameters
num_epochs = 35
img_size = 512
batch_size = 16
save_model = False
model_path = './Net.pth'


# Image transforms
# Uses data augmentations to increase the size of the training image set.
image_transforms = {
    'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=720, scale=(0.95, 1.0)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    'test':
        transforms.Compose([
            transforms.Resize(size=img_size),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
}

train_set = datasets.ImageFolder(root=train_dir, transform=image_transforms['train'])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

val_set = datasets.ImageFolder(root=val_dir, transform=image_transforms['test'])
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

test_set = datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Place train and validation datasets into a dictionary
dataloaders_dict = {'train': train_loader, 'val': val_loader}

# Initialize the neural network and send it to the GPU if available
Net = Net()
Net.to('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = SGD(Net.parameters(), lr=0.001, momentum=.92)

# Train the model
Net, t_hist, v_hist = train_model(Net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

# Test the model on the test data set and print results
test_accuracy = test_model(Net, test_loader)
print('\nThe trained neural net model has an accuracy of {:,.2f}%'.format(test_accuracy * 100), 'on the test dataset.\n')

# Saves model if save_model hyper parameter is set to "True"
if save_model:
    torch.save(Net.state_dict(), model_path)

# Plot per class accuracy metrics
class_data = class_accuracy(Net, test_loader, test_set)
print(class_data)
plot_class_accuracy(class_data)
# Plot the model history for training and validation accuracy
plot_history(t_hist, v_hist)
