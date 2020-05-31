import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from NeuralNet import Net
from Functions import train_model, test_model, plot_history

try:
    train_dir = './chest_xray/train'
except Exception as e:
    print(e)
    print("Error finding training image directory.")

try:
    val_dir = './chest_xray/val'
except Exception as e:
    print(e)
    print("Error finding validation image directory.")

try:
    test_dir = './chest_xray/test'
except Exception as e:
    print(e)
    print("Error finding test image directory.")

# Hyper parameters
num_epochs = 25
img_size = 224
batch_size = 16
save_model = False
model_path = './Net.pth'

# Image transforms
# Uses data augmentations to increase the size of the training image set.
image_transforms = {
    'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
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

# Dataloaders
train_set = datasets.ImageFolder(root=train_dir, transform=image_transforms['train'])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = datasets.ImageFolder(root=val_dir, transform=image_transforms['train'])
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

test_set = datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Place train and validation datasets into a dictionary
dataloaders_dict = {'train': train_loader, 'val': val_loader}
# Initialize the neural network and send it to the GPU if available
Net = Net()
Net.to('cuda:0' if torch.cuda.is_available() else 'cpu')
# Initialize the loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = SGD(Net.parameters(), lr=0.002, momentum=.92)
# Train the model
Net, t_hist, v_hist = train_model(Net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
# Test the model on the test data set and print results
test_accuracy = test_model(Net, test_loader)
print('\nThe trained neural net model has an accuracy of {:,.2f}%'.format(test_accuracy * 100), 'on the test dataset.')
# Saves model if save_model hyper parameter is set to "True"
if save_model:
    torch.save(Net.state_dict(), model_path)

# Plot the model history for training and validation accuracy
plot_history(t_hist, v_hist)
