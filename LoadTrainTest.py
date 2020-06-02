import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from NeuralNet import Net
from Functions import train_model, test_model, plot_history


# Hyper parameters
num_epochs = 10
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

# Setup train-test split
all_data = datasets.ImageFolder(root='./chest_xray')
train_data_len = int(len(all_data)*.8)
valid_data_len = int((len(all_data) - train_data_len)/2)
test_data_len = int(len(all_data) - train_data_len - valid_data_len)
train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
train_data.dataset.transform = image_transforms['train']
val_data.dataset.transform = image_transforms['test']
test_data.dataset.transform = image_transforms['test']

# Dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

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
