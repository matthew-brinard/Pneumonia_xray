import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(in_features=256 * 12 * 12, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
        # Convolution layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 3
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 4
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Reshape tensor
        x = x.reshape(-1, 256 * 12 * 12)
        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        # Fully connected layer 2
        x = self.fc2(x)
        x = F.relu(x)
        # Output layer, because we are using cross entropy as our loss function
        # we do not need to apply a Softmax activation function.
        x = self.out(x)
        return x

