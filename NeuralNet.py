import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=512)

        self.fc1 = nn.Linear(in_features=512 * 6 * 6, out_features=5000)
        self.fc2 = nn.Linear(in_features=5000, out_features=500)
        self.out = nn.Linear(in_features=500, out_features=2)

    def forward(self, x):
        # Convolution layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Convolution layer 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # Reshape tensor
        x = x.reshape(-1, 512 * 6 * 6)
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
