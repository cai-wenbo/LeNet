import torch
import torch.nn as nn



#  Lenet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.tanh1 = nn.Tanh()
        self.norm1 = nn.BatchNorm2d(num_features=6)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()
        self.norm2 = nn.BatchNorm2d(num_features=16)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.tanh3 = nn.Tanh()
        self.norm3 = nn.BatchNorm1d(num_features=1)
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.norm4 = nn.BatchNorm1d(num_features=1)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim = 1)
            

    def forward(self, x):
        y = self.conv1(x)
        y = self.tanh1(y)
        y = self.norm1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.tanh2(y)
        y = self.norm2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], 1, -1)
        y = self.fc1(y)
        y = self.tanh3(y)
        y = self.norm3(y)
        y = self.fc2(y)
        y = self.tanh4(y)
        y = self.norm4(y)
        y = y.view(y.shape[0], -1)
        y = self.fc3(y)
        y = self.softmax(y)
        return y
