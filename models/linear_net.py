import torch
import torch.nn as nn


class LinearNet(nn.Module):
    # 线性神经网络，输入 2D 切片
    def __init__(self, in_channels=1, num_classes=3, hidden=256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * 64 * 64, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes * 64 * 64)
        self.num_classes = num_classes

    def forward(self, x):
        b = x.shape[0]
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(b, self.num_classes, 64, 64)
