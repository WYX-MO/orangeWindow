import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
plt.ion()  # 启用交互式模式,用于实时显示图像

size = 28
batch_size = 64
num_classes = 10
num_epocchs = 3

train_dataset = datasets.MNIST(root='D:\\項目\\pyLearn\\pro\\mnist\\data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='D:\\項目\\pyLearn\\pro\\mnist\\data', train=False, transform=transforms.ToTensor(), download=True)

#构建batch数据
train_loder = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loder = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=5,stride = 1,padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #接全连接层
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def accuracy(self, output, target):
        _, pred = torch.max(output, 1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)

torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimiser2 = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epocchs):
    for i, (images, labels) in enumerate(train_loder):
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epocchs}], Step [{i+1}/{len(train_loder)}], Loss: {loss.item():.4f}')
            acc = net.accuracy(output, labels)
            print(f'Accuracy: {acc:.4f}')
